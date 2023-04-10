import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import MinMaxScaler
# from collections import defaultdict
from scipy.linalg import block_diag
import time
# from __future__ import division
class LaneTracker:
    def __init__(self, n_lanes, proc_noise_scale, meas_noise_scale, process_cov_parallel=0, proc_noise_type='white'):
        self.n_lanes = n_lanes
        self.meas_size = 4 * self.n_lanes
        self.state_size = self.meas_size * 2
        self.contr_size = 0

        self.kf = cv2.KalmanFilter(self.state_size, self.meas_size, self.contr_size)
        self.kf.transitionMatrix = np.eye(self.state_size, dtype=np.float32)
        self.kf.measurementMatrix = np.zeros((self.meas_size, self.state_size), np.float32)
        for i in range(self.meas_size):
            self.kf.measurementMatrix[i, i*2] = 1

        if proc_noise_type == 'white':
            block = np.matrix([[0.25, 0.5],
                               [0.5, 1.]], dtype=np.float32)
            self.kf.processNoiseCov = block_diag(*([block] * self.meas_size)) * proc_noise_scale
        if proc_noise_type == 'identity':
            self.kf.processNoiseCov = np.eye(self.state_size, dtype=np.float32) * proc_noise_scale
        for i in range(0, self.meas_size, 2):
            for j in range(1, self.n_lanes):
                self.kf.processNoiseCov[i, i+(j*8)] = process_cov_parallel
                self.kf.processNoiseCov[i+(j*8), i] = process_cov_parallel

        self.kf.measurementNoiseCov = np.eye(self.meas_size, dtype=np.float32) * meas_noise_scale

        self.kf.errorCovPre = np.eye(self.state_size)

        self.meas = np.zeros((self.meas_size, 1), np.float32)
        self.state = np.zeros((self.state_size, 1), np.float32)

        self.first_detected = False

    def _update_dt(self, dt):
        for i in range(0, self.state_size, 2):
            self.kf.transitionMatrix[i, i+1] = dt

    def _first_detect(self, lanes):
        for l, i in zip(lanes, range(0, self.state_size, 8)):
            self.state[i:i+8:2, 0] = l
        self.kf.statePost = self.state
        self.first_detected = True

    def update(self, lanes):
        if self.first_detected:
            for l, i in zip(lanes, range(0, self.meas_size, 4)):
                if l is not None:
                    self.meas[i:i+4, 0] = l
            self.kf.correct(self.meas)
        else:
            if lanes.count(None) == 0:
                self._first_detect(lanes)

    def predict(self, dt):
        if self.first_detected:
            self._update_dt(dt)
            state = self.kf.predict()
            lanes = []
            for i in range(0, len(state), 8):
                lanes.append((state[i], state[i+2], state[i+4], state[i+6]))
            return lanes
        else:
            return None
class LaneDetector:
    def __init__(self, road_horizon, prob_hough=True):
        self.prob_hough = prob_hough
        self.vote = 50
        self.roi_theta = 0.3
        self.road_horizon = road_horizon

    def _standard_hough(self, img, init_vote):
        # Hough transform wrapper to return a list of points like PHough does
        lines = cv2.HoughLines(img, 1, np.pi/180, init_vote)
        points = [[]]
        for l in lines:
            for rho, theta in l:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*a)
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*a)
                points[0].append((x1, y1, x2, y2))
        return points

    def _base_distance(self, x1, y1, x2, y2, width):
        # compute the point where the give line crosses the base of the frame
        # return distance of that point from center of the frame
        if x2 == x1:
            return (width*0.5) - x1
        m = (y2-y1)/(x2-x1)
        c = y1 - m*x1
        base_cross = -c/m
        return (width*0.5) - base_cross

    def _scale_line(self, x1, y1, x2, y2, frame_height):
        # scale the farthest point of the segment to be on the drawing horizon
        if x1 == x2:
            if y1 < y2:
                y1 = self.road_horizon
                y2 = frame_height
                return x1, y1, x2, y2
            else:
                y2 = self.road_horizon
                y1 = frame_height
                return x1, y1, x2, y2
        if y1 < y2:
            m = (y1-y2)/(x1-x2)
            x1 = ((self.road_horizon-y1)/m) + x1
            y1 = self.road_horizon
            x2 = ((frame_height-y2)/m) + x2
            y2 = frame_height
        else:
            m = (y2-y1)/(x2-x1)
            x2 = ((self.road_horizon-y2)/m) + x2
            y2 = self.road_horizon
            x1 = ((frame_height-y1)/m) + x1
            y1 = frame_height
        return x1, y1, x2, y2

    def detect(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        roiy_end = frame.shape[0]
        roix_end = frame.shape[1]
        roi = img[self.road_horizon:roiy_end, 0:roix_end]
        cv2.imshow("sdf", roi)
        blur = cv2.medianBlur(roi, 5)
        contours = cv2.Canny(blur, 60, 120)
        cv2.imshow("ascs", contours)

        if self.prob_hough:
            lines = cv2.HoughLinesP(contours, 1, np.pi/180, self.vote, minLineLength=30, maxLineGap=100)
        else:
            lines = self.standard_hough(contours, self.vote)

        if lines is not None:
            # find nearest lines to center
            lines = lines+np.array([0, self.road_horizon, 0, self.road_horizon]).reshape((1, 1, 4))  # scale points from ROI coordinates to full frame coordinates
            left_bound = None
            right_bound = None
            for l in lines:
                # find the rightmost line of the left half of the frame and the leftmost line of the right half
                for x1, y1, x2, y2 in l:
                    theta = np.abs(np.arctan2((y2-y1), (x2-x1)))  # line angle WRT horizon
                    if theta > self.roi_theta:  # ignore lines with a small angle WRT horizon
                        dist = self._base_distance(x1, y1, x2, y2, frame.shape[1])
                        if left_bound is None and dist < 0:
                            left_bound = (x1, y1, x2, y2)
                            left_dist = dist
                        elif right_bound is None and dist > 0:
                            right_bound = (x1, y1, x2, y2)
                            right_dist = dist
                        elif left_bound is not None and 0 > dist > left_dist:
                            left_bound = (x1, y1, x2, y2)
                            left_dist = dist
                        elif right_bound is not None and 0 < dist < right_dist:
                            right_bound = (x1, y1, x2, y2)
                            right_dist = dist
            if left_bound is not None:
                left_bound = self._scale_line(left_bound[0], left_bound[1], left_bound[2], left_bound[3], frame.shape[0])
                cv2.line(img,(int(left_bound[0]),int(left_bound[1])),(int(left_bound[2]),int(left_bound[3])),255,6)
            if right_bound is not None:
                right_bound = self._scale_line(right_bound[0], right_bound[1], right_bound[2], right_bound[3], frame.shape[0])
                cv2.line(img,(int(right_bound[0]),int(right_bound[1])),(int(right_bound[2]),int(right_bound[3])),255,6)

            # cv2.imshow("sdvsv", img)
            # cv2.waitKey(0)
            return [left_bound, right_bound]
def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    
    # Retrieve the number of color channels of the image.
    #channel_count = img.shape[2]
    
    # color used to fill polygon
    match_mask_color = 255
      
    # Fill the polygon with white
    cv2.fillPoly(mask, vertices, (255,255,255))
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)

    
    return masked_image
def gamma_correction(RGBimage, correct_param = 0.35,equalizeHist = False):
    red = RGBimage[:,:,2]
    green = RGBimage[:,:,1]
    blue = RGBimage[:,:,0]
    
    red = red/255.0
    red = cv2.pow(red, correct_param)
    red = np.uint8(red*255)
    if equalizeHist:
        red = cv2.equalizeHist(red)
    
    green = green/255.0
    green = cv2.pow(green, correct_param)
    green = np.uint8(green*255)
    if equalizeHist:
        green = cv2.equalizeHist(green)
        
    
    blue = blue/255.0
    blue = cv2.pow(blue, correct_param)
    blue = np.uint8(blue*255)
    if equalizeHist:
        blue = cv2.equalizeHist(blue)
    

    output = cv2.merge((blue,green,red))
    return output
def gamma_correction_auto(RGBimage,equalizeHist = False): #0.35
    originalFile = RGBimage.copy()
    red = RGBimage[:,:,2]
    green = RGBimage[:,:,1]
    blue = RGBimage[:,:,0]
    
    forLuminance = cv2.cvtColor(originalFile,cv2.COLOR_BGR2YUV)
    Y = forLuminance[:,:,0]
    totalPix = vidsize[0]* vidsize[1]
    summ = np.sum(Y[:,:])
    Yaverage = np.divide(totalPix,summ)
    #Yclipped = np.clip(Yaverage,0,1)
    epsilon = 1.19209e-007
    correct_param = np.divide(-0.3,np.log10([Yaverage + epsilon]))
    correct_param = 0.7 - correct_param 

    red = red/255.0
    red = pow(red, correct_param)
    red = np.uint8(red*255)
    if equalizeHist:
        red = cv2.equalizeHist(red)
    
    green = green/255.0
    green = pow(green, correct_param)
    green = np.uint8(green*255)
    if equalizeHist:
        green = cv2.equalizeHist(green)
        
    
    blue = blue/255.0
    blue = pow(blue, correct_param)
    blue = np.uint8(blue*255)
    if equalizeHist:
        blue = cv2.equalizeHist(blue)
    

    output = cv2.merge((blue,green,red))
    #print(correct_param)
    return output
def hsv_filter(image, min_val_y, max_val_y,  min_val_w, max_val_w):
    """
    A function returning a mask for pixels within min_val - max_val range
    Inputs:
    - image - a BGR image you want to apply function on
    - min_val_y - array of shape (3,) giving minumum HSV values for yellow color
    - max_val_y - array of shape (3,) giving maximum HSV values for yellow color
    - min_val_w - array of shape (3,) giving minumum HSV values for white color
    - max_val_w - array of shape (3,) giving maximum HSV values for white color
    Returns:
    - img_filtered - image of pixels being in given threshold
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, min_val_y, max_val_y)
    mask_white = cv2.inRange(hsv, min_val_w, max_val_w)
    mask = cv2.bitwise_or(mask_yellow, mask_white)
    img_filtered = cv2.bitwise_and(image, image, mask=mask)
    
    return img_filtered
def hough_transform(original, gray_img, threshold, discard_horizontal = 0.4):
    """
    A function fitting lines that intersect >=threshold white pixels
    Input:
    - original - image we want to draw lines on
    - gray_img - image with white/black pixels, e.g. a result of Canny Edge Detection
    - threshold - if a line intersects more than threshold white pixels, draw it
    - discard_horizontal - smallest abs derivative of line that we want to take into account
    Return:
    - image_lines - result of applying the function
    - lines_ok - rho and theta
    """
    lines = cv2.HoughLines(gray_img, 0.5, np.pi / 360, threshold)
    image_lines = original
    lines_ok = [] #list of parameters of lines that we want to take into account (not horizontal)
            
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            #discard horizontal lines
            m = -math.cos(theta)/(math.sin(theta)+1e-10) #adding some small value to avoid dividing by 0
            if abs(m) < discard_horizontal:
                continue
            else:
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(image_lines, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)
                lines_ok.append([rho,theta])
        
    lines_ok = np.array(lines_ok)
                    
    return image_lines, lines_ok
def clustering(lines, original, region_of_interest_points, eps = 0.05, min_samples = 3):
    """
    A function using DBSCAN clustering algorithm for finding best lines to be drawn on the output video
    Inputs:
    - lines - output of hough tranform function, array containing parameters of found lines
    - original - image we want to draw final lines on
    - region_of_interest_points - for drawing lines of desired length
    Output:
    - img - image with detected lane lines drawn
    """
    img = original
    img_lines = np.zeros_like(img, dtype=np.int32)

    if lines.shape[0] != 0:
        #preprocessing features to be in (0-1) range
        scaler = MinMaxScaler()
        scaler.fit(lines)
        lines = scaler.fit_transform(lines)

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(lines) #applying DBSCAN Algorithm on our normalized lines
        labels = db.labels_

        lines = scaler.inverse_transform(lines) #getting back our original values

        grouped = defaultdict(list)
        #grouping lines by clusters
        for i, label in enumerate(labels):
            grouped[label].append([lines[i,0],lines[i,1]])

        num_clusters = np.max(labels) + 1
        means = []
        #getting mean values by cluster
        for i in range(num_clusters):
            mean = np.mean(np.array(grouped[i]), axis=0)
            means.append(mean)

        means = np.array(means)
        
        #printing the result on original image
        for rho, theta in means:
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(img, pt1, pt2, (255,255,255), 2, cv2.LINE_AA)
        
    return img
def IPM(image,ROI_points):
    pts_src = np.array(ROI_points,dtype=float)
    size = (700,600,3)
    ipm_out = np.zeros(size, np.uint8)
    
    dst_points = np.array(
                       [
                        [0,0],
                        [size[0] - 1, 0],
                        [size[0] - 1, size[1] -1],
                        [0, size[1] - 1 ]
                        ], dtype=float
                       )
    h, status = cv2.findHomography(pts_src, dst_points)
    ipm_out = cv2.warpPerspective(image, h, size[0:2])
    ipm_out = cv2.rotate(ipm_out,cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return ipm_out
#capturing video
cap = cv2.VideoCapture('road3.mp4')
# vidsize = (640,480,3)

#out = cv2.VideoWriter('final_kalman_night.avi', -1, 40.0, None, True)

#defining corners for ROI
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

topLeftPt = (0, height*(3.1/5))
topRightPt = (width, height*(3.1/5))

region_of_interest_points = [
(0, height),
#(0, height*(4.1/5)),
topLeftPt,
topRightPt,
#(width, height*(4.1/5)),
(width, height),
]

#defining color thresholds
min_val_y = np.array([15,80,190])
max_val_y = np.array([30,255,255])
min_val_w = np.array([0,0,195])
max_val_w = np.array([255, 80, 255])

ticks = 0

lt = LaneTracker(2, 0.1, 15)
ld = LaneDetector(100)


while True:
    precTick = ticks
    ticks = cv2.getTickCount()
    dt = (ticks - precTick) / cv2.getTickFrequency()
    ret, frame = cap.read()
    if ret:
        vidsize = (frame.shape[1],frame.shape[0],3)
        start_time = time.time()
        # cv2.imshow("aaaaa", frame)
        # cv2.waitKey(0)
        gamma = gamma_correction_auto(frame,equalizeHist = False) #0.2
        # cv2.imshow('gamma', np.hstack((frame,gamma)))
        # cv2.waitKey(0)
        cropped = region_of_interest(gamma, np.array([region_of_interest_points], np.int32))
        # cv2.imshow('cropped', cropped)
        # cv2.waitKey(0)
        # exit()
        #bilateral = cv2.bilateralFilter(cropped, 9, 80, 80)
        #cv2.imshow('bilateral', bilateral)
        #hsv = hsv_filter(cropped, min_val_y, max_val_y,  min_val_w, max_val_w)
        predicted = lt.predict(dt)

        lanes = ld.detect(cropped)
                
        helper = np.zeros_like(frame)
        
        if predicted is not None:
            cv2.line(helper, (int(predicted[0][0]), int(predicted[0][1])), (int(predicted[0][2]), int(predicted[0][3])), (0, 255, 0), 2)
            cv2.line(helper, (int(predicted[1][0]), int(predicted[1][1])), (int(predicted[1][2]), int(predicted[1][3])), (0, 255, 0), 2)
        # if lanes is not None:
        #     cv2.line(helper, (int(lanes[0][0]), int(lanes[0][1])), (int(lanes[0][2]), int(lanes[0][3])), (0, 255, 0), 2)
        #     cv2.line(helper, (int(lanes[1][0]), int(lanes[1][1])), (int(lanes[1][2]), int(lanes[1][3])), (0, 255, 0), 2)
        
        helper[:int(helper.shape[0]*0.55),:] = 0
        frame = cv2.add(helper,frame)
        ipmout = IPM(helper,region_of_interest_points)
        lt.update(lanes)
        #cv2.imshow('hsv', hsv)
        #canny = cv2.Canny(hsv, 80, 255) #100
        #cv2.imshow('canny', canny)
        #hough, lines = hough_transform(frame, canny, 11, discard_horizontal = 0.7) #14 0.4
        #cv2.imshow('hough', hough)
        #_, frame = cap.read()
        #final = clustering(lines, frame, np.array([region_of_interest_points], np.int32), eps = 0.5, min_samples = 4)
        cv2.imshow('final', frame)
        cv2.imshow('IPM', ipmout)
        #out.write(frame)
        
        print("-----Duration: ", time.time()-start_time)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()