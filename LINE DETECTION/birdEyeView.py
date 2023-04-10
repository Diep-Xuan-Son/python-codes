import cv2
import numpy as np
import time
import math as m
# import matplotlib.pyplot as plt

def _base_distance(x1, y1, x2, y2, width):
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

def gamma_correction_auto(RGBimage,equalizeHist = False): #0.35
	originalFile = RGBimage.copy()
	red = RGBimage[:,:,2]
	green = RGBimage[:,:,1]
	blue = RGBimage[:,:,0]
	
	forLuminance = cv2.cvtColor(originalFile,cv2.COLOR_BGR2YUV)
	Y = forLuminance[:,:,0]
	totalPix = IMAGE_H* IMAGE_W
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

def extend_line(p1, p2, distance=10000):
	diff = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
	p3_x = int(p1[0] + distance*np.cos(diff))
	p3_y = int(p1[1] + distance*np.sin(diff))
	p4_x = int(p1[0] - distance*np.cos(diff))
	p4_y = int(p1[1] - distance*np.sin(diff))
	return ((p3_x, p3_y), (p4_x, p4_y))

path = "./road3.mp4"
cap = cv2.VideoCapture(path)
i = 0
#defining corners for ROI
IMAGE_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
IMAGE_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
topLeftPt = (0, IMAGE_H*(3.1/5))
topRightPt = (IMAGE_W, IMAGE_H*(3.1/5))
region_of_interest_points = [
(0, IMAGE_H),
topLeftPt,
topRightPt,
(IMAGE_W, IMAGE_H),
]

while (cap.isOpened()):
	ret, image = cap.read()
	
	if not ret:
		break
	image = gamma_correction_auto(image,equalizeHist = False) #0.2
	start_time = time.time()
	image0 = image.copy()
	#-----------------calculate perspective transform-----------------------
	src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
	dst = np.float32([[4*IMAGE_W//9, IMAGE_H], [5*IMAGE_W//9, IMAGE_H], [0, 0], [IMAGE_W, 0]])
	# dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
	M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
	Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

	# image = image[2*(IMAGE_H//3):IMAGE_H, 0:IMAGE_W] # Apply np slicing for ROI crop
	image = image[int(topLeftPt[1]):IMAGE_H, 0:IMAGE_W] # Apply np slicing for ROI crop
	# cv2.imshow("vasvs", image)
	# cv2.waitKey(0)
	# exit()

	point = (4*IMAGE_W//10,image.shape[0])
	px = (M[0][0]*point[0] + M[0][1]*point[1] + M[0][2]) / ((M[2][0]*point[0] + M[2][1]*point[1] + M[2][2]))	#calculate x coordinate after perspective for point
	py = (M[1][0]*point[0] + M[1][1]*point[1] + M[1][2]) / ((M[2][0]*point[0] + M[2][1]*point[1] + M[2][2]))
	point_transform_left = (int(px), int(py))

	point1 = (6*IMAGE_W//10,image.shape[0])
	px1 = (M[0][0]*point1[0] + M[0][1]*point1[1] + M[0][2]) / ((M[2][0]*point1[0] + M[2][1]*point1[1] + M[2][2]))	#calculate x coordinate after perspective for point
	py1 = (M[1][0]*point1[0] + M[1][1]*point1[1] + M[1][2]) / ((M[2][0]*point1[0] + M[2][1]*point1[1] + M[2][2]))
	point_transform_right = (int(px1), int(py1))

	cv2.circle(image,point, 5, (0,255,0), -1)
	cv2.circle(image,point1, 5, (0,255,0), -1)
	# cv2.imshow("aaa", image0)
	# cv2.imshow("ddd", image)
	# cv2.waitKey(0)
	# exit()

	warped_img = cv2.warpPerspective(image, M, (IMAGE_W, IMAGE_H)) # Image warping
	#----------------------------------------------------------------------------
	warped_img0 = warped_img.copy()
	# if i == 74:
	# 	cv2.imwrite("birdEye.jpg", warped_img)
	# 	exit()
	#---------------------------------Canny---------------------------------------
	warped_img_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
	# blur = cv2.GaussianBlur(warped_img_gray,(3,3),1)
	blur = cv2.medianBlur(warped_img_gray, 5)
	canny = cv2.Canny(blur, 60, 120)
	lines = cv2.HoughLinesP(canny, 1, np.pi/180, 50, minLineLength=30, maxLineGap=100)
	if lines is not None:
		left_bound = None
		right_bound = None
		for l in lines:
			# find the rightmost line of the left half of the frame and the leftmost line of the right half
			# for x1, y1, x2, y2 in l:
			x1, y1, x2, y2 = l[0]
			# angle = m.atan2(y2-y1, x2-x1)
			# angle = angle*180/m.pi
			# if (angle>70 and angle<110) or (angle>-110 and angle<-70):
			theta = np.abs(np.arctan2((y2-y1), (x2-x1)))  # line angle WRT horizon
			if theta > 0.3:  # ignore lines with a small angle WRT horizon
				dist = _base_distance(x1, y1, x2, y2, IMAGE_W)
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
		if left_bound!=None:
			cv2.line(canny,(int(left_bound[0]),int(left_bound[1])),(int(left_bound[2]),int(left_bound[3])),255,6)
		if right_bound!=None:
			cv2.line(canny,(int(right_bound[0]),int(right_bound[1])),(int(right_bound[2]),int(right_bound[3])),255,6)
	cv2.imshow("ascs", canny)
	#-----------------------------------------------------------------------------
	#---------------------detach lane depend on color-----------------------------
	blur = cv2.GaussianBlur(warped_img,(3,3),1)
	hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
	#----yellow----
	lower0 = np.array([14,105,75])
	higher0 = np.array([179, 255, 255])
	mask0 = cv2.inRange(hsv, lower0, higher0)
	#--------------
	#------white-----
	# lower1 = np.array([8,0,232])
	# higher1 = np.array([179, 38, 255])
	# mask1 = cv2.inRange(hsv, lower1, higher1)
	#night
	lower1 = np.array([0,0,74])
	higher1 = np.array([120, 120, 115])
	mask1 = cv2.inRange(hsv, lower1, higher1)
	#----------------

	kernel = np.ones((3, 3), np.uint8)
	mask0 = cv2.dilate(mask0, kernel, iterations=1)
	mask0_line = np.zeros_like(mask0)
	mask1 = cv2.dilate(mask1, kernel, iterations=1)
	mask1_line = np.zeros_like(mask1)
	lines1 = cv2.HoughLinesP(mask1,1,np.pi/180,50,maxLineGap = 60)
	if lines1 is not None:
		for line1 in lines1:
			x10,y10,x20,y20  = line1[0]
			angle = m.atan2(y20-y10, x20-x10)
			angle = angle*180/m.pi
			if (angle>70 and angle<110) or (angle>-110 and angle<-70):
				cv2.line(mask1_line,(x10,y10),(x20,y20),255,6)
			# else:
			# 	print(angle)
	lines0 = cv2.HoughLinesP(mask0,1,np.pi/180,50,maxLineGap = 60)
	if lines0 is not None:
		for line0 in lines0:
			x10,y10,x20,y20  = line0[0]
			angle = m.atan2(y20-y10, x20-x10)
			angle = angle*180/m.pi
			if (angle>70 and angle<110) or (angle>-110 and angle<-70):
				cv2.line(mask0_line,(x10,y10),(x20,y20),255,6)
			# else:
			# 	print(angle)
	cv2.imshow("ddd", mask1_line)
	# cv2.waitKey(0)
	#-----------------------------------------------------------------------------
	# exit()
	#---------------------------------------line_yellow-------------------------------
	contours, hierarchy = cv2.findContours(mask0_line,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	if len(contours) != 0:
		countour_max = []
		for ctr in contours:
			if len(ctr) > len(countour_max):
				countour_max = ctr
		contour = countour_max
		#----------------------draw box for contour-----------------------
		#--------minbox---------
		rect = cv2.minAreaRect(contour)
		cenx, ceny = rect[0][0], rect[0][1]
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(warped_img0, [box], -1, (0,0,255), 2)	#71 ,91, 371
		#----------------------
		# perimeter = cv2.arcLength(contour, True)
		# approx = cv2.approxPolyDP(contour, 0.02*perimeter, True)
		# x,y,w,h = cv2.boundingRect(approx)
		# cv2.rectangle(warped_img0, (x,y), (x+w, y+h), (0, 255, 0), 1)
		# cv2.drawContours(warped_img0, contours, -1, (0,255,0), 2)	#71 ,91, 371
		#-----------------------------------------------------------------
		# if ((x+w/2) > (point_transform_left[0]) and (x+w/2) < 640) or ((x+w/2) < (point_transform_right[0]) and (x+w/2) > 640):	
		# 	print("wrong_lane")
		if (cenx > (point_transform_left[0]) and cenx < 640) or (cenx < (point_transform_right[0]) and cenx > 640):		#for min box
			print("wrong_lane")
	print("-------Duration: ", time.time() - start_time)
	#----------------------------------------------------------------------------------
	#-----------------------------------------line_white-------------------------------
	contours1, hierarchy = cv2.findContours(mask1_line,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	if len(contours1) != 0:
		dist_ctr2center = IMAGE_W
		closet_rect = []
		for ctr1 in contours1:
			rect1 = cv2.minAreaRect(ctr1)
			cenx1, ceny1 = rect1[0][0], rect1[0][1]
			if abs(cenx1-IMAGE_W//2) < dist_ctr2center:
				dist_ctr2center = abs(cenx1-IMAGE_W//2)
				closet_rect = rect1
		box1 = cv2.boxPoints(closet_rect)
		box1 = np.int0(box1)
		# (box1[0], box1[3]) = extend_line(box1[0], box1[3], IMAGE_H)
		# (box1[1], box1[2]) = extend_line(box1[1], box1[2], IMAGE_H)
		cv2.drawContours(warped_img0, [box1], -1, (0,255,0), 2)	#71 ,91, 371

	# img_inv = cv2.warpPerspective(warped_img, Minv, (IMAGE_W, IMAGE_H)) # Inverse transformation
	cv2.imshow("test", np.hstack((cv2.cvtColor(warped_img0, cv2.COLOR_BGR2RGB), image0))) # Show results
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
	i += 1
	print(i)