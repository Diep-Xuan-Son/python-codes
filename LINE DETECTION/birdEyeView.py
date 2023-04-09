import cv2
import numpy as np
import time
import math as m
# import matplotlib.pyplot as plt

path = "./road1.mp4"
cap = cv2.VideoCapture(path)
i = 0

while (cap.isOpened()):
	ret, image = cap.read()

	if not ret:
		break
	start_time = time.time()
	image0 = image.copy()
	IMAGE_H = image.shape[0]
	IMAGE_W = image.shape[1]
	#-----------------calculate perspective transform-----------------------
	src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
	dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
	M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
	Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

	image = image[2*(IMAGE_H//3):IMAGE_H, 0:IMAGE_W] # Apply np slicing for ROI crop

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
	# if i == 140:
	# 	cv2.imwrite("birdEye.jpg", warped_img)
	# 	exit()
	#---------------------detach lane depend on color-----------------------------
	blur = cv2.GaussianBlur(warped_img,(3,3),1)
	hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
	#----yellow----
	lower0 = np.array([14,105,75])
	higher0 = np.array([179, 255, 255])
	mask0 = cv2.inRange(hsv, lower0, higher0)
	#--------------
	#------white-----
	lower1 = np.array([8,0,232])
	higher1 = np.array([179, 38, 255])
	mask1 = cv2.inRange(hsv, lower1, higher1)
	#----------------

	kernel = np.ones((3, 3), np.uint8)
	mask0 = cv2.dilate(mask0, kernel, iterations=1)
	mask1 = cv2.dilate(mask1, kernel, iterations=1)
	mask2 = np.zeros_like(mask1)
	lines0 = cv2.HoughLinesP(mask1,1,np.pi/180,50,maxLineGap = 60)
	if lines0 is not None:
		for line0 in lines0:
			x10,y10,x20,y20  = line0[0]
			angle = m.atan2(y20-y10, x20-x10)
			angle = angle*180/m.pi
			if (angle>70 and angle<110) or (angle>-110 and angle<-70):
				cv2.line(mask2,(x10,y10),(x20,y20),255,6)
			else:
				print(angle)
	cv2.imshow("ddd", mask2)
	#-----------------------------------------------------------------------------
	# cv2.imshow("ddd", mask0)
	cv2.waitKey(0)
	# exit()
	contours1, hierarchy = cv2.findContours(mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	contours, hierarchy = cv2.findContours(mask0,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	countour_max = []
	for ctr in contours:
		if len(ctr) > len(countour_max):
			countour_max = ctr
	contour = countour_max
	#----------------------draw box for contour-----------------------
	#--------minbox---------
	rect = cv2.minAreaRect(contour)
	cenx, ceny = rect[0][0], rect[0][1]
	print(cenx, ceny)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	cv2.drawContours(warped_img0, [box], -1, (0,0,255), 2)	#71 ,91, 371
	#----------------------
	# perimeter = cv2.arcLength(contour, True)
	# approx = cv2.approxPolyDP(contour, 0.02*perimeter, True)
	# x,y,w,h = cv2.boundingRect(approx)
	# cv2.rectangle(warped_img0, (x,y), (x+w, y+h), (0, 255, 0), 1)
	# cv2.drawContours(warped_img0, contours, -1, (0,255,0), 2)	#71 ,91, 371
	for ctr1 in contours1:
		cv2.drawContours(warped_img0, contours1, -1, (0,255,0), 2)	#71 ,91, 371
	#-----------------------------------------------------------------
	# if ((x+w/2) > (point_transform_left[0]) and (x+w/2) < 640) or ((x+w/2) < (point_transform_right[0]) and (x+w/2) > 640):	
	# 	print("wrong_lane")
	print(point_transform_left[0])
	if (cenx > (point_transform_left[0]) and cenx < 640) or (cenx < (point_transform_right[0]) and cenx > 640):		#for min box
		print("wrong_lane")
	print("-------Duration: ", time.time() - start_time)

	# img_inv = cv2.warpPerspective(warped_img, Minv, (IMAGE_W, IMAGE_H)) # Inverse transformation
	cv2.imshow("test", np.hstack((cv2.cvtColor(warped_img0, cv2.COLOR_BGR2RGB), image0))) # Show results
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
	i += 1
	print(i)