import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "./road1.mp4"
cap = cv2.VideoCapture(path)

while (cap.isOpened()):
	ret, image = cap.read()

	if not ret:
		break

	image0 = image.copy()
	IMAGE_H = image.shape[0]
	IMAGE_W = image.shape[1]

	src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
	dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
	M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
	Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

	# img = cv2.imread('./test_img.jpg') # Read the test img
	image = image[2*(IMAGE_H//3):IMAGE_H, 0:IMAGE_W] # Apply np slicing for ROI crop
	# cv2.imshow("aaa", image0)
	# cv2.imshow("ddd", image)
	# cv2.waitKey(0)
	# exit()
	warped_img = cv2.warpPerspective(image, M, (IMAGE_W, IMAGE_H)) # Image warping
	# img_inv = cv2.warpPerspective(warped_img, Minv, (IMAGE_W, IMAGE_H)) # Inverse transformation
	cv2.imshow("test", np.hstack((cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB), image0))) # Show results
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break