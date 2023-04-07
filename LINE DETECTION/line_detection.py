import cv2 as cv
import numpy as np
import cvzone


cap = cv.VideoCapture('road.mp4')


while (cap.isOpened()):
    ret,vid = cap.read()
    if ret:  # if ret is True:
        vid = cv.resize(vid, (640, 480), interpolation=cv.INTER_AREA)

        blur = cv.GaussianBlur(vid,(3,3),1)
        hsv = cv.cvtColor(blur,cv.COLOR_BGR2HSV)

        lower0 = np.array([18,100,140])
        higher0 = np.array([60, 255, 255])
        mask0 = cv.inRange(hsv, lower0, higher0)
        canny0 = cv.Canny(mask0,90,100)
        lines0 = cv.HoughLinesP(canny0,1,np.pi/180,50,maxLineGap = 60)
        if lines0 is not None:
            for line0 in lines0:
                x10,y10,x20,y20  = line0[0]
                cv.line(vid,(x10,y10),(x20,y20),(0,0,255),6)

        lower = np.array([60,0,220])
        higher = np.array([150, 25, 255])
        mask = cv.inRange(hsv, lower, higher)
        canny = cv.Canny(mask,90,100)

        lines = cv.HoughLinesP(canny,1,np.pi/180,50,maxLineGap = 60)
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2  = line[0]
                cv.line(vid,(x1,y1),(x2,y2),(0,0,255),6)

        all = cvzone.stackImages([vid,hsv,mask,canny,mask0,canny0],2,0.65)
        cv.imshow('frame',all)
        #cv.waitKey(30)
        # define q as the exit button
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else: 
        break
    
# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv.destroyAllWindows()
