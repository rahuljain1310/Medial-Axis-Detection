## Importing Libraries
import cv2
import numpy as np

VideoPath = "1.mp4"
vidObj = cv2.VideoCapture(VideoPath)
BgSubstract = cv2.createBackgroundSubtractorMOG2()

while(1):
		ret, frame = vidObj.read()
		fgmask = BgSubstract.apply(frame)
		cv2.imshow('frame',fgmask)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
				break
vidObj.release()
cv2.destroyAllWindows()