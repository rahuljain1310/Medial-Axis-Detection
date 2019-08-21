## Importing Libraries
import cv2
import numpy as np

## Background Removal ##
BgSubstract = cv2.createBackgroundSubtractorMOG2(history=4000,varThreshold=32,detectShadows=False)
# kernel = np.ones((5,5),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

def BackgroundRemove (frame, noiseFilter = False):
	# x = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=2)
	fgmask = BgSubstract.apply(frame)
	if noiseFilter:
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
	showFrame('BackGround Remove',fgmask)
	return fgmask

def showFrame (name,frame):
	cv2.imshow(name,frame)
	return 27 == (cv2.waitKey(30) & 0xff)

def getPoints(rho,theta):
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 - 1000*b)
	y1 = int(y0 + 1000*a)
	x2 = int(x0 + 1000*b)
	y2 = int(y0 - 1000*a)
	print ((x1,y1),(x2,y2))
	return (x1,y1),(x2,y2)

def AddHoughLines(iframe, frame):
	edges = cv2.Canny(iframe,50,200,apertureSize = 3)
	Transformed = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
	lines = cv2.HoughLines(Transformed,1,np.pi/180,200)
	showFrame('Edges',edges)
	try:
		for line in lines:
			for rho,theta in line:
				p1,p2 = getPoints(rho,theta)
				cv2.line(frame,p1,p2,(255,255,0),2)
	except:
		pass
	
if __name__ == "__main__":
	VideoPath = "1.mp4"
	vidObj = cv2.VideoCapture(VideoPath)
	ret, frame = vidObj.read()
	while(ret):
		fr = cv2.resize(frame, (960, 540))
		blur = cv2.GaussianBlur(fr,(25,25),0)
		iframe = BackgroundRemove(blur,noiseFilter=True)
		AddHoughLines(iframe,fr)
		if showFrame('Video', fr):
			break
		ret, frame = vidObj.read()

## Cleanup ##
vidObj.release()
cv2.destroyAllWindows()