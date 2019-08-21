## Importing Libraries
import cv2
import numpy as np

## Background Removal ##
BgSubstract = cv2.createBackgroundSubtractorMOG2()
# kernel = np.ones((5,5),np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

def BackgroundRemove (frame, noiseFilter = False):
	fgmask = BgSubstract.apply(frame)
	if noiseFilter:
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
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
	return (x1,y1),(x2,y2)


def AddHoughLines(iframe, frame):
	edges = cv2.Canny(iframe,50,200,apertureSize = 3)
	lines = cv2.HoughLines(edges,1,np.pi/180,200)
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
		frame = cv2.resize(frame, (960, 540))
		iframe = BackgroundRemove(frame,noiseFilter=True)
		AddHoughLines(iframe,frame)
		if showFrame('Video', frame):
			break
		ret, frame = vidObj.read()

## Cleanup ##
vidObj.release()
cv2.destroyAllWindows()