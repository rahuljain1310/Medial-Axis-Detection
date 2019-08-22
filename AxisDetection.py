## Importing Libraries
import cv2
import numpy as np

## Background Removal ##
BgSubstract = cv2.createBackgroundSubtractorMOG2(history=4000,detectShadows=False)
kernel = np.ones((5,5),np.uint8)
kernelMorph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

def BackgroundRemove (frame, noiseFilter = False, blur = False):
	if blur:
		frame = cv2.GaussianBlur(frame,(3,3),0)
	fgmask = BgSubstract.apply(frame)
	if noiseFilter:
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernelMorph)
		# fgmask = cv2.erode(fgmask,kernel)
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
	# print ((x1,y1),(x2,y2))
	return (x1,y1),(x2,y2)

def AddHoughLines(iframe, frame):
	edges = cv2.Canny(iframe,100,200,apertureSize = 3)
	# edges = cv2.Sobel(iframe,ddepth=-1,dx=1,dy=1)
	lines = cv2.HoughLines(edges,2,np.pi/180,200)
	showFrame('Edges',edges)
	try:
		for line in lines:
			print(line)
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
		gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
		blank_image = np.zeros(shape=fr.shape, dtype=np.uint8)
		iframe = BackgroundRemove(gray, noiseFilter=True, blur=False)
		AddHoughLines(iframe,blank_image)
		# axisImage = cv2.bitwise_and(fr,blank_image,mask=iframe)
		# axisFrame = cv2.add(axisImage,)
		cv2.imshow('Video',cv2.add(fr,blank_image))
		if 27 == (cv2.waitKey(20) & 0xff):
			break
		ret, frame = vidObj.read()

## Cleanup ##
vidObj.release()
cv2.destroyAllWindows()