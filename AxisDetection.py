## Importing Libraries
import cv2
import math
import numpy as np

## Constant ##
Cyan = (255,255,0)

## Background Removal ##
# BgSubstract = cv2.createBackgroundSubtractorMOG2(history=8000,detectShadows=False)
BgSubstract = cv2.createBackgroundSubtractorKNN(history=12000,detectShadows=False)
kernel = np.ones((3,3),np.uint8)
kernelBig = np.ones((5,5),np.uint8)
kernelMorph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernelRect = np.array([
	[0,0,1,0,0],
	[0,0,1,0,0],
	[0,1,1,1,0],
	[0,0,1,0,0],
	[0,0,1,0,0]
],dtype=np.uint8)

def BackgroundRemove (frame):
	frame = cv2.GaussianBlur(frame,(3,3),0)
	fgmask = BgSubstract.apply(frame)
	# fgmask = cv2.dilate(fgmask,kernel)
	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
	# fgmask = cv2.erode(fgmask,kernel,iterations=2)
	# fgmask = cv2.dilate(fgmask,kernelRect, iterations =2)
	# fgmask = cv2.erode(fgmask,kernel,iterations=1)
	# fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE ,kernelBig, iterations=2)
	showFrame('BackGround Remove',fgmask)
	return fgmask

def showFrame (name,frame):
	cv2.imshow(name,frame)
	return 27 == (cv2.waitKey(30) & 0xff)

def getMedianLine(lines):
	L = len(lines)
	rho,theta = 0,0
	for line in lines:
		for r,t in line:
			rho += r
			theta += t
	return rho/L,theta/L

def getHoughLines(iframe):
	edges = cv2.Canny(iframe,50,100,apertureSize = 3)
	# edges = cv2.Sobel(iframe,ddepth=-5,dx=1,dy=1)
	showFrame('Edges',edges)
	lines = cv2.HoughLines(edges,1,np.pi/180,100)
	Segments = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength = 100,maxLineGap = 50)
	return lines,Segments

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

def AddHoughLines(lines,frame):
	try:
		for line in lines:
			for rho,theta in line:
				p1,p2 = getPoints(rho,theta)
				cv2.line(frame,p1,p2,(0,255,255),2)
	except:
		pass

def AddHoughSegments(Segments,frame):
	try:
		for c in Segments:
			for x1,y1,x2,y2 in c:
				cv2.line(frame,(x1,y1),(x2,y2),(255,165,0),2)
	except:
		pass

def AddMedianAxis(rho,theta,frame):
	p1,p2 = getPoints(rho,theta)
	cv2.line(frame,p1,p2,(255,0,255),2)

def getYBoundary(Segments):
	ymin = math.inf
	ymax = 0
	for s in Segments:
		for _,y1,_,y2 in s:
			if y2<y1:
				t = y1
				y1 = y2
				y2 = t
			if y1<ymin:
				ymin = y1
			if y2>ymax:
				ymax = y2
	return ymin,ymax

def getMedianLineSegment(rho,theta,Ymin,Ymax):
	if theta == 0:
		return (rho,Ymin),(rho,Ymax)
	Xmin = int(rho/np.cos(theta) - Ymin*np.tan(theta))
	Xmax = int(rho/np.cos(theta) - Ymax*np.tan(theta))
	return (Xmin,Ymin),(Xmax,Ymax)	

if __name__ == "__main__":
	VideoPath = "1.mp4"
	vidObj = cv2.VideoCapture(VideoPath)
	ret, frame = vidObj.read()
	while(ret):
		# fr = frame
		fr = cv2.resize(frame, (960, 540))
		gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
		iframe = BackgroundRemove(gray)
		lines,Segments = getHoughLines(iframe)
		try:
			# AddHoughLines(lines,fr)
			# AddHoughSegments(Segments,fr)
			rho,theta = getMedianLine(lines)
			Ymin,Ymax = getYBoundary(Segments)
			p1,p2 = getMedianLineSegment(rho,theta,Ymin,Ymax)
			# print(p1,p2)
			cv2.line(fr,p1,p2,(255,0,0),2)
			print("Success")
		except:
			pass
		cv2.imshow('Video',fr)
		if 27 == (cv2.waitKey(20) & 0xff):
			break
		ret, frame = vidObj.read()

## Cleanup ##
vidObj.release()
cv2.destroyAllWindows()