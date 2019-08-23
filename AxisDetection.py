## Importing Libraries
import cv2
import numpy as np
import statistics as st
## Constant ##
Cyan = (255,255,0)

## Background Removal ##
# BgSubstract = cv2.createBackgroundSubtractorMOG2(history=8000,detectShadows=False)
BgSubstract = cv2.createBackgroundSubtractorKNN()
kernel1 = np.ones((10,1),np.uint8)
kernel2 = np.ones((3,1),np.uint8)
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
	# frame = cv2.GaussianBlur(frame,(3,3),0)
	fgmask = BgSubstract.apply(frame)
	fgmask = cv2.dilate(fgmask,kernel2)
	fgmask = cv2.dilate(fgmask,kernel1,iterations = 2)
	fgmask = cv2.erode(fgmask,kernel1,iterations =2)
	# fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
	# fgmask = cv2.dilate(fgmask,kernelRect, iterations =2)
	# fgmask = cv2.erode(fgmask,kernel,iterations=1)
	# fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE ,kernelBig, iterations=2)
	showFrame('BackGround Remove',fgmask)
	return fgmask

def showFrame (name,frame):
	cv2.imshow(name,frame)
	return 27 == (cv2.waitKey(30) & 0xff)

def getAverageLine(lines):
	L = len(lines)
	rho,theta = 0,0
	for line in lines:
		for r,t in line:
			rho += r
			theta += t
	rho = rho/L
	theta = theta/L
	return rho,theta

def joinLines(lines):
	for x1,y1,x2,y2 in lines:
		pass

def theta_filter(lines):
	pass

def rho_filter(lines):
	rhos = list(x for x,y in lines)
	# max_rho = max(rhos)
	# min_rho = min(rhos)
	freq,bins = np.histogram(rhos,bins=60, range=range(1,1200))
	mf = max(freq)
	bins = 60
	max_rho = 1199
	min_rho = 0
	scale_factor = (max_rho-min_rho+1)//bins
	line_partition = list([] for i in range(bins))
	for l in lines:
		line_partition[min(l[0],max_rho)//scale_factor].append(l)
	max_index = 0
	max_freq = 0
	for i in range(len(line_partition)):
		l1 = len(line_partition[i])
		if  (l1>max_freq):
			max_freq = l1
			max_index = i
	return line_partition[i]


def getMedialLine(lines):
	L = len(lines)
	rho_a = []
	theta_a = []
	for line in lines:
		for r,t in line:
			rho_a.append(r)
			theta_a.append(t)
	rho = st.median(rho_a)
	theta = st.median(theta_a)
	return rho,theta

def getHoughLines(iframe):
	edges = cv2.Canny(iframe,50,100,apertureSize = 3)
	# edges = cv2.Sobel(iframe,ddepth=-5,dx=1,dy=1)
	showFrame('Edges',edges)
	lines = cv2.HoughLinesP(edges,rho = 1,theta = np.pi/180,threshold = 50,minLineLength=20,maxLineGap=10)
	return lines

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
			for x1,y1,x2,y2 in line:
				p1 = (x1,y1)
				p2 = (x2,y2)
				cv2.line(frame,p1,p2,(0,255,255),2)
	except:
		pass

def AddMedianAxis(rho,theta,frame):
	p1,p2 = getPoints(rho,theta)
	# print(p1,p2)
	cv2.line(frame,p1,p2,(255,0,255),2)



if __name__ == "__main__":
	VideoPath = "1.mp4"
	vidObj = cv2.VideoCapture(VideoPath)
	ret, frame = vidObj.read()
	while(ret):
		
		# fr = frame
		fr = cv2.resize(frame, (960, 540))
		print(len(fr))
		gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
		iframe = BackgroundRemove(gray)
		lines = getHoughLines(iframe)
		print(lines[0][0])
		# print(lines)
		try:
			AddHoughLines(lines,fr)
			rho,theta = getMedialLine(lines)
			AddMedianAxis(rho,theta,fr)
		except:
			pass
		cv2.imshow('Video',fr)
		if 27 == (cv2.waitKey(20) & 0xff):
			break
		ret, frame = vidObj.read()

## Cleanup ##
vidObj.release()
cv2.destroyAllWindows()