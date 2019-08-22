## Importing Libraries
import cv2
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
	fgmask = cv2.erode(fgmask,kernel,iterations=2)
	fgmask = cv2.dilate(fgmask,kernelRect, iterations =2)
	fgmask = cv2.erode(fgmask,kernel,iterations=1)
	# fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE ,kernelBig, iterations=2)
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
	return (x1,y1),(x2,y2)

def AddHoughLines(iframe, frame):
	edges = cv2.Canny(iframe,50,100,apertureSize = 3)
	# edges = cv2.Sobel(iframe,ddepth=-5,dx=1,dy=1)
	lines = cv2.HoughLines(edges,1,np.pi/180,100)
	showFrame('Edges',edges)
	try:
		# z = [sum(y) / len(y) for y in zip(*lines)]
		# print(lines)
		# print(z)
		# p1,p2 = getPoints(z[0],z[1])
		# cv2.line(frame,p1,p2,Cyan,2)
		for line in lines:
			print(line)
			for rho,theta in line:
				p1,p2 = getPoints(rho,theta)
				cv2.line(frame,p1,p2,(0,255,255),2)
	except:
		pass
	
if __name__ == "__main__":
	VideoPath = "1.mp4"
	vidObj = cv2.VideoCapture(VideoPath)
	ret, frame = vidObj.read()
	while(ret):
		# fr = frame
		fr = cv2.resize(frame, (960, 540))
		gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
		iframe = BackgroundRemove(gray)
		AddHoughLines(iframe,fr)
		# axisImage = cv2.bitwise_and(fr,blank_image,mask=iframe)
		# axisFrame = cv2.add(axisImage,)
		cv2.imshow('Video',fr)
		if 27 == (cv2.waitKey(20) & 0xff):
			break
		ret, frame = vidObj.read()

## Cleanup ##
vidObj.release()
cv2.destroyAllWindows()