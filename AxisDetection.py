## Importing Libraries
import cv2
import math
import numpy as np
import statistics as st
## Constant ##
from math import sqrt
# angle between two points

	
Cyan = (255,255,0)

## Background Removal ##
# BgSubstract = cv2.createBackgroundSubtractorMOG2(history=8000,detectShadows=False)
BgSubstract = cv2.createBackgroundSubtractorKNN(detectShadows=False)
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

def getMedianLine(lines):
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
	bins = 180
	scale_factor = int(round(180/bins))
	line_partition = list([] for i in range(bins+1))
	if (not lines.__class__ == np.ndarray):

		if lines==None:
			return lines,0
	for line in lines:
		line_partition[int((line[0][1]*179.99/np.pi))//scale_factor].append(line)
	max_index = 0
	max_freq = 0
	for i in range(len(line_partition)):
		l1 = len(line_partition[i])
		if  (l1>max_freq):
			max_freq = l1
			max_index = i
	return (line_partition[max_index],max_index*scale_factor)
	# except:
	# 	if lines.__class__==None:
	# 		return lines
	# 	else:
	# 		print (lines)
	# 		return None
	


def rho_filter(lines):
	bins = 60
	max_rho = 1199
	min_rho = 0
	scale_factor = (max_rho-min_rho+1)//bins
	line_partition = list([] for i in range(bins))
	if (not lines.__class__==np.ndarray):
		if (lines==None):
			return lines,0,lines,0
	for l in lines:
		line_partition[int(min(abs(l[0][0]),max_rho)/scale_factor)].append(l)
	max_index = 0
	second_max_index = 0

	max_freq = 0
	second_max_freq = 0

	for i in range(len(line_partition)):
		l1 = len(line_partition[i])
		if  (l1>max_freq):
			second_max_freq = max_freq
			max_freq = l1
			second_max_index = max_index
			max_index = i

		elif (l1>second_max_freq):
			second_max_freq = l1
			second_max_index = i
	if (second_max_index==0):
		second_max_index = max_index
	
	return (line_partition[max_index],scale_factor*max_index,line_partition[second_max_index],scale_factor*second_max_index)
	# except:
	# 	if lines.__class__ ==None:
	# 		return lines
	# 	else:
	# 		print (lines)
	# 		return None

def get_extreme_coordinates(lines):
	histogram_bin_size = 10
	pass

def distance_from_line(x,y,rho,theta):
	return abs(x*np.cos(theta)+y*np.sin(theta)-rho)

def segment_filter(x1,y1,x2,y2,rho,thet):
	theta = thet*np.pi/180
	a1 = x1*np.cos(theta)+y1*np.sin(theta)-rho
	a2 = x2*np.cos(theta)+y2*np.sin(theta)-rho
	sl = sqrt((x1-x2)**2 + (y1-y2)**2)
	
	
	angle = np.arcsin((a1-a2)/sl)
	if (abs(angle*180/np.pi)>2):
		return False
	mean_dist = (abs(a1)+abs(a2))//2
	if (mean_dist>15):
		return False
	return True
	


		



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
	edges = cv2.Canny(iframe,50,100,apertureSize = 7)
	# edges = iframe
	# edges = cv2.Sobel(iframe,ddepth=-5,dx=1,dy=1)
	showFrame('Edges',edges)
	lines = cv2.HoughLines(edges,1,np.pi/180,90)
	linesf1,theta = theta_filter(lines)
	linesf2,rho,linesf3,rho2 = rho_filter(linesf1) 

	Segments = cv2.HoughLinesP(edges,rho = 1,theta = np.pi/180,threshold = 75,minLineLength=20,maxLineGap=10)
	def segf(segment):
		x1,y1,x2,y2 = segment[0][0],segment[0][1],segment[0][2],segment[0][3]
		return segment_filter(x1,y1,x2,y2,rho,theta) or segment_filter(x1,y1,x2,y2,-rho,theta)
	def segf2(segment):
		x1,y1,x2,y2 = segment[0][0],segment[0][1],segment[0][2],segment[0][3]
		return segment_filter(x1,y1,x2,y2,rho2,theta) or segment_filter(x1,y1,x2,y2,-rho2,theta)
	linesf = []
	segmentsf4 = []
	if (Segments.__class__==np.ndarray and linesf2.__class__==list and linesf3.__class__==list):
		#print(rho,theta)
		#print(len(Segments))
		segmentsf1 = list(filter(segf,Segments))
		segmentsf2 = list(filter(segf2,Segments))
		#print(len(segmentsf1))
		linesf = linesf2+linesf3
		segmentsf4 = segmentsf1 + segmentsf2

	else:
		segmentsf1 = None

	return (linesf,segmentsf4)

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
	
	ys = list(s[0][1] for s in Segments) + list(s[0][3] for s in Segments)
	ys.sort()
	ll = len(ys)
	picks = int((ll//5+1))
	#print(picks)
	ymin = st.mean(ys[:picks])
	ymax = st.mean(ys[ll-1-picks:]) 
	#print(ymin,ymax,10000)
	# yhists = np.histogram(ys,10)
	# ymin = st.median(yhists[0])
	# ymax = st.median(yhists[9])
	return ymin,ymax
	# ymin = math.inf
	# ymax = 0
	# for s in Segments:
	# 	for _,y1,_,y2 in s:
	# 		if y2<y1:
	# 			t = y1
	# 			y1 = y2
	# 			y2 = t
	# 		if y1<ymin:
	# 			ymin = y1
	# 		if y2>ymax:
	# 			ymax = y2
	# return ymin,ymax

def getMedianLineSegment(rho,theta,Ymin,Ymax):
	if theta == 0:
		return (rho,int(Ymin)),(rho,int(Ymax))
	Xmin = int(rho/np.cos(theta) - Ymin*np.tan(theta))
	Xmax = int(rho/np.cos(theta) - Ymax*np.tan(theta))
	return (Xmin,int(Ymin)),(Xmax,int(Ymax))	

if __name__ == "__main__":
	VideoPath = "1.mp4"
	vidObj = cv2.VideoCapture(VideoPath)
	ret, frame = vidObj.read()
	while(ret):
		fr = cv2.resize(frame, (960, 540))
		# print(len(fr))
		gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
		iframe = BackgroundRemove(gray)
		lines,Segments = getHoughLines(iframe)
		
		try:
			# AddHoughLines(lines,fr)
			
			# AddHoughSegments(Segments,fr)
			rho,theta = getMedianLine(lines)
			Ymin,Ymax = getYBoundary(Segments)
			p1,p2 = getMedianLineSegment(rho,theta,Ymin,Ymax)
			print(p1,p2)
			print(lines[0],'3')
			cv2.line(fr,(p1),(p2),(0,0,255),1)
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


def length(v):
    return sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]
def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]
def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=np.arccos(cosx) # in radians
   return rad*180/np.pi # returns degrees
def angle_clockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner
