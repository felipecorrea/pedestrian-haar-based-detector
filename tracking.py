import cv2
import cv
import uuid
import sys
import glob
import numpy as np
import random
from matplotlib import pyplot as plt
######################################

availablecolors = [(255, 0, 0), (0,0,255), (0,255,0), (255,255,0), (0,255,255), (0,0,0), (255,255,255), (125,220,221), (30,100,30)]
availablenames = ["batata","cenoura","joseph","maria","ferrugem","abacate","banana","laranja","alface","prego","cadeira","gandalf","frodo",
"gloin","toin","saruman","legolas"]

class People():
	#New pedestrian detected
	def __init__(self, x, y, w, h, color=None, label=None):
		self.x = x
		self.y = y
		self.h = h
		self.myarray = []
		self.w = w
		if color is None:
			self.color = availablecolors[random.randint(0,len(availablecolors)-1)]
		else:
			self.color = color
		
		if label is None:
			self.label = availablenames[random.randint(0,len(availablenames)-1)]
		else:
			self.label = label
		
	def get_label(self):
		return self.label
	
	def drawRect(self, image):
		cv2.rectangle(image, (self.x, self.y), (self.x+self.w, self.y+self.h), self.color, 2)
		return True
		
	def drawLabel(self, image):
		cv2.putText(image, self.label, (self.x, self.y-10), cv2.FONT_HERSHEY_PLAIN, 1, self.color, 2);
		return True
	
	def cropImg(self,image):
		self.myarray = image[self.x:self.x+self.w, self.y:self.y+self.h]
		self.generateHistogram(self.myarray)
	
	def generateHistogram(self,img):
		hist,bins = np.histogram(img.flatten(),256,[0,256])
		self.hist = hist
		return hist
	
	def getHistogram(self):
		return self.hist
	

# If True = New pedestrian
# Else, returns  (x, y, w, h, color=None, label=None)

def compareHistograms(image,x,y,w,h, ppl):
	#temporary crop detected target
	tempCrop = image[x:x+w, y:y+h]
	#generate temporary histogram to compare to existant ones
	tempHist = generateHistogram(tempCrop)
	
	if(len(ppl) > 0):
		b = checkSimilarity(tempHist, ppl, image)
		if(b):
			return (b.x, b.y, b.w, b.h, b.color, b.label)
		else:
			return None
	else:
		return None
	
	return None
	
def checkSimilarity(temphist, ppl, image):
	results = {}

	for i in ppl:
		hist1 = i.generateHistogram(image)
		distance = chi2_distance(temphist, hist1)
		results[distance] = i
		return i

	results = sorted(results.items())
	
	return None
	
# quisquare distance calculation for histograms
def chi2_distance(histA, histB, eps = 1e-10):
	# compute the chi-squared distance
	d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
		for (a, b) in zip(histA, histB)])

	# return the chi-squared distance
	return d

def generateHistogram(img):
	hist,bins = np.histogram(img.flatten(),256,[0,256])
	return hist
	

def normalize_grayimage(image):
	image = cv2.equalizeHist(image)
	cv2.imshow("Equalized img", image)
	
	return image


ppl = []

def main():
	#Values for statistical evaluation
	num_ppl = 0
	num_pos_detections = 0
	num_false_positivos = 0 
	num_false_negatives = 0
	
	#IMG PATHS
	images = glob.glob("videotracking/*.jpg")
	print images 
	for filename in images:
		imagePath = filename
		cascPath = "cascades/haarcascade_pedestrian.xml"

		pplCascade = cv2.CascadeClassifier(cascPath)
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		gray = normalize_grayimage(gray)
		 
		pedestrians = pplCascade.detectMultiScale(
			gray,
			scaleFactor=1.2,
			minNeighbors=10,
			minSize=(32,96),
			flags = cv2.cv.CV_HAAR_SCALE_IMAGE
		)

		#print "Found {0} ppl!".format(len(pedestrians))
		
		
		#Draw a rectangle around the detected objects
		for (x, y, w, h) in pedestrians:
			a = compareHistograms(image,x,y,w,h,ppl)
			if(a):
				ppl.pop()
				ppl.append(People(x,y,w,h, a[4], a[5]))
			else:
				ppl.append(People(x,y,w,h))
				
				
		for i in ppl:
			i.drawRect(image)
			i.drawLabel(image)
			i.cropImg(image)
			
		
		outputname = "testoutput/output_"+filename.split("/")[1]
		cv2.imwrite(outputname, image)
		#cv2.imshow("Ppl found", image)
		#cv2.waitKey(0)
	
	return 0



if __name__ == '__main__':
	main()


