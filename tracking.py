import cv2
import cv
import uuid
import sys
import glob
import numpy as np
import random
from matplotlib import pyplot as plt
######################################

availablecolors = [(255, 0, 0), (0,0,255), (0,255,0)]

class People():
	def __init__(self, x, y, h, w):
		self.x = x
		self.y = y
		self.h = h
		self.w = w
		self.color = availablecolors[random.randint(0,2)]
		self.label = "batata"
		
	def get_label(self):
		return self.label
	
	def drawRect(self, image):
		cv2.rectangle(image, (self.x, self.y), (self.x+self.w, self.y+self.h), self.color, 2)
		return True
		
	def drawLabel(self, image):
		cv2.putText(image, self.label, (self.x, self.y-10), cv2.FONT_HERSHEY_PLAIN, 1, self.color, 2);
		return True
		
	def generateHistogram(self):
		pass
		
	def compareHistograms(hist):
		pass
	
def generate_histogram(img):
	hist,bins = np.histogram(img.flatten(),256,[0,256])
	
	#cumulative distribution function calculation
	cdf = hist.cumsum()
	
	plt.plot(cdf_normalized, color = 'b')
	plt.hist(img.flatten(),256,[0,256], color = 'r')
	plt.xlim([0,256])
	plt.legend(('cdf','histogram'), loc = 'upper left')
	plt.show()
	
	return hist


def normalize_grayimage(image):
	image = cv2.equalizeHist(image)
	cv2.imshow("Equalized img", image)
	
	return image



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

		print "Found {0} ppl!".format(len(pedestrians))
		
		
		ppl = []
		#Draw a rectangle around the detected objects
		for (x, y, w, h) in pedestrians:
			ppl.append(People(x,y,h,w))
		
		for i in ppl:
			i.drawRect(image)
			i.drawLabel(image)
		
		outputname = "testoutput/output_"+filename.split("/")[1]
		cv2.imwrite(outputname, image)
		#cv2.imshow("Ppl found", image)
		#cv2.waitKey(0)
	
	return 0



if __name__ == '__main__':
	main()


