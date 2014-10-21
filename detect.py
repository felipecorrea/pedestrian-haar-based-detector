import cv2
import cv
import sys
import numpy as np
from matplotlib import pyplot as plt
######################################

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
	#IMG PATHS
	imagePath = "test3.jpg"
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

	#Draw a rectangle around the detected objects
	for (x, y, w, h) in pedestrians:
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

	cv2.imwrite("saida.jpg", image)
	cv2.imshow("Ppl found", image)
	cv2.waitKey(0)
	
	return 0



if __name__ == '__main__':
	main()


