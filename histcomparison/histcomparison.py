### Histogram comparison with histogram equalization method
### To evaluate equalization

import cv2
import cv
import sys
import numpy as np
from matplotlib import pyplot as plt

def generate_histogram(img):
	hist,bins = np.histogram(img.flatten(),256,[0,256])
	
	#cumulative distribution function calculation
	cdf = hist.cumsum()
	cdf_normalized = cdf *hist.max()/ cdf.max() # this line not necessary.	
	plt.plot(cdf_normalized, color = 'b')
	plt.hist(img.flatten(),256,[0,256], color = 'r')
	plt.xlim([0,256])
	plt.legend(('cdf','histograma'), loc = 'upper left')
	plt.show()
	
	return hist



def main():
	imagePath = "img.jpg"
	
	img = cv2.imread(imagePath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	generate_histogram(gray)
	
	cv2.imwrite("before.jpg", gray)

	gray = cv2.equalizeHist(gray)
	
	generate_histogram(gray)
	
	cv2.imwrite("after.jpg",gray)
	
	return 0

if __name__ == '__main__':
	main()

