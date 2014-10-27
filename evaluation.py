#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  evaluation.py
#  
#  Copyright 2014 Felipe Correa (https://github.com/felipecorrea)
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import glob
import cv2
import cv
import sys
import numpy as np
from matplotlib import pyplot as plt
######################################

def normalize_grayimage(image):
	image = cv2.equalizeHist(image)
	#cv2.imshow("Equalized img", image)
	
	return image



def main():
	#Values for statistical evaluation
	num_ppl = 0
	num_pos_detections = 0
	num_false_positivos = 0 
	num_false_negatives = 0
	
	#IMG PATHS
	images = glob.glob("img/*.jpg")
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
		
		
		
		#Draw a rectangle around the detected objects
		for (x, y, w, h) in pedestrians:
			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		
		
		outputname = "testoutput/output_"+filename.split("/")[1]
		cv2.imwrite(outputname, image)
		#cv2.imshow("Ppl found", image)
		#cv2.waitKey(0)
	
	return 0



if __name__ == '__main__':
	main()



