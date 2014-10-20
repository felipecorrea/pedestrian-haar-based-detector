import cv2
import sys

#PATHS
imagePath = "test3.jpg"
cascPath = "cascades/haarcascade_pedestrian.xml"
######

pplCascade = cv2.CascadeClassifier(cascPath)
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = gray - 20

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

