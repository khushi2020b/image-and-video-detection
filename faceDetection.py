import cv2 as cv
import numpy as np

faceCascade = cv.CascadeClassifier("data/haarcascades_cuda/haarcascade_frontalface_default.xml")
faceImage = cv.imread("bigcrowd.jpg")
grayscale = cv.cvtColor(faceImage, cv.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(grayscale, 1.1, 5)
print(faces)
for (x,y,w,h) in faces:
    cv.rectangle(grayscale, (x,y), (x+w, y+h), (0,225,0), 3)

cv.imshow("My Picture", grayscale)
cv.waitKey(0)
cv.destroyAllWindows()