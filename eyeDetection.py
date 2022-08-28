import cv2 as cv
import numpy as np

faceCascade = cv.CascadeClassifier("data/haarcascades_cuda/haarcascade_frontalface_default.xml")
eyeCascade = cv.CascadeClassifier("data/haarcascades_cuda/haarcascade_eye.xml")
faceImage = cv.imread("groupFacePic1.jpg")
grayscale = cv.cvtColor(faceImage, cv.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(grayscale, 1.1, 5)
#detects
for (x,y,w,h) in faces:
    cv.rectangle(faceImage, (x,y), (x+w, y+h), (0,225,0), 3)
    eye_gray = grayscale[y: y + h, x: x + w]
    eye_image = faceImage[y: y + h, x: x + w]
    eyes = eyeCascade.detectMultiScale(eye_gray)
    print(eyes)
    for (x1, y1, w1, h1) in eyes:
        cv.rectangle(eye_image, (x1, y1), (x1 + w1, y1 + h1), (0,225, 0), 1)

cv.imshow("My Picture", faceImage)
cv.waitKey(0)
cv.destroyAllWindows()