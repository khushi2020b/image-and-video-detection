import cv2 as cv

# read the video
video = cv.VideoCapture(0)

faceCascade = cv.CascadeClassifier("data/haarcascades_cuda/haarcascade_frontalface_default.xml")
eyeCascade = cv.CascadeClassifier("data/haarcascades_cuda/haarcascade_eye.xml")
# video is many images in a infite loop
while True:
    frame, image = video.read()
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayscale, 1.1, 5) 
    for (x,y,w,h) in faces:
        cv.rectangle(image, (x,y), (x+w, y+h), (0,225,0), 3)
        eye_gray = grayscale[y: y + h, x: x + w]
        eye_image = image[y: y + h, x: x + w]
        eyes = eyeCascade.detectMultiScale(eye_gray)
        print(eyes)
        for (x1, y1, w1, h1) in eyes:
            cv.rectangle(eye_image, (x1, y1), (x1 + w1, y1 + h1), (0,225, 0), 1)

    cv.imshow("My face", image)

    k = cv.waitKey(1)

    if k == ord("q"):
        break

video.release()
cv.destroyAllWindows()

