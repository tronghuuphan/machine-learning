import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

while(True):
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 3, 5)
    for (x, y, w, h) in faces:
        frame = cv.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h, x:x+h]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, eh, ew) in eyes:
            cv.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0),2)
    cv.imshow('Camera', frame)

    k = cv.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
cv.destroyAllWindow()
