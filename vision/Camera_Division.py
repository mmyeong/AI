import cv2
import numpy as np

cap = cv2.VideoCapture(0)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #넓이
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #높이

while True:
    ret, frame = cap.read()

    roi = frame[int(h/2):h, 0:w]

    cv2.imshow('frame',frame)
    cv2.imshow('division',roi)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()