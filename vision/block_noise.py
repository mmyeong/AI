import cv2
import numpy


src = cv2.imread('need/Block.png')
dst1 = cv2.inRange(src,(0,128,128),(100, 255, 255))#Yellow
src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV) #영상 HSV모델로 변환
dst2 = cv2.inRange(src_hsv,(0,150,0),(45, 255, 255))#Yellow
se = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
dilate_BGR = cv2.dilate(dst1, se)
dilate_HSV = cv2.dilate(dst2, se)

cv2.imshow('original',src)
cv2.imshow('dilate_BGR',dilate_BGR)
cv2.imshow('dilate-HSV',dilate_HSV)
cv2.waitKey()

cv2.destroyWindow()