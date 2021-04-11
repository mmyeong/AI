import sys
import numpy as np
import cv2

#영상 불러오기

src = cv2.imread('need/Block.png')

if src is None: #예외처리
    print('실패')
    sys.exit()

src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV) #영상 HSV모델로 변환

dst1 = cv2.inRange(src,(0,128,128),(100, 255, 255))#Yellow
dst2 = cv2.inRange(src_hsv,(0,150,0),(45, 255, 255))#Yellow
kernel = np.ones((11, 11), np.uint8)
result = cv2.morphologyEx(dst2, cv2.MORPH_CLOSE, kernel)
#H 0~45 S 150~255 V 0~255

cv2.imshow('src',src) #원본영상
cv2.imshow('dst1',dst1) #RGB
cv2.imshow('dst2', result) #HSV

cv2.waitKey()

cv2.destroyWindow()