import sys
import numpy as np
import cv2

#영상 불러오기

src = cv2.imread('need/gotest.JPG')

if src is None: #예외처리
    print('실패')
    sys.exit()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2HSV) #영상 HSV모델로 변환

dst1 = cv2.inRange(src,(0,128,128),(100, 255, 255))#Yellow
dst2 = cv2.inRange(gray,(0,150,0),(45, 255, 255))#Yellow


#4x4로 크기 축소
gray=cv2.resize(dst2,(4,4))


#모폴로지
kernel = np.ones((11, 11), np.uint8)
result = cv2.morphologyEx(dst2, cv2.MORPH_CLOSE, kernel)
#H 0~45 S 150~255 V 0~255

#평균값 구하기
avg = gray.mean()
#평균값을 기준으로 0과 1로 변환

hash = 1*(gray>avg)
print(hash)

cv2.imshow('src',src) #원본영상
cv2.imshow('dst2', result) #HSV
cv2.waitKey()

cv2.destroyWindow()