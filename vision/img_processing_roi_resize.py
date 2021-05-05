import cv2
import numpy as np
import imutils

img = cv2.imread('need/threeblocktest.JPG')
image = imutils.resize(img,width=400)

#Bounding Box / Color
cv2.rectangle(image, (190,140),(230,180),(0,0,255),2)

#img min,max
region = image[140:180,190:230]

b,g,r = np.mean(region, axis=(0,1)).round(2)
print([b,g,r])

kernel=np.ones((15,15), np.uint8)

#creating range from average bgr
lower = (b-20,g-20,r-20)
higher = (b+50,g+50,r+50)

dst = cv2.inRange(image,lower,higher)

#morphology
closed = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

#이미지 평균값 구하기
avg = dst.mean()
#평균값을 기준으로 0과 1로 변환
#4x4로 크기 축소
dst=cv2.resize(opened,(5,5))
hash = 1*(dst>avg)
sum_width = []
sum_length = []
for a in hash[3,:]:
    sum_width.append(a) #3행 추출
for a in hash[:,3]:
    sum_length.append(a) #3열 추출
left = (hash[2,0],hash[2,1],hash[2,2],hash[2,2],hash[3,2],hash[4,2],hash[2,4],hash[2,3])
print(left)
if sum(sum_width) + sum(sum_length) == 10:
    print("전방에 사거리블록이 있습니다.")
elif left == (1, 1, 1, 1, 1, 1, 0, 0):
    print('전방에 좌회전블록이 있습니다.')
elif left == (0, 0, 1, 1, 1, 1, 1, 1):
    print('전방에 우회전블록이 있습니다.')
else:
    print("전방에 삼거리블록이 있습니다.")


cv2.imshow('',image)
cv2.imshow('1',opened)
cv2.waitKey()
cv2.destroyAllWindows()