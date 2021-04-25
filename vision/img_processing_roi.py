import cv2
import numpy as np
import imutils

img = cv2.imread('need/threeblock1.jpg')
image = imutils.resize(img,width=400)

#Bounding Box / Color
cv2.rectangle(image, (190,180),(230,220),(0,0,255),2)

#img min,max
region = image[180:220,190:230]

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

cv2.imshow('',image)
cv2.imshow('1',opened)
cv2.waitKey()
cv2.destroyAllWindows()