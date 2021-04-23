import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import math

def setLabel(img, pts, label):
    (x,y,w,h)= cv2.boundingRect(pts)
    pt1 = (x,y)
    pt2 = (x+w,y+h)
    cv2.rectangle(img,pt1,pt2,(0,0,255),1)
    cv2.putText(img,label,pt1,cv2.FONT_HERSHEY_PLAIN,1,(0,0,255))


def main():
    img = cv2.imread('need/threeblock.jpg',cv2.IMREAD_COLOR)

    if img is None:
        print('Image load failed!')
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #grayscale 변환
    _,img_bin = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    #threshold를 OTSU로 img_bin저장
    contours,_ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #findContours로 객체 찾기
    #External로 바깥쪽 찾기
    for pts in contours: #pts가 객체하나하나의 외곽선을 ndarray로 받음
        if cv2.contourArea(pts)<400: #너무 작으면 무시
            continue
            #노이즈 제거
        approx = cv2.approxPolyDP(pts,cv2.arcLength(pts,True)*0.02,True)
        line = len(approx)
        if line == 2:
            setLabel(img,pts,'ThreeBlock')
        elif line == 4:
            setLabel(img,pts,'FourBlock')
        elif line == None:
            setLabel(img,pts,'GoBlock')
        else:
            setLabel(img,pts,'false')

    cv2.imshow('dst',img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # cv2.imshow('three',img)
    # cv2.waitKey()
    # cv2.destroyWindow()
if __name__ == '__main__':
    main()