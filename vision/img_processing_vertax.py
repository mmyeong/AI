import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import math


def setLabel(img, pts, label):
    (x, y, w, h) = cv2.boundingRect(pts)
    pt1 = (x, y + 30)
    cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255))


def main():
    img = cv2.imread('이미지', cv2.IMREAD_COLOR)

    if img is None:
        print('Image load failed!')
        return

    src_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 영상 HSV모델로 변환

    dst2 = cv2.inRange(src_hsv, (0, 50, 0), (255, 255, 255))  # Yellow
    # 커널생성부분 21by3(세로, 가로)으로 했다. 설정가능
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    # dilate_BGR = cv2.dilate(dst1, se)  #se를 넣지않고 None으로하면 3by3 커널 자동생성
    dilate_HSV = cv2.dilate(dst2, se)
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7))
    erode_HSV = cv2.erode(dilate_HSV, se1)

    Gaussian_HSV = cv2.GaussianBlur(erode_HSV, (0, 0), 1)  # 노이즈 완화


    contours, _ = cv2.findContours(
        Gaussian_HSV, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # findContours로 각각의 객체를 찾아냄
    # EXTERNAL로 바깥쪽만 찾아냄
    for pts in contours:  # pts가 객체하나하나의 외곽선을 ndarray로 받는다.
        if cv2.contourArea(pts) < 2000:  # 너무 작으면 무시
            continue
            # 노이즈 제거
        approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True) * 0.02, True)

        vtc = len(approx)

        if vtc == 3:
            setLabel(img, pts, 'TRI')
        elif vtc == 4:
            setLabel(img, pts, 'GoStraight')
        elif vtc == 10 or 12:  # 삼거리이거나 사거리는 10 또는 12
            print("삼거리입니다.")
            setLabel(img, pts, 'ThreeWayBlock')
        else:
            length = cv2.arcLength(pts, True)
            area = cv2.contourArea(pts)
            ratio = 4. * math.pi * area / (length * length)

            if ratio > 0.85:
                setLabel(img, pts, 'CIR')
    print(vtc)
    plt.axis('off'), plt.imshow(img)
    Canny_HSV = cv2.Canny(Gaussian_HSV, 50, 150)
    cv2.imshow('Canny_HSV', Canny_HSV)
    cv2.imshow('Gaussian_HSV', Gaussian_HSV)
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
