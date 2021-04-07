import cv2
import numpy as np

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("test.mp4")

while True:
  ret, frame = cap.read()
  roi = frame[0:480, 0:852]  # roi의 기본 개념은 관심영역이다. 탐지하고자하는 범위를 영상내에서 설정해주는 것이다.
  rows, cols, _ = roi.shape

  gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
  #영상 데이터가 칼라이미지면 데이터량이 3배이다. 트래킹하는데 색상 데이터가 의미가 없어서 gray로 변환, 회색은 분석, 컬러는 표현
  gray_roi = cv2.GaussianBlur(gray_roi, (7,7), 1) #가우시안 필터를 사용하여 노이즈를 줄임 7,7이라 정확도가 떨어질 수 있음

  _, threshold = cv2.threshold(gray_roi, 5, 255, cv2.THRESH_BINARY_INV)
  #threshold를 사용해서, gray_roi 이미지에서, 임계값 5까지는 그냥 검은색으로, 그 이상은 흰색으로 이진화 시킨다.
  #THRESH_BINARY_INV은 흰색이 검은색으로 검은색이 흰색으로 감
  #THRESH_BINARY은 흰색은 흰색 검은색은 검은색

  contours,_= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  #contours 등고선, _ 계층구조이다. 19년도 자료라 형식이 바뀐듯 함 영상에서는 _, contours, _이다.
  #앞에 _ 붙으면 값이 다르게 나옴
  contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
  #contour의 백터를 크기에 따라 정렬해주는데 key로 해당 contours를 어떤 기준으로 정렬할 것인지를 정한다.
  #람타 x로 contours를 받고, 그것을 사용해서 사이즈를 반환하는 cv2메소드를 돌리면 크기에 따라 정렬이 된다.
  #윤곽선이 가장 큰 부분만 남김
  print(contours)
  for cnt in contours:
    (x,y,w,h) = cv2.boundingRect(cnt)
    cv2.rectangle(roi, (x,y), (x+w, y+h), (255, 0, 0), 2)
    cv2.line(roi, (x+int(w/2), 0), (x+int(w/2), rows), (0, 255, 0), 2)
    cv2.line(roi, (0, y+int(h / 2)), (cols, y+int(h / 2)), (0, 255, 0), 2)
    #먼저 contour에 바운딩 박스를 입히고 화면 상단에서 아래로 선을 바운딩 박스 중앙에 교차해주면 시선에 따른 십자선이 화면에 출력됨

    cv2.drawContours(roi, [cnt], -1, (0,0,255), 3)
    #roi - contour를 나타낼 대상 이미지, contours 의 좌표 영역, -1는 roi에 실제로 그릴 인덱스 파라미터 값이 음수면 모든 contour를 그림, 선의 색상, 선의 두깨
    break

  cv2.imshow("Threshold", threshold)
  cv2.imshow("Gray roi", gray_roi)
  cv2.imshow("Roi", roi)

  key = cv2.waitKey(30)
  #각 프레임을 표시하기 위해 각 프레임 사이에 30밀리초를 기다림
  if key == 27: #27은 esc와 같음 esc키를 누르면 코드가 꺼짐
    break
cv2.destroyWindow()