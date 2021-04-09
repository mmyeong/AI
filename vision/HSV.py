import cv2
import numpy as np

def nothing(x):
  #any operation
  pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars") #트렉바 만들기
cv2.createTrackbar("L-H", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 180, 180, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

while True:
  _, frame = cap.read()
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  l_h = cv2.getTrackbarPos("L-H", "Trackbars")
  l_s = cv2.getTrackbarPos("L-S", "Trackbars")
  l_v = cv2.getTrackbarPos("L-V", "Trackbars")
  u_h = cv2.getTrackbarPos("U-H", "Trackbars")
  u_s = cv2.getTrackbarPos("U-S", "Trackbars")
  u_v = cv2.getTrackbarPos("U-V", "Trackbars")

  lower_red = np.array([l_h, l_s, l_v])#배열 순서대로 H, S, V임
  upper_red = np.array([u_h, u_s, u_v])

  mask =cv2.inRange(hsv, lower_red, upper_red)

  cv2.imshow("Frame", frame)
  cv2.imshow("Mask", mask)

  key = cv2.waitKey(1)
  if key == 27:
    break

cap.release()
cv2.destroyWindow()