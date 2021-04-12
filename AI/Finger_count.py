import cv2
import time
import os
import AI.HandTrackingModule as htm
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)

cap.set(3, wCam)
cap.set(4, hCam)


#Fingerimg 폴더를 list형태로 가져옴
folderPath = 'need'
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:

    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
pTime = 0

#미리 만들어 놓은 HandTrackingModule를 사용
detector = htm.handDetecotr(detectionCon=0.75)
#tipIds 는 손가락에 끝 위치점들
tipIds = [4,8,12,16,20]

while True:

    success, img = cap.read()
    img = detector.findHands(img)

    #Hand 좌표 값 나타내기기
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)


    #finger open으로 손이 펴졌다고 나타내기
    if len(lmList) !=0:
        # fingers 이용해서 list로 몇번 째 손가락이 펴졌는지 인식
        fingers = []

        #Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1),
        else:
            fingers.append(0)
        #4 Fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1),
            else:
                fingers.append(0)
        #print(fingers)
        totalFingers = fingers.count(1)
        
        print(totalFingers)

        #이미지를 사이즈에 맞게 적용
        #lins count 값으로 숫자 형태로 나옴
        h,w,c = overlayList[totalFingers-1].shape
        img[0:h, 0:w]= overlayList[totalFingers-1]

        cv2.rectangle(img,(20,225),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img, str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,
                    10,(255,0,0),25)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #FPS
    cv2.putText(img, f'FPS: {int(fps)}',(400,70), cv2.FONT_HERSHEY_PLAIN,
                3,(255,0,0),3)
    cv2.imshow('Image',img)
    cv2.waitKey(1)
