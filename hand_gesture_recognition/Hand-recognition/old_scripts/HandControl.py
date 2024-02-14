import cv2
import numpy as np
import TrackingMod as htm
import math

from pynput.keyboard import Key, Controller

keyboard = Controller()

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)

last_angle = None
last_length = None

minAngle = 0
maxAngle = 180
angle = 0
angleBar = 400
angleDeg = 0
minHand = 50  # 50
maxHand = 300  # 300
vol_mute = False

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0, 255, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        angle = np.interp(length, [minHand, maxHand], [minAngle, maxAngle])
        angleBar = np.interp(length, [minHand, maxHand], [400, 150])
        angleDeg = np.interp(length, [minHand, maxHand], [0, 180])

        vol_mode = "None"

        if last_length:
            if vol_mute == False:
                if length > last_length:
                    keyboard.press(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up)
                    vol_mode = "Volume up"

                elif length < last_length:
                    keyboard.press(Key.media_volume_down)
                    keyboard.release(Key.media_volume_down)
                    vol_mode = "Volume down"
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
            vol_mode = "Volume mute"
            if vol_mute == False:
                keyboard.press(Key.media_volume_mute)
                vol_mute = True
        else:
            vol_mute = False

        cv2.putText(img, vol_mode, (10, 50), cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0), 2)

        last_angle = angle
        last_length = length

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(angleBar)), (85, 400), (0, 255, 0), cv2.FILLED)

    cv2.imshow("Control", img)
    cv2.waitKey(1)