import cv2
import mediapipe as mp
import time

img = cv2.imread("../../../hand_finger_detection/images/photo_5312237266950803807_y.jpg")
img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
 
pTime = 0
cTime = 0

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = hands.process(imgRGB)
 
if results.multi_hand_landmarks:
    for handLms in results.multi_hand_landmarks:
        for id, lm in enumerate(handLms.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
 
        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
 
cTime = time.time()
fps = 1 / (cTime - pTime)
pTime = cTime
 
cv2.imshow("Hand Detection", img)
cv2.waitKey(0)