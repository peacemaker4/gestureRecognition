import cv2
import mediapipe as mp
import time
 
cap = cv2.VideoCapture(0)
 
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
 
pTime = 0
cTime = 0

x = ""
y = ""
z = ""
 
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.BORDER_REFLECT)
            x = str(round(handLms.landmark[0].x, 3))
            y = str(round(handLms.landmark[0].y, 3))
            z = str(round(handLms.landmark[0].z, 3))
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
 
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime


    if str(results.multi_hand_landmarks) == "None":
        cv2.putText(img, str("Not detected"), (10, 50), cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0), 2)
    else:
        cv2.putText(img, "x: " + str(x), (10, 40), cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0), 2)
        cv2.putText(img, "y: " + str(y), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0), 2)
        cv2.putText(img, "z: " + str(z), (10, 100), cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0), 2)

    cv2.imshow("Hand Detection", img)
    cv2.waitKey(1)