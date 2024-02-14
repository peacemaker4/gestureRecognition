import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import math
import time


#Volume control variables
from pynput.keyboard import Key, Controller
keyboard = Controller()

last_angle = None
last_length = None

minAngle = 0
maxAngle = 180
angle = 0
# angleBar = 400
# angleDeg = 0
minHand = 50  # 50
maxHand = 300  # 300
vol_mute = False



# mediapipe initialization
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Get gesture recognition model trained on dataset
model = load_model('mp_hand_gesture')

# Used gestures
gestures = ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist',
              'smile']
vol_control = False

# Webcamera initialization
cap = cv2.VideoCapture(0)

last_time = time.perf_counter()
quit_timer_on = False
quit_time = 3.0

# Loop each frame
while True:
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip frame
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    curr_gesture = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        landmarks_w_id = []
        for handslms in result.multi_hand_landmarks:
            for id, lm in enumerate(handslms.landmark):
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks_w_id.append([id, lmx, lmy])
                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            curr_gesture = gestures[classID]

        # Gestures control
        if curr_gesture == "smile":
            vol_control = True
        elif curr_gesture == "fist":
            vol_control = False
        elif curr_gesture == "peace" and not vol_control:
            keyboard.press(Key.print_screen)
            keyboard.release(Key.print_screen)

        if curr_gesture == "thumbs down":
            if quit_timer_on == False:
                last_time = time.perf_counter()
                quit_timer_on = True
        else:
            quit_timer_on = False

        # Volume control
        if vol_control == True:
            x1, y1 = landmarks_w_id[4][1], landmarks_w_id[4][2]
            x2, y2 = landmarks_w_id[8][1], landmarks_w_id[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(frame, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 15, (0, 255, 0), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(frame, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            angle = np.interp(length, [minHand, maxHand], [minAngle, maxAngle])
            # angleBar = np.interp(length, [minHand, maxHand], [400, 150])
            # angleDeg = np.interp(length, [minHand, maxHand], [0, 180])

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
                cv2.circle(frame, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
                vol_mode = "Volume mute"
                if vol_mute == False:
                    keyboard.press(Key.media_volume_mute)
                    vol_mute = True
            else:
                vol_mute = False

            cv2.putText(frame, vol_mode, (10, 100), cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (0, 0, 255), 2)
            last_angle = angle
            last_length = length

        # Quit app on timer
        if quit_timer_on == True:
            left_to_quit = round(quit_time - (time.perf_counter() - last_time), 2)
            cv2.putText(frame, str(left_to_quit), (10, 150), cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (0, 0, 255), 2)
            if left_to_quit <= 0:
                break

    # Text output of prediction
    cv2.putText(frame, curr_gesture, (10, 50), cv2.FONT_HERSHEY_PLAIN,
                2, (0, 255, 0), 2, cv2.LINE_AA)

    # Window
    cv2.imshow("GesturesRecognition", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Close
cap.release()

cv2.destroyAllWindows()