import cv2
import math
import pyautogui
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
width, height = int(cap.get(3)), int(cap.get(4))

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
length_prev = 0
cursor_visible = True
clicking = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            thumb_x = int(hand_landmarks.landmark[4].x * width)
            thumb_y = int(hand_landmarks.landmark[4].y * height)
            index_x = int(hand_landmarks.landmark[8].x * width)
            index_y = int(hand_landmarks.landmark[8].y * height)

            curr_x = index_x
            curr_y = index_y
            
            length = math.hypot(index_x - thumb_x, index_y - thumb_y)
            
            cv2.circle(frame, (thumb_x, thumb_y), 8, (0, 255, 0), -1)
            cv2.circle(frame, (index_x, index_y), 8, (0, 255, 0), -1)
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 2)

            if length < 30:
                if not clicking:
                    pyautogui.click()
                    clicking = True
                cv2.putText(frame, "Gesture: Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                clicking = False

            if length > length_prev:
                cv2.putText(frame, "Gesture: Move", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            screen_w, screen_h = pyautogui.size()
            pyautogui.moveTo(curr_x * screen_w / width, curr_y * screen_h / height)
            
            length_prev = length

    cv2.imshow('Gestor Control - ESC to Exit', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()