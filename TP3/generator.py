import os
import cv2
import mediapipe as mp
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = False,
                       max_num_hands = 1,
                       min_detection_confidence = 0.5,
                       min_tracking_confidence = 0.5)

mp_drawing = mp.solutions.drawing_utils

csv_file = "dataset_letters.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    columns = ['letter'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)]
    df = pd.DataFrame(columns=columns)

webcam = cv2.VideoCapture(0)
print("Press a key (A - Z) to save the sign as that letter. Press ESC to exit.")

while True:
    success, frame = webcam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            cv2.putText(frame, "Press a key (A-Z) to save", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    key = cv2.waitKey(1)
    if key != -1 and (65 <= key <= 90 or 97 <= key <= 122):
        letter_display = chr(key).upper()
    else:
        letter_display = ''
    cv2.putText(frame, f"Read letter: {letter_display}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Hand gesture caption", frame)

    if key == 27: # ESC
        break

    elif results.multi_hand_landmarks and (65 <= key <= 90 or 97 <= key <= 122): # A-Z
        letter = chr(key).upper()
        if landmark_list:
            print("Entro")
            row = [letter] + landmark_list
            df.loc[len(df)] = row
            print(f"Gesture saved as: {letter}")
            print(row)

df.to_csv(csv_file, index = False)
webcam.release()
cv2.destroyAllWindows()