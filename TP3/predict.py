import os
import time
import cv2
import joblib
import mediapipe as mp
import numpy as np

MODEL_FILE = "letter_model_tp3.pkl"

if not os.path.exists(MODEL_FILE):
    print(f"Model file '{MODEL_FILE}' not found.")
    exit()

model = joblib.load(MODEL_FILE)
print("Model loaded successfully.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print('Error turning on the camera')
    exit()

print('Camera initiallized!')
current_letter = ''
current_word = ''
last_prediction = ''
last_pred_time = 0
prediction_delay = 1.5

while True:
    success, image = webcam.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    detected_letter = ''
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color = (255, 0, 0), thickness = 2, circle_radius = 2),
                                   mp_draw.DrawingSpec(color = (0, 255, 0), thickness = 2, circle_radius = 2))
            
            coords = []

            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            input_features = np.array([coords])

            if input_features.shape[1] == model.n_features_in_:
                prediction = model.predict(input_features)[0]
                now = time.time()

                if prediction != last_prediction or (now - last_pred_time) >= prediction_delay:
                    current_letter = prediction
                    last_prediction = prediction
                    last_pred_time = now
            else:
                current_letter = '???'
    else:
        current_letter = '???'
    
    cv2.putText(image, f'Letter: {current_letter}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(image, f'Word: {current_word}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)
    
    cv2.imshow('Hand Gesture', image)

    key = cv2.waitKey(1)

    if key == 27: # ESC
            break
    elif key == 32:  # SPACE para agregar letra
        if current_letter and current_letter != '?':
            current_word += current_letter
            print(f'Current word: {current_word}')
    elif key == ord('s'):  # S para cerrar palabra y empezar otra
        print(f'Final word: {current_word}')
        current_word = ''

hands.close()
webcam.release()
cv2.destroyAllWindows()