import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Donde se guardarán los datos
archivo_csv = "dataset_letras.csv"
if not os.path.exists(archivo_csv):
    columnas = ['letra'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)]
    df = pd.DataFrame(columns=columnas)
    df.to_csv(archivo_csv, index=False)
    with open(archivo_csv) as f:
        print(f.read())
# Iniciar webcam
cap = cv2.VideoCapture(0)
print("Presioná una tecla (A-Z) para guardar la seña actual como esa letra. ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Espejar imagen
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Capturar landmarks
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            # Mostrar en pantalla que puede guardar
            cv2.putText(frame, "Presiona una tecla A-Z para guardar", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    key = cv2.waitKey(1)
    cv2.putText(frame, f"Letra Leida {key}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Captura de signos", frame)

    if key == 27:  # ESC
        break

    elif results.multi_hand_landmarks and 65 <= key <= 90:  # A-Z
        letra = chr(key)
        if landmark_list:
           print("entró")
           df = pd.read_csv(archivo_csv) #if os.path.getsize(archivo_csv) > 0 else pd.DataFrame()
           fila = [letra] + landmark_list
           df.loc[len(df)] = fila
           df.to_csv(archivo_csv, index=False)
           print(f"Seña guardada como: {letra}")

cap.release()
cv2.destroyAllWindows()