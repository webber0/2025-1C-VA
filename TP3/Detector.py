import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
import joblib 

# --- CONFIGURACIÓN Y CARGA DE DATOS ---
CSV_FILE = 'dataset_letras.csv'
MODEL_SAVE_PATH = 'modelo_letras_mediapipe.pkl'

# 1. Cargar el archivo CSV
try:
    data = pd.read_csv(CSV_FILE)
    print(f"CSV cargado exitosamente. Columnas: {data.columns.tolist()}")
except FileNotFoundError:
    print(f"Error: El archivo '{CSV_FILE}' no se encontró. Asegúrate de que el CSV esté en la misma carpeta que el script.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al cargar el CSV: {e}")
    exit()

# 2. Preparar los datos para el entrenamiento
# La primera columna es la letra, el resto son las características
labels = data.iloc[:, 0]
features = data.iloc[:, 1:]

# Validar el número de columnas de características
expected_features = 21 * 3 # 21 landmarks * (x, y, z)
if features.shape[1] != expected_features:
    print(f"Advertencia: El número de columnas de características en el CSV ({features.shape[1]}) no coincide con el esperado ({expected_features}).")
    print("Por favor, revisa la estructura de tu CSV para asegurarte de que tiene 1 columna para la letra y 63 columnas para las coordenadas.")
    print("Se espera el orden: x0...x20, y0...y20, z0...z20.")

# Dividir los datos en conjuntos de entrenamiento y prueba
#X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 3. Entrenar el modelo de clasificación
model = LogisticRegression(max_iter=1000, solver='liblinear')
print("\nEntrenando el modelo...")
model.fit(X_train, y_train)
print("Modelo entrenado.")

# Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo en el conjunto de prueba: {accuracy:.2f}")

# 4. Guardar el modelo entrenado
try:
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Modelo guardado en: {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"Error al guardar el modelo: {e}")

# --- CONFIGURACIÓN DE MEDIAPIPE Y WEBCAM ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# 5. Inicializar la webcam
cap = cv2.VideoCapture(0) # 0 para la cámara predeterminada

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara. Asegúrate de que esté conectada y no esté en uso.")
    exit()

print("\nCámara inicializada. Muestra tu mano para detectar una letra. Presiona 'q' para salir.")

# --- BUCLE PRINCIPAL DE DETECCIÓN ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignorando fotogramas vacíos de la cámara.")
        continue

    # Voltear la imagen horizontalmente (efecto espejo) y convertir BGR a RGB
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Mejorar el rendimiento: marcar la imagen como no escribible antes de procesar
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True # Volver a marcar como escribible para dibujar

    detected_letter = "Detectando..."
    prediction_confidence = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar los puntos clave y las conexiones de la mano
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            coords_x = []
            coords_y = []
            coords_z = []
            
            for lm in hand_landmarks.landmark:
                coords_x.append(lm.x)
                coords_y.append(lm.y)
                coords_z.append(lm.z)

            # Unir las listas en el orden esperado
            keypoints_flat = coords_x + coords_y + coords_z
            
            # Convertir a array de NumPy para la predicción
            input_features = np.array([keypoints_flat])

            # Asegurarse de que el número de características coincida con el del entrenamiento
            if input_features.shape[1] == model.n_features_in_:
                # Realizar la predicción
                prediction = model.predict(input_features)
                detected_letter = str(prediction[0])

                # Obtener la confianza de la predicción (si el modelo lo soporta)
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(input_features)[0]
                    # Encontrar la probabilidad de la letra predicha
                    prob_index = np.where(model.classes_ == detected_letter)[0]
                    if prob_index.size > 0:
                        confidence = probabilities[prob_index[0]] * 100
                        prediction_confidence = f"Confianza: {confidence:.2f}%"
            else:
                detected_letter = f"Error: Mismatch de features ({input_features.shape[1]} vs {model.n_features_in_})"

    # Mostrar la letra detectada y la confianza en la imagen
    cv2.putText(image, detected_letter, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
    if prediction_confidence:
        cv2.putText(image, prediction_confidence, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

    # Mostrar la imagen
    cv2.imshow('Reconocimiento de Letras por Senas', image)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. Liberar recursos
hands.close()
cap.release()
cv2.destroyAllWindows()
print("Aplicación finalizada.")