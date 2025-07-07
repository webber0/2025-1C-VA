import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path

# === CONFIGURACIÓN ===
BASE_DIR_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR_ROOT / "models/modelo_mejorado.h5"

IMG_SIZE = (150, 150)  # debe coincidir con el input_shape
CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']  # ajustar según las clases reales

# === CARGAR MODELO ===
model = load_model(MODEL_PATH)
print(f"Modelo cargado desde {MODEL_PATH}")

# === INICIAR CÁMARA ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

print("Presioná 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocesar el frame
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predicción
    preds = model.predict(img)
    class_id = np.argmax(preds)
    class_name = CLASSES[class_id]
    confidence = np.max(preds)

    # Mostrar resultados
    label = f"{class_name} ({confidence * 100:.2f}%)"
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("Clasificador de Residuos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
