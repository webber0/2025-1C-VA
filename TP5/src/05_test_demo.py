import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_DIR_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR_ROOT / "models/modelo_mejorado.h5"
TEST_DIR = BASE_DIR / "splits/test/"

# 1. Cargar el modelo entrenado
model = load_model(MODEL_PATH)
print("✅ Modelo cargado con éxito.")

# 2. Preprocesamiento para test
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 3. Evaluación
print("🔍 Evaluando en el set de test...")
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes
labels = list(test_generator.class_indices.keys())

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels))

# 4. Matriz de Confusión
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()
input("Presioná Enter para cerrar la ventana...")
