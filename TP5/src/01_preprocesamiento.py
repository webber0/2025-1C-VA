import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuración
IMG_SIZE = 224
INPUT_PATH = Path("data/original")
OUTPUT_PATH = Path("data/procesada")
CLASSES = ['carton', 'vidrio', 'metal', 'papel', 'plastico', 'basura']
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_SEED = 42

def preparar_directorios(base):
    for tipo in ['train', 'val', 'test']:
        for clase in CLASSES:
            path = base / tipo / clase
            path.mkdir(parents=True, exist_ok=True)

def redimensionar_y_guardar(imagen_path, destino):
    img = cv2.imread(str(imagen_path))
    if img is None:
        return
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    destino_path = destino / imagen_path.name
    cv2.imwrite(str(destino_path), img)

def main():
    preparar_directorios(OUTPUT_PATH)
    for clase in CLASSES:
        imagenes = list((INPUT_PATH / clase).glob("*.jpg"))
        train_val, test = train_test_split(imagenes, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        train, val = train_test_split(train_val, test_size=VAL_SIZE, random_state=RANDOM_SEED)

        for img_path in train:
            redimensionar_y_guardar(img_path, OUTPUT_PATH / "train" / clase)
        for img_path in val:
            redimensionar_y_guardar(img_path, OUTPUT_PATH / "val" / clase)
        for img_path in test:
            redimensionar_y_guardar(img_path, OUTPUT_PATH / "test" / clase)

    print("Preprocesamiento finalizado.")

if __name__ == "__main__":
    main()