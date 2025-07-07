import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Parámetros
BASE_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_DIR = BASE_DIR / "original"
OUTPUT_DIR = BASE_DIR / "processed"
TARGET_SIZE = (128, 128)  # o (224, 224) si usan modelos preentrenados
CHANNELS = 3  # o 1 para grises

def preprocess_image(img_path):
    try:
        img = cv2.imread(str(img_path))

        if img is None:
            raise ValueError(f"Imagen no cargada: {img_path}")

        # Redimensionar
        img = cv2.resize(img, TARGET_SIZE)

        # Normalizar (0-1)
        img = img.astype(np.float32) / 255.0

        # Escala de grises si se desea (y luego convertir a 3 canales si necesario)
        if CHANNELS == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)

        return img
    except Exception as e:
        print(f"Error procesando {img_path}: {e}")
        return None

def preprocess_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for category in os.listdir(INPUT_DIR):
        category_path = INPUT_DIR / category
        output_path = OUTPUT_DIR / category
        output_path.mkdir(parents=True, exist_ok=True)

        for img_file in tqdm(os.listdir(category_path), desc=f"Procesando {category}"):
            img_path = category_path / img_file
            output_file = output_path / img_file

            if output_file.exists():
                continue

            img = preprocess_image(img_path)
            if img is not None:
                # Convertimos la imagen a uint8 antes de guardar
                img_to_save = (img * 255).astype(np.uint8)
                if CHANNELS == 1:
                    img_to_save = img_to_save.squeeze()
                cv2.imwrite(str(output_file), img_to_save)

if __name__ == "__main__":
    preprocess_dataset()