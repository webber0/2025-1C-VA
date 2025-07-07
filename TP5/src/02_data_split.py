import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Config
BASE_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_DIR = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "splits"
SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train, val, test

# Set seed for reproducibility
random.seed(42)

# Crear carpetas destino
for split in ['train', 'val', 'test']:
    for class_name in os.listdir(INPUT_DIR):
        class_dir = INPUT_DIR / class_name
        if not class_dir.is_dir():
            continue
        target_dir = OUTPUT_DIR / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

# Repartir las imágenes
for class_name in tqdm(os.listdir(INPUT_DIR), desc="Dividiendo datos"):
    class_path = INPUT_DIR / class_name
    if not class_path.is_dir():
        continue

    images = list(class_path.glob("*.jpg"))
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * SPLIT_RATIOS[0])
    n_val = int(n_total * SPLIT_RATIOS[1])

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    for img_list, split in zip([train_imgs, val_imgs, test_imgs], ['train', 'val', 'test']):
        for img_path in img_list:
            dest = OUTPUT_DIR / split / class_name / img_path.name
            shutil.copy(img_path, dest)
