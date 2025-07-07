import os
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_DIR_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "splits/train"
VAL_DIR = BASE_DIR / "splits/val"
MODEL_OUT = BASE_DIR_ROOT / "models/modelo_basico_v1.h5"

# Hiperparámetros
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001

# Generadores de datos
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Definición del modelo CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento
checkpoint = ModelCheckpoint(str(MODEL_OUT), save_best_only=True, monitor='val_accuracy', mode='max')

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)
