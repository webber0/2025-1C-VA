from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

BASE_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_DIR_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "splits/train"
VAL_DIR = BASE_DIR / "splits/val"
MODEL_OUT = BASE_DIR_ROOT / "models/modelo_mejorado.h5"

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10

# Generador con data augmentation para entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generador de validación sin augmentation
val_datagen = ImageDataGenerator(rescale=1./255)

# Cargar datos desde carpetas
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


### Defino la CNN Mejorada

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  # salida igual al número de clases
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

model.summary()

### Entrenamiento con validacion

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

model.save(MODEL_OUT)
print("Modelo guardado en models/modelo_mejorado.h5")