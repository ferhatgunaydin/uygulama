import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Ayarlar
img_size = 224
batch_size = 32
epochs = 10

# Görsel ön işleme ve veri artırma
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Eğitim ve doğrulama verisi yükleme
train_gen = train_datagen.flow_from_directory(
    'data/classification',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    'data/classification',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# CNN model mimarisi
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary sınıflama
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model eğitimi
model.fit(train_gen, validation_data=val_gen, epochs=epochs)

# Modeli kaydet
os.makedirs("models", exist_ok=True)
model.save("models/classifier_model.keras")

print("✅ Model başarıyla eğitildi ve kaydedildi.")