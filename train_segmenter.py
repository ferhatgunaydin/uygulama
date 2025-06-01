import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam

# Ayarlar
IMG_HEIGHT = 256
IMG_WIDTH = 256

def load_data(image_dir, mask_dir):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        if not filename.endswith(".png"):
            continue
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

def build_unet_model():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))

    c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(128, 3, activation='relu', padding='same')(c3)

    u4 = UpSampling2D()(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(64, 3, activation='relu', padding='same')(u4)
    c4 = Conv2D(64, 3, activation='relu', padding='same')(c4)

    u5 = UpSampling2D()(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(32, 3, activation='relu', padding='same')(u5)
    c5 = Conv2D(32, 3, activation='relu', padding='same')(c5)

    outputs = Conv2D(1, 1, activation='sigmoid')(c5)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Veri yükle
X, Y = load_data("data/segment/images", "data/segment/masks")
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

# Modeli oluştur ve eğit
model = build_unet_model()
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=8)

# Kaydet
os.makedirs("models", exist_ok=True)
model.save("models/segmenter_model.h5")
print("✅ Segmentasyon modeli başarıyla kaydedildi.")
