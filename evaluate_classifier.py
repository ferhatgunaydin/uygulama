import os
import numpy as np
import cv2
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

IMG_DIR = "data/classification"
IMG_SIZE = 224

model = load_model("models/classifier_model.keras")

X = []
y = []

for label, class_name in enumerate(["negative", "positive"]):
    folder = os.path.join(IMG_DIR, class_name)
for fname in os.listdir(folder):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue  # klasör veya desteklenmeyen dosya değilse geç
    path = os.path.join(folder, fname)
    img = cv2.imread(path)
    if img is None:
        print(f"Uyarı: {path} okunamadı, atlanıyor.")
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    X.append(img)
    y.append(label)

X = np.array(X)
y = np.array(y)

y_pred_probs = model.predict(X).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)

print("📊 Sınıflandırma Performans Raporu:\n")
print(classification_report(y, y_pred, target_names=["negative", "positive"]))
