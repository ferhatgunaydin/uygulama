import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanIoU

IMG_HEIGHT = 256
IMG_WIDTH = 256

def load_data(image_dir, mask_dir):
    images, masks = [], []
    for fname in os.listdir(image_dir):
        if not fname.endswith(".png"):
            continue
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Veri yükle
X, Y = load_data("data/segment/images", "data/segment/masks")
_, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2)

# Modeli yükle
model = load_model("models/segmenter_model.h5")

# Tahmin
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5).astype(np.uint8)
Y_test = (Y_test > 0.5).astype(np.uint8)

# IoU (MeanIoU)
miou = MeanIoU(num_classes=2)
miou.update_state(Y_test, Y_pred)
iou_score = miou.result().numpy()

# Dice Score
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-7) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-7)

dice_scores = [dice_coef(Y_test[i], Y_pred[i]) for i in range(len(Y_test))]
mean_dice = np.mean(dice_scores)

print(f"📊 Segmentasyon Performansı:")
print(f"- Ortalama IoU:  {iou_score:.4f}")
print(f"- Ortalama Dice: {mean_dice:.4f}")
