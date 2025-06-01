import gdown
import os

# Klasörü oluştur
os.makedirs("models", exist_ok=True)

# Classifier model
clf_url = "https://drive.google.com/uc?id=1kHGgS0R7jisu5FYAhO58Ju0jZ2ARTGx2"
clf_output = "models/classifier_model.keras"

if not os.path.exists(clf_output):
    print("İnme sınıflandırma modeli indiriliyor...")
    gdown.download(clf_url, clf_output, quiet=False)
else:
    print("Model zaten mevcut.")
