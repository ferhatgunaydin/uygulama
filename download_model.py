import gdown
import os

def download_model_if_not_exists():
    model_path = "models/classifier_model.keras"
    if not os.path.exists(model_path):
        print("Model indiriliyor...")
        os.makedirs("models", exist_ok=True)
        url = "https://drive.google.com/uc?id=1kHGgS0R7jisu5FYAhO58Ju0jZ2ARTGx2"
        gdown.download(url, model_path, quiet=False)
    else:
        print("Model zaten mevcut.")

if __name__ == "__main__":
    download_model_if_not_exists()
