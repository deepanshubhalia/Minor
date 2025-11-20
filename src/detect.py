import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
import timm
from train_ae import AutoEncoder


# ------------------------------------------
# Load Models
# ------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Load CNN model
cnn = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
cnn.load_state_dict(torch.load("models/cnn_efficientnet_b0.pt", map_location=device))
cnn = cnn.to(device)
cnn.eval()

# Load Autoencoder (correct 160x160)
ae = AutoEncoder().to(device)
ae.load_state_dict(torch.load("models/ae.pt", map_location=device))
ae.eval()

# MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=10, post_process=True, device=device)

# CNN transform
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# AE transform (must be EXACT 160×160)
ae_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])


# ------------------------------------------
# Detect function
# ------------------------------------------
def detect_image(img_path):
    img = Image.open(img_path).convert("RGB")

    # Face detection
    face = mtcnn(img)
    if face is None:
        return {"error": "No face detected. Try a clearer image."}

    face_pil = transforms.ToPILImage()(face)

    # CNN prediction
    cnn_img = cnn_transform(face_pil).unsqueeze(0).to(device)
    cnn_out = cnn(cnn_img)
    prob_fake = torch.softmax(cnn_out, dim=1)[0][1].item()

    # Autoencoder reconstruction error
    ae_img = ae_transform(face_pil).unsqueeze(0).to(device)
    ae_recon = ae(ae_img)
    ae_loss = torch.mean(torch.abs(ae_img - ae_recon)).item()

    # Normalize AE error — based on actual observed range
    # Real images typically have lower reconstruction error
    # Fake images have higher reconstruction error
    # Map to 0-1 scale where higher = more fake-like
    ae_norm = min(max((ae_loss - 0.10) / 0.15, 0), 1)

    # Weighted decision - rely more on CNN (better classifier)
    # CNN is more reliable, AE helps as secondary signal
    final_score = (prob_fake * 0.90) + (ae_norm * 0.10)
    
    # Lower threshold to catch more fakes - 0.15 instead of 0.25
    # If CNN says >15% fake probability, mark as FAKE
    prediction = "FAKE" if final_score > 0.15 else "REAL"

    return {
        "cnn_prob_fake": round(prob_fake, 3),
        "ae_error": round(ae_loss, 5),
        "ae_norm": round(ae_norm, 3),
        "final_score": round(final_score, 3),
        "prediction": prediction
    }


# ------------------------------------------
# Command Line Run
# ------------------------------------------
if __name__ == "__main__":
    path = input("Enter image path: ").strip()

    if os.path.exists(path):
        print(detect_image(path))
    else:
        print("❌ File not found:", path)