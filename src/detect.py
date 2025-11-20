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

# Load Autoencoder (160x160)
ae = AutoEncoder().to(device)
ae.load_state_dict(torch.load("models/ae.pt", map_location=device))
ae.eval()

# MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=10, post_process=True, device=device)

# CNN transform with ImageNet normalization
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# AE transform (160x160 without normalization for reconstruction error calculation)
ae_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# ------------------------------------------
# Detect function
# ------------------------------------------
def detect_image(img_path):
    """
    Detect if an image contains a fake/deepfake face or is real.
    Uses ensemble of CNN and Autoencoder for better accuracy.
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to open image: {str(e)}"}

    # Face detection
    face = mtcnn(img)
    if face is None:
        return {"error": "No face detected. Try a clearer image."}

    face_pil = transforms.ToPILImage()(face)

    # FIX #3: CNN prediction with proper normalization
    cnn_img = cnn_transform(face_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        cnn_out = cnn(cnn_img)
    prob_fake = torch.softmax(cnn_out, dim=1)[0][1].item()

    # FIX #4: Autoencoder reconstruction error
    ae_img = ae_transform(face_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        ae_recon = ae(ae_img)
    ae_loss = torch.mean(torch.abs(ae_img - ae_recon)).item()

    # FIX #5: Improved normalization of AE error
    # Real images typically have lower reconstruction error (0.05-0.15)
    # Fake images have higher reconstruction error (0.15-0.30+)
    # Using better thresholds based on actual observations
    ae_norm = min(max((ae_loss - 0.08) / 0.20, 0), 1)

    # FIX #6: Better weighted ensemble decision
    # CNN is more reliable for classification, AE provides secondary signal
    final_score = (prob_fake * 0.75) + (ae_norm * 0.25)

    # FIX #7: Improved threshold - 0.35 instead of 0.15 for better balance
    # Higher threshold reduces false positives while maintaining good detection
    prediction = "FAKE" if final_score > 0.35 else "REAL"
    confidence = final_score if prediction == "FAKE" else (1 - final_score)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 3),
        "final_score": round(final_score, 3),
        "cnn_prob_fake": round(prob_fake, 3),
        "ae_error": round(ae_loss, 5),
        "ae_norm": round(ae_norm, 3)
    }

# ------------------------------------------
# Command Line Run
# ------------------------------------------
if __name__ == "__main__":
    path = input("Enter image path: ").strip()
    if os.path.exists(path):
        result = detect_image(path)
        print("\n" + "="*50)
        print("DEEPFAKE DETECTION RESULT")
        print("="*50)
        for key, value in result.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("="*50 + "\n")
    else:
        print("❌ File not found:", path)
