import os
import sys
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
import timm
from torch import nn

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Autoencoder Model
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160 * 160 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 160 * 160 * 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 3, 160, 160)

# Load Models
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("Loading models...")

try:
    cnn = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    cnn.load_state_dict(torch.load("models/cnn_efficientnet_b0.pt", map_location=device))
    cnn = cnn.to(device)
    cnn.eval()
    print("OK CNN loaded")
except Exception as e:
    print(f"ERROR CNN: {e}")
    print("Train first: python src/train_cnn.py")
    sys.exit(1)

try:
    ae = AutoEncoder().to(device)
    ae.load_state_dict(torch.load("models/ae.pt", map_location=device))
    ae.eval()
    print("OK Autoencoder loaded")
except Exception as e:
    print(f"ERROR AE: {e}")
    print("Train first: python src/train_ae.py")
    sys.exit(1)

try:
    mtcnn = MTCNN(image_size=160, margin=10, post_process=True, device=device)
    print("OK MTCNN loaded")
except Exception as e:
    print(f"ERROR MTCNN: {e}")
    sys.exit(1)

# Transforms
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

ae_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

print("\nAll models ready!\n")

def detect_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to open image: {e}"}

    try:
        face = mtcnn(img)
    except Exception as e:
        return {"error": f"Face detection failed: {e}"}
    
    if face is None:
        return {"error": "No face detected in image"}

    try:
        face_pil = transforms.ToPILImage()(face)
    except Exception as e:
        return {"error": f"Face conversion failed: {e}"}

    # CNN prediction
    try:
        cnn_img = cnn_transform(face_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            cnn_out = cnn(cnn_img)
        prob_fake = torch.softmax(cnn_out, dim=1)[0][1].item()
    except Exception as e:
        return {"error": f"CNN failed: {e}"}

    # AE prediction
    try:
        ae_img = ae_transform(face_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            ae_recon = ae(ae_img)
        ae_loss = torch.mean(torch.abs(ae_img - ae_recon)).item()
    except Exception as e:
        return {"error": f"Autoencoder failed: {e}"}

    ae_norm = min(max((ae_loss - 0.08) / 0.20, 0), 1)
    final_score = (prob_fake * 0.75) + (ae_norm * 0.25)
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

if __name__ == "__main__":
    print("="*60)
    print("DEEPFAKE DETECTION")
    print("="*60)
    
    path = input("\nImage path: ").strip()
    
    if not path or not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    
    print(f"Processing...\n")
    result = detect_image(path)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("="*60)
        print(f"RESULT: {result['prediction']}")
        print("="*60)
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"Final Score: {result['final_score']}")
        print(f"CNN Fake Prob: {result['cnn_prob_fake']*100:.1f}%")
        print(f"AE Error: {result['ae_error']}")
        print(f"AE Normalized: {result['ae_norm']}")
        print("="*60 + "\n")
