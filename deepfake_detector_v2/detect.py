import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import sys

class FakeDetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        return self.resnet(x)

def detect(image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = FakeDetectorCNN().to(device)
    model_path = 'deepfake_detector_v2/models/detector.pt'
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Train first: python deepfake_detector_v2/train_simple.py")
        return None
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load and preprocess image
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return None
    
    img = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        confidence = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
    
    # Results
    fake_confidence = confidence[0][1].item() * 100
    real_confidence = confidence[0][0].item() * 100
    
    result = {
        'prediction': 'FAKE' if prediction == 1 else 'REAL',
        'fake_confidence': fake_confidence,
        'real_confidence': real_confidence
    }
    
    return result

def main():
    print("\n" + "="*60)
    print("DEEPFAKE DETECTOR v2 - Simple & Accurate")
    print("="*60)
    
    image_path = input("\nEnter image path: ").strip()
    
    if not image_path:
        print("No path provided")
        return
    
    print("\nAnalyzing...")
    result = detect(image_path)
    
    if result:
        print("\n" + "="*60)
        print(f"RESULT: {result['prediction']}")
        print("="*60)
        
        if result['prediction'] == 'FAKE':
            print(f"\n🚨 DEEPFAKE DETECTED!")
            print(f"   Confidence: {result['fake_confidence']:.2f}%")
        else:
            print(f"\n✓ REAL IMAGE")
            print(f"   Confidence: {result['real_confidence']:.2f}%")
        
        print(f"\nDetailed Scores:")
        print(f"  Fake Score: {result['fake_confidence']:.2f}%")
        print(f"  Real Score: {result['real_confidence']:.2f}%")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()
