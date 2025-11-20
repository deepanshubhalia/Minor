import os
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import random

class FakeDetectorCNN(nn.Module):
    """ResNet-based detector for deepfakes"""
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
            nn.Linear(128, 2)  # binary: fake or real
        )
    
    def forward(self, x):
        return self.resnet(x)

class FaceDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        # Real faces (label=0)
        for f in os.listdir(real_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.samples.append((os.path.join(real_dir, f), 0))
        
        # Fake faces (label=1)
        for f in os.listdir(fake_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.samples.append((os.path.join(fake_dir, f), 1))
        
        random.shuffle(self.samples)
        print(f"Loaded {len(self.samples)} images: {sum(1 for _, l in self.samples if l==0)} real, {sum(1 for _, l in self.samples if l==1)} fake")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Paths
    real_dir = "data/faces_real"
    fake_dir = "data/faces_fake"
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"ERROR: Create data folders:")
        print(f"  - {real_dir}/")
        print(f"  - {fake_dir}/")
        return
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset
    dataset = FaceDataset(real_dir, fake_dir, transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    # Model
    model = FakeDetectorCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f} Acc: {accuracy:.2f}%")
    
    # Save
    os.makedirs('deepfake_detector_v2/models', exist_ok=True)
    torch.save(model.state_dict(), 'deepfake_detector_v2/models/detector.pt')
    print("Model saved to deepfake_detector_v2/models/detector.pt")

if __name__ == "__main__":
    train()
