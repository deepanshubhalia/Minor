import os
import random
import torch
import timm
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# ------------------------------------
# Dataset Loader
# ------------------------------------
class FaceDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.samples = []

        # Real (label = 0)
        for f in os.listdir(real_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.samples.append((os.path.join(real_dir, f), 0))

        # Fake (label = 1)
        for f in os.listdir(fake_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.samples.append((os.path.join(fake_dir, f), 1))

        random.shuffle(self.samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label).long()


# ------------------------------------
# Training Function
# ------------------------------------
def train_cnn():

    # ✅ YOUR EXACT ABSOLUTE PATHS  
    real_dir = "C:/Users/devan/OneDrive/Desktop/Minor/data/faces_real"
    fake_dir = "C:/Users/devan/OneDrive/Desktop/Minor/data/faces_fake"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training CNN on device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    dataset = FaceDataset(real_dir, fake_dir, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # EfficientNet-B0
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Training Loop
    for epoch in range(5):
        total_loss = 0
        correct = 0

        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        accuracy = correct / len(dataset)
        print(f"Epoch [{epoch+1}/5] Loss: {total_loss:.4f} Acc: {accuracy:.4f}")

    # Save Model
    os.makedirs("C:/Users/devan/OneDrive/Desktop/Minor/models", exist_ok=True)
    save_path = "C:/Users/devan/OneDrive/Desktop/Minor/models/cnn_efficientnet_b0.pt"
    torch.save(model.state_dict(), save_path)

    print(f"✓ CNN saved at: {save_path}")


# ------------------------------------
# Run Script
# ------------------------------------
if __name__ == "__main__":
    train_cnn()
