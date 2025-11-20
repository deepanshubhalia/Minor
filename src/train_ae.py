import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# ------------------------------
# Dataset Loader
# ------------------------------
class FaceDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# ------------------------------
# Correct Autoencoder (160×160)
# ------------------------------
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),                       # 3×160×160 → 76800
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


# ------------------------------
# Training Function
# ------------------------------
def train_ae():
    real_dir = "data/faces_real"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training Autoencoder on device:", device)

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
    ])

    dataset = FaceDataset(real_dir, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = AutoEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    for epoch in range(3):
        total_loss = 0

        for imgs in dataloader:
            imgs = imgs.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/3] Loss: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/ae.pt")
    print("✓ Autoencoder saved at models/ae.pt")


if __name__ == "__main__":
    train_ae()
