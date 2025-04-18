import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. Préparation des données ===
CLASS_NAME = "bottle"  # <-- change ici
DATA_PATH = f"mvtec_anomaly_detection/{CLASS_NAME}/train/good"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = torchvision.datasets.ImageFolder(root=f"mvtec_anomaly_detection/{CLASS_NAME}/train", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# === 2. Définir l’AutoEncoder simple ===
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),  # 64 -> 128
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

model = AutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# === 3. Entraînement ===
for epoch in range(10):  
    model.train()
    total_loss = 0
    for img, _ in dataloader:
        img = img.to(device)
        output = model(img)
        loss = criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# === 4. Génération d’images synthétiques ===
model.eval()
os.makedirs("generated_images", exist_ok=True)

with torch.no_grad():
    for i, (img, _) in enumerate(dataloader):
        img = img.to(device)
        output = model(img)
        for j in range(output.shape[0]):
            gen_img = output[j].cpu()
            torchvision.utils.save_image(gen_img, f"generated_images/image_{i*32 + j}.png")

print("✅ Images synthétiques générées dans le dossier 'generated_images'")
