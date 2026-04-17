import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===============================
# DEVICE
# ===============================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ===============================
# DATASET
# ===============================
class VeinDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))  # sorted for stability

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path, 0)

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        image = image / 255.0
        mask = mask / 255.0

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask

# ===============================
# MODEL
# ===============================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(1, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))

        b = self.bottleneck(self.pool2(d2))

        u2 = self.up2(b)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return torch.sigmoid(self.final(u1))

# ===============================
# LOSS FUNCTIONS
# ===============================
def dice_loss(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# ===============================
# TRAINING FUNCTION
# ===============================
def train_model():

    dataset = VeinDataset("Dataset/images", "Dataset/masks")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # smaller LR = more stable

    epochs = 15

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for img, mask in loader:
            img = img.to(device)
            mask = mask.to(device)

            pred = model(img)

            loss = F.binary_cross_entropy(pred, mask) + dice_loss(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "vein_unet.pth")
    print("Model saved.")

# ===============================
# MAIN GUARD (IMPORTANT FIX)
# ===============================
if __name__ == "__main__":
    train_model()
