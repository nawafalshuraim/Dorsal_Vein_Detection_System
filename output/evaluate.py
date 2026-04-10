import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# DEVICE
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# DATASET
class VeinDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        img = cv2.imread(os.path.join(self.image_dir, name), 0)
        mask = cv2.imread(os.path.join(self.mask_dir, name), 0)

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        img = img / 255.0
        mask = mask / 255.0

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img, mask, name

# MODEL (same as training)
class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(1, 32)
        self.pool1 = torch.nn.MaxPool2d(2)

        self.down2 = DoubleConv(32, 64)
        self.pool2 = torch.nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up2 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1 = torch.nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.final = torch.nn.Conv2d(32, 1, 1)

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

# DICE SCORE
def dice_score(pred, target):
    pred = (pred > 0.5).float()
    smooth = 1e-5
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# LOAD MODEL
model = UNet().to(device)
model.load_state_dict(torch.load("vein_unet.pth", map_location=device))
model.eval()

print("Model loaded")

# ===============================
# DB2 DATA
dataset = VeinDataset("dataset2/images", "dataset2/masks")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

os.makedirs("predictions_db2", exist_ok=True)

# EVALUATION LOOP
dice_scores = []

with torch.no_grad():
    for img, mask, name in loader:
        img = img.to(device)
        mask = mask.to(device)

        pred = model(img)

        d = dice_score(pred, mask)
        dice_scores.append(d.item())

        # save predicted mask
        pred_np = pred.squeeze().cpu().numpy()
        pred_np = (pred_np > 0.5).astype(np.uint8) * 255

        cv2.imwrite(
            os.path.join("predictions_db2", name[0]),
            pred_np
        )

print("=================================")
print("Mean Dice on DB2:", np.mean(dice_scores))
print("=================================")
