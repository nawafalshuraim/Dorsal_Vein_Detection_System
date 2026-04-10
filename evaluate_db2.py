# import os
# import cv2
# import torch
# import numpy as np
# from skimage.filters import frangi
# from skimage import img_as_float
# import torch.nn.functional as F

# # ==========================
# # DEVICE
# # ==========================
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print("Using device:", device)

# # ==========================
# # LOAD MODEL
# # ==========================
# from train import UNet  # reuse your architecture

# model = UNet().to(device)
# model.load_state_dict(torch.load("vein_unet.pth", map_location=device))
# model.eval()

# # ==========================
# # DICE FUNCTION
# # ==========================
# def dice_score(pred, target):
#     smooth = 1e-5
#     intersection = (pred * target).sum()
#     return (2. * intersection + smooth) / \
#            (pred.sum() + target.sum() + smooth)

# # ==========================
# # PARAMETERS
# # ==========================
# db2_path = "Data/DorsalHandVeins_DB2_png"
# clahe = cv2.createCLAHE(2.0, (8,8))

# frangi_threshold = 30

# dice_list = []

# # ==========================
# # EVALUATION LOOP
# # ==========================
# for filename in os.listdir(db2_path):

#     if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
#         continue

#     path = os.path.join(db2_path, filename)
#     img = cv2.imread(path, 0)

#     if img is None:
#         continue

#     # Resize
#     img_resized = cv2.resize(img, (256,256))

#     # ---- Frangi mask ----
#     cl = clahe.apply(img_resized)
#     vessel = frangi(img_as_float(cl))
#     vessel = cv2.normalize(vessel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     _, frangi_mask = cv2.threshold(vessel, frangi_threshold, 255, cv2.THRESH_BINARY)

#     frangi_mask = frangi_mask / 255.0
#     frangi_mask = torch.tensor(frangi_mask, dtype=torch.float32).to(device)

#     # ---- U-Net prediction ----
#     image_tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32)
#     image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(device)

#     with torch.no_grad():
#         pred = model(image_tensor)

#     pred = (pred > 0.5).float().squeeze()

#     # ---- Dice ----
#     d = dice_score(pred, frangi_mask)
#     dice_list.append(d.item())

# # ==========================
# # RESULTS
# # ==========================
# mean_dice = np.mean(dice_list)
# std_dice = np.std(dice_list)

# print("Mean Dice (U-Net vs Frangi):", mean_dice)
# print("Std Dice:", std_dice)
