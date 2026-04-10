import numpy as np
import cv2
import os
from skimage.filters import frangi
from skimage import img_as_float # since Frangi expects float images, not uint8.

# Paths
input_folder = "data/DorsalHandVeins_DB1_png"
output_folder = "output/DB1_Final"   # vein detection overlays
mask_folder   = "output/DB2_Final"   # binary masks

os.makedirs(output_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)

# Parameters (tunable)
clahe_clip = 2.0 # controls contrast amplification
clahe_grid = (8,8)

frangi_threshold = 45 # lower value: detect more faint veins, Higher value: stricter detection
adaptive_block = 25 # must be odd, larger block more global behavior.
adaptive_C = 6 # subtracts constant to fine-tune sensitivity

min_area = 500 # removes small contours (noise). Anything smaller than 500 pixels is ignored.

kernel_open = np.ones((5,5), np.uint8) # 5×5 erases thin hair strands (hair ~1-3px, veins ~5-15px)
kernel_close = np.ones((7,7), np.uint8) # 7×7 for connecting broken veins

clahe = cv2.createCLAHE(
    clipLimit=clahe_clip,
    tileGridSize=clahe_grid
)

# Processing Loop
all_files = [f for f in os.listdir(input_folder)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for filename in all_files[:5]:

    path = os.path.join(input_folder, filename)
    img = cv2.imread(path, 0)

    if img is None:
        continue

    # CLAHE
    cl = clahe.apply(img)

    # Gentle blur to suppress thin hair strands before thresholding
    cl = cv2.GaussianBlur(cl, (5, 5), 0)

    # Adaptive Threshold
    adaptive = cv2.adaptiveThreshold(
        cl,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block,
        adaptive_C
    )

    # Frangi (Faint Veins)
    img_float = img_as_float(cl)
    vessel = frangi(img_float)

    vessel = cv2.normalize(
        vessel, None, 0, 255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    _, vessel_thresh = cv2.threshold(
        vessel,
        frangi_threshold,
        255,
        cv2.THRESH_BINARY
    )

    # Combine Both Methods
    combined = cv2.bitwise_or(adaptive, vessel_thresh)

    # Remove Background
    _, background_mask = cv2.threshold(
        img, 15, 255, cv2.THRESH_BINARY
    )

    combined = cv2.bitwise_and(combined, background_mask)

    # Morphology Cleaning
    opened = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        kernel_open
    )

    cleaned = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    # Remove small disconnected dots from mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    clean_mask = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255
    cleaned = clean_mask

    # Save mask to DB2_Final
    cv2.imwrite(os.path.join(mask_folder, filename), cleaned)

    # Contours
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    filtered = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            smoothed = cv2.approxPolyDP(cnt, epsilon=1.5, closed=True)
            filtered.append(smoothed)

    # If filtering removed everything, fallback to original contours
    if len(filtered) == 0:
        filtered = contours

    result = cv2.drawContours(
        img_color,
        filtered,
        -1,
        (255,0,0),
        1
    )

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, result)

print("Vein detection completed.")
