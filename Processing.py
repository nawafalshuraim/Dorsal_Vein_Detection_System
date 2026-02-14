import numpy as np
import cv2
import os
from skimage.filters import frangi
from skimage import img_as_float # since Frangi expects float images, not uint8.

# Paths
input_folder = "data/DorsalHandVeins_DB1_png"
output_folder = "output/DB1_Final"

os.makedirs(output_folder, exist_ok=True)

# Parameters (tunable)
clahe_clip = 2.0 # controls contrast amplification
clahe_grid = (8,8)

frangi_threshold = 30 # lower value: detect more faint veins, Higher value: stricter detection
adaptive_block = 25 # must be odd, larger block more global behavior.
adaptive_C = 2 # subtracts constant to fine-tune sensitivity

min_area = 200 # removes small contours (noise). Anything smaller than 200 pixels is ignored.

kernel_open = np.ones((3,3), np.uint8) # 3×3 for noise removal
kernel_close = np.ones((5,5), np.uint8) # 5×5 for connecting broken veins

clahe = cv2.createCLAHE(
    clipLimit=clahe_clip,
    tileGridSize=clahe_grid
)

# Processing Loop
for filename in os.listdir(input_folder):

    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(input_folder, filename)
    img = cv2.imread(path, 0)

    if img is None:
        continue

    # CLAHE
    cl = clahe.apply(img)

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
            filtered.append(cnt)

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
