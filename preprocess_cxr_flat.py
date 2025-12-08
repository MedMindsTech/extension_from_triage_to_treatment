import os
import pydicom
import numpy as np
import cv2

input_folder = "raw_dicom"          
output_folder = "processed_images"

os.makedirs(output_folder, exist_ok=True)

def preprocess(img):
    img = img.astype(np.float32)

    # Normalize to 0–1
    img -= img.min()
    img /= (img.max() + 1e-7)

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply((img * 255).astype(np.uint8))

    # Resize for models like DenseNet
    img_resized = cv2.resize(img_clahe, (224, 224))

    return img_resized

print("Scanning raw_dicom/ for DICOM files...")

count = 0

for file in os.listdir(input_folder):
    if file.lower().endswith(".dcm"):
        filepath = os.path.join(input_folder, file)

        print("Processing:", file)

        # Load DICOM
        ds = pydicom.dcmread(filepath)
        img = ds.pixel_array

        # Preprocess
        processed_img = preprocess(img)

        # Save PNG with the SAME base filename
        png_name = file.replace(".dcm", ".png")
        save_path = os.path.join(output_folder, png_name)
        cv2.imwrite(save_path, processed_img)

        count += 1

print(f"\nAll DICOM files processed. Total: {count}")
