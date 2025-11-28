import os
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt

input_folder = "."      # current folder
output_folder = "processed_images"

os.makedirs(output_folder, exist_ok=True)

def preprocess(img):
    img = img.astype(np.float32)

    # Normalize 0 to 1
    img -= img.min()
    img /= img.max()

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply((img * 255).astype(np.uint8))

    # Resize for CNN input
    img_resized = cv2.resize(img_clahe, (224, 224))

    return img_resized

# Loop through all files in folder
for file in os.listdir(input_folder):
    if file.endswith(".dcm"):
        filepath = os.path.join(input_folder, file)

        print("Processing:", file)

     
        ds = pydicom.dcmread(filepath)   # Load DICOM
        img = ds.pixel_array

        processed_img = preprocess(img) # Preprocess

        
        save_path = os.path.join(output_folder, file.replace(".dcm", ".png")) # Save as PNG
        cv2.imwrite(save_path, processed_img)

        print("Saved:", save_path)

print("\nAll DICOM files processed.")
