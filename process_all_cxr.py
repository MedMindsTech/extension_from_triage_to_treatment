import os
import pydicom
import numpy as np
import cv2

input_folder = "."
output_folder = "processed_images"

os.makedirs(output_folder, exist_ok=True)

def preprocess(img):
    img = img.astype(np.float32)

    # Normalize 0 to 1
    img -= img.min()
    img /= img.max()
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # CLAHE enhancement
    img_clahe = clahe.apply((img * 255).astype(np.uint8))
   
    img_resized = cv2.resize(img_clahe, (224, 224))   # Resize to CNN input

    return img_resized

# Loop through all DICOM files
for filename in os.listdir(input_folder):
    if filename.endswith(".dcm"):
        path = os.path.join(input_folder, filename)
        print("Processing:", filename)

        ds = pydicom.dcmread(path)
        img = ds.pixel_array

        processed = preprocess(img)

        save_path = os.path.join(output_folder, filename.replace(".dcm", ".png"))
        cv2.imwrite(save_path, processed)
        print("Saved:", save_path)

print("All files processed.")
