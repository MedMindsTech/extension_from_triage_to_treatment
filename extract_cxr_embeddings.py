import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

# Folder with processed PNGs
input_folder = "processed_images"
output_file = "cxr_embeddings.npy"

model = models.densenet121(weights="IMAGENET1K_V1") # Load DenseNet121 pretrained on ImageNet
model.classifier = torch.nn.Identity()   # remove classifier to get embeddings
model.eval()

# Image transformation (same as ImageNet)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

embeddings = []
filenames = []

# Loops through processed CXRs
for file in os.listdir(input_folder):
    if file.endswith(".png"):
        path = os.path.join(input_folder, file)
        img = Image.open(path).convert("RGB")

        x = transform(img).unsqueeze(0)   # shape (1,3,224,224)

        with torch.no_grad():
            feat = model(x)              # DenseNet embedding
            feat = feat.squeeze().numpy() # (1024,)

        embeddings.append(feat)
        filenames.append(file)

        print("Extracted:", file)

embeddings = np.array(embeddings)  # Convert to numpy array

np.save(output_file, embeddings)   # Saves to file

print("\nSaved embeddings to", output_file)
print("Embedding shape:", embeddings.shape)
print("Number of images:", len(filenames))
