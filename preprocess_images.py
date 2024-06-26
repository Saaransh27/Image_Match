import torch
import open_clip
import cv2
import os
from PIL import Image
import pickle
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)

def imageEncoder(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_transformed = preprocess(img_pil).to(device)
    encoded_img = model.encode_image(img_transformed.unsqueeze(0))
    return encoded_img

def preprocess_folder(image_folder):
    encodings = {}
    image_files = os.listdir(image_folder)
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            encoded_img = imageEncoder(img)
            encodings[image_file] = encoded_img.cpu().detach().numpy()
    return encodings

image_folder = "Class_1"  # Change this to your folder path
print(f"Starting preprocessing for images in {image_folder}...")
encodings = preprocess_folder(image_folder)

with open('image_encodings.pkl', 'wb') as f:
    pickle.dump(encodings, f)

print("Preprocessing complete. Encodings saved to image_encodings.pkl.")