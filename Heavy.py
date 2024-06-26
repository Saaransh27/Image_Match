import streamlit as st
import torch
import open_clip
import cv2
import os
from PIL import Image
from sentence_transformers import util
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)

def imageEncoder(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_transformed = preprocess(img_pil).to(device)
    encoded_img = model.encode_image(img_transformed.unsqueeze(0))
    return encoded_img

def generateScore(image1, encodings):
    img1 = imageEncoder(image1)
    similarity_scores = {}

    for image_file, encoded_img2 in encodings.items():
        img2 = torch.tensor(encoded_img2).to(device)
        cos_scores = util.pytorch_cos_sim(img1, img2)
        score = round(float(cos_scores[0][0]) * 100, 2)
        similarity_scores[image_file] = score

    return similarity_scores

st.title("Image Similarity Scoring")

image1_upload = st.file_uploader("Upload the first image", type=["jpg", "jpeg", "png"])
image2_folder_path = st.text_input("Enter the folder path containing the comparison images")

if image1_upload and image2_folder_path:
    if not os.path.exists("temp"):
        os.makedirs("temp")
        
    image1_path = os.path.join("temp", image1_upload.name)
    with open(image1_path, "wb") as f:
        f.write(image1_upload.getbuffer())
    
    image1 = cv2.imread(image1_path, cv2.IMREAD_UNCHANGED)
    
    if image1 is not None and os.path.isdir(image2_folder_path):
        with open('image_encodings.pkl', 'rb') as f:
            encodings = pickle.load(f)

        scores_dict = generateScore(image1, encodings)
        sorted_scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
        top_score_file, top_score = sorted_scores[0]

        st.image(image1, caption="Uploaded Image 1", use_column_width=True)
        
        st.write("Top 10 Similarity Scores:")
        for image_file, score in sorted_scores[:10]:
            top_image_path = os.path.join(image2_folder_path, image_file)
            top_image = cv2.imread(top_image_path, cv2.IMREAD_UNCHANGED)
            st.image(top_image, caption=f"{image_file} with score {score}", use_column_width=True)
    else:
        st.error("Invalid image or folder path")