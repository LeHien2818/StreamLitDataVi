import streamlit as st
import numpy as np
import requests
import base64
import os
from PIL import Image
import json
from io import BytesIO
import zipfile

st.set_page_config(page_title="Image Classification Demo", layout="wide")

API_ENDPOINT = "http://34.142.220.207:8000/api/image-classification"
DATA_FOLDER = os.path.join("C:", "Users", "PCM", "AppData", "Local", "Temp", "task_bundle_zkytvh3t", "image_classification", "data")
CONFIG_FOLDER = os.path.join("C:", "Users", "PCM", "AppData", "Local", "Temp", "task_bundle_zkytvh3t", "image_classification")
LABEL_MAPPING_FILE = os.path.join(CONFIG_FOLDER, "label_mapping.json")

@st.cache_data
def load_label_mapping():
    try:
        with open(LABEL_MAPPING_FILE, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)
        return label_mapping
    except Exception as e:
        st.error(f"Error loading label mapping: {e}")
        return {}

label_mapping = load_label_mapping()

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

def prepare_payload(image_base64: str) -> dict:
    return {"data": image_base64}

def call_api(image_base64: str):
    payload = prepare_payload(image_base64)
    try:
        response = requests.post(API_ENDPOINT, json=payload)
        response.raise_for_status()
        response_json = response.json()
        data = response_json.get("data", None)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], list):
                return data[0]
        return data
    except Exception as e:
        st.error(f"API request failed: {e}")
        return None

def get_top_prediction(logits: list):
    logits_array = np.array(logits)
    probs = softmax(logits_array)
    top_idx = np.argmax(probs)
    top_prob = probs[top_idx]
    label = label_mapping.get(str(top_idx), f"Class {top_idx}")
    return label, top_prob

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def main():
    st.title("Image Classification Demo")
    st.write("Upload a single image or a folder of images to classify objects into ImageNet classes.")

    uploaded_files = None
    upload_type = st.radio("Select upload method:", ("Single Image", "Image Folder"))

    images = []

    if upload_type == "Single Image":
        uploaded_files = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_files is not None:
            try:
                image = Image.open(uploaded_files).convert("RGB")
                images.append((uploaded_files.name, image))
            except Exception as e:
                st.error(f"Error opening image: {e}")
    else:
        folder = st.file_uploader("Upload a folder of images (zip)", type=["zip"])
        if folder is not None:
            try:
                with zipfile.ZipFile(BytesIO(folder.read())) as z:
                    for filename in z.namelist():
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                            with z.open(filename) as file:
                                img = Image.open(file).convert("RGB")
                                images.append((filename, img))
            except Exception as e:
                st.error(f"Error processing zip file: {e}")

    if images:
        st.write(f"Processing {len(images)} image(s)...")
        results = []

        for filename, img in images:
            with st.spinner(f"Classifying {filename}..."):
                try:
                    img_base64 = encode_image_to_base64(img)
                    logits = call_api(img_base64)
                    if logits is None:
                        continue
                    label, probability = get_top_prediction(logits)
                    results.append((filename, img, label, probability))
                except Exception as e:
                    st.error(f"Error processing {filename}: {e}")

        for filename, img, label, prob in results:
            st.subheader(f"Prediction for {filename}")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img, use_container_width=True)
            with col2:
                st.write(f"**Predicted Label:** {label}")
                st.write(f"**Probability:** {prob:.4f}")

if __name__ == "__main__":
    main()