import streamlit as st
import numpy as np
import pickle
import cv2
from PIL import Image
import pydicom
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# =========================
# Load Model
# =========================
svm_model = pickle.load(open("svm_model.pkl", "rb"))

# =========================
# Load Dataset Function
# =========================
def load_data(image_size, folder):
    data = []
    labels = []

    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)

        if not os.path.isdir(label_path):
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            # Load JPG
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = Image.open(img_path).convert('L')
                img = img.resize(image_size)
                img_array = np.array(img).flatten()

            # Load DICOM
            elif img_name.lower().endswith('.dcm'):
                dicom_data = pydicom.dcmread(img_path)
                img_array = dicom_data.pixel_array

                if len(img_array.shape) > 2:
                    img_array = img_array[:, :, 0]

                img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                img_array = cv2.resize(img_array, image_size)
                img_array = img_array.flatten()

            else:
                continue

            data.append(img_array / 255.0)
            labels.append(label)

    return np.array(data), np.array(labels)

# =========================
# Load Data
# =========================
extracted_folder = 'brain_dataset'
image_size = (64, 64)
data, labels = load_data(image_size, extracted_folder)

# =========================
# Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# =========================
# Label Encoding
# =========================
def encode_labels(labels):
    le = LabelEncoder()
    encoded = le.fit_transform(labels)
    return encoded, le

y_train_encoded, label_encoder = encode_labels(y_train)
y_test_encoded, _ = encode_labels(y_test)

# =========================
# Predict Single Image
# =========================
def predict_image(uploaded_file):
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        # JPG / PNG
        if file_extension in ['jpg', 'jpeg', 'png']:
            img = Image.open(uploaded_file).convert('L')
            img = img.resize(image_size)
            img_array = np.array(img).flatten() / 255.0

        # DICOM
        elif file_extension == 'dcm':
            dicom_data = pydicom.dcmread(uploaded_file)
            img_array = dicom_data.pixel_array

            if len(img_array.shape) > 2:
                img_array = img_array[:, :, 0]

            img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_array = cv2.resize(img_array, image_size)
            img_array = img_array.flatten() / 255.0

        else:
            return "Unsupported file format"

        # Prediction
        prediction = svm_model.predict(np.expand_dims(img_array, axis=0))
        decoded = label_encoder.inverse_transform(prediction)

        return decoded[0]

    except Exception as e:
        return f"Error processing file: {e}"

# =========================
# STREAMLIT UI
# =========================
st.title("Brain Disease Classification using SVM")
st.write("Upload JPG/PNG/DICOM to predict brain tumor category.")

uploaded_file = st.file_uploader("Upload medical image", type=["jpg", "jpeg", "png", "dcm"])

if uploaded_file is not None:
    st.success("File uploaded!")

    prediction = predict_image(uploaded_file)

    st.subheader("Prediction Result:")
    st.write(prediction)
