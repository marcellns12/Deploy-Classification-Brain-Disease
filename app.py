import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2

# ============================
# LOAD MODEL
# ============================
model = pickle.load(open("svm_model.pkl", "rb"))

# Label mapping (sesuaikan jika berbeda)
label_names = {0: "Tumor", 1: "Cancer", 2: "Aneurysm"}

# ============================
# CUSTOM CSS
# ============================
page_bg_img = """
<style>
body {
    background: #0f0f0f;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    color: #00eaff;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    margin-top: -10px;
    color: #cccccc;
}

.upload-box {
    border: 2px dashed #00eaff;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    background: rgba(255,255,255,0.03);
}

.pred-box {
    background: #1c1c1c;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    box-shadow: 0 0 15px rgba(0, 234, 255, 0.3);
}

.label-tag {
    padding: 8px 20px;
    border-radius: 50px;
    font-size: 20px;
    color: black;
    font-weight: 700;
}

.tumor { background: #ff0059; }
.cancer { background: #ffaa00; }
.aneurysm { background: #00ff9d; }
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ============================
# TITLE
# ============================
st.markdown("<div class='main-title'>🧠 Brain Disease Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Tumor • Cancer • Aneurysm</div>", unsafe_allow_html=True)
st.write("")

# ============================
# PREPROCESSING FUNCTION
# (SAMA PERSIS DGN TRAINING)
# ============================
def preprocess_image(img_file, image_size=(64, 64)):
    # Jika DICOM (.dcm)
    if img_file.name.lower().endswith(".dcm"):
        dicom_data = pydicom.dcmread(img_file)
        img_array = apply_voi_lut(dicom_data.pixel_array, dicom_data)

        if dicom_data.PhotometricInterpretation == "MONOCHROME1":
            img_array = np.amax(img_array) - img_array

        img = Image.fromarray(img_array).convert("L")
        img = img.resize(image_size)
        img_array = np.array(img)

    else:
        # JPG / PNG
        img = Image.open(img_file).convert("L")
        img = img.resize(image_size)
        img_array = np.array(img)

    # Flatten + Normalize (sesuai training)
    features = img_array.flatten() / 255.0
    return features.reshape(1, -1)


# ============================
# FILE UPLOAD
# ============================
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
img_file = st.file_uploader("Upload MRI/CT Scan (JPG/PNG/DICOM)", type=["jpg", "jpeg", "png", "dcm"])
st.markdown("</div>", unsafe_allow_html=True)

# ============================
# PREDICTION
# ============================
if img_file:
    # Tampilkan preview (hanya JPG/PNG)
    if not img_file.name.lower().endswith(".dcm"):
        img = Image.open(img_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
    else:
        st.info("DICOM file uploaded. Processing...")

    if st.button("🔍 Predict"):
        processed = preprocess_image(img_file)
        pred = model.predict(processed)[0]
        label = label_names[pred]

        # Warna indikator
        color_class = "tumor" if label == "Tumor" else "cancer" if label == "Cancer" else "aneurysm"

        # OUTPUT BOX
        st.markdown(f"""
            <div class="pred-box">
                <h3 style="text-align:center;">Prediction Result</h3>
                <div style="text-align:center; margin-top:15px;">
                    <span class="label-tag {color_class}">{label}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
