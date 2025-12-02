import streamlit as st
import pickle
import numpy as np
from PIL import Image
import cv2
import base64

# ============ LOAD MODEL ============
model = pickle.load(open("svm_model.pkl", "rb"))

# ============ CUSTOM CSS ============
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

# ============ TITLE ============
st.markdown("<div class='main-title'>🧠 Brain Disease Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Tumor • Cancer • Aneurysm</div>", unsafe_allow_html=True)
st.write("")

# ============ FILE UPLOAD ============
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
img_file = st.file_uploader("Upload MRI/CT Scan (JPG/PNG)", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

# ============ PREDICTION ============
label_names = {0: "Tumor", 1: "Cancer", 2: "Aneurysm"}

def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(1, -1)
    return img

if img_file:
    img = Image.open(img_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        processed = preprocess_image(img)
        pred = model.predict(processed)[0]
        label = label_names[pred]

        color_class = ""
        if label == "Tumor":
            color_class = "tumor"
        elif label == "Cancer":
            color_class = "cancer"
        else:
            color_class = "aneurysm"

        # Hasil Prediction Box
        st.markdown(f"""
            <div class="pred-box">
                <h3 style="text-align:center;">Prediction Result</h3>
                <div style="text-align:center; margin-top:15px;">
                    <span class="label-tag {color_class}">{label}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
