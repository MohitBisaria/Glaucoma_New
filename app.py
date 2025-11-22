import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from torch import nn
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import joblib
import streamlit_authenticator as stauth
from fpdf import FPDF
import tempfile
import os

from m3_feature_extractor import extract_m3_features
import sys
import os
import zipfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
M3_PATH = os.path.join(BASE_DIR, "M3")
M3_ZIP_PATH = os.path.join(BASE_DIR, "M3.zip")

# If M3 folder doesn't exist but zip exists → unzip it
if not os.path.exists(M3_PATH) and os.path.exists(M3_ZIP_PATH):
    with zipfile.ZipFile(M3_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)

# Add M3 folder to Python path
if M3_PATH not in sys.path:
    sys.path.append(M3_PATH)

st.set_page_config(layout="wide")


# ------------- PDF CLASS ------------------
class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 15)
        self.cell(0, 10, "Glaucoma Screening Report", border=0, ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", border=0, ln=True, align="C")


def create_report_pdf(patient_info, original_img, overlay_img, metrics_df, metrics_fig, prediction_label, confidence):
    pdf = PDF("P", "mm", "A4")
    pdf.add_page()

    # Patient Info
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Patient Details", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for key, value in patient_info.items():
        pdf.cell(40, 8, f"{key}:", border=0)
        pdf.cell(0, 8, str(value), ln=True)

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, f"Prediction: {prediction_label}    |   Confidence: {confidence:.2%}", border=1, ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Screening Images", ln=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_o, tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_m:
        cv2.imwrite(tmp_o.name, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(tmp_m.name, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        pdf.image(tmp_o.name, x=15, w=80)
        pdf.image(tmp_m.name, x=110, w=80)

    pdf.ln(80)

    # Features Table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Extracted Retinal Features", ln=True)
    pdf.set_font("Helvetica", "", 7)

    for col, val in metrics_df.iloc[0].items():
        pdf.cell(60, 6, str(col), border=1)
        pdf.cell(0, 6, str(val), border=1, ln=True)

    return pdf.output(dest="S").encode("latin1")


# --------------- User Authentication ------------------
config = {
    'credentials': {
        'usernames': {
            'testuser': {
                'email': 'test@user.com',
                'name': 'Test User',
                'password': '$2b$12$pMQfhnxFyeKAUJ6IYOBsC.LU/RRQELL9jrpfa3o6j3U39GnaQj4oy'
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'glaucoma_key',
        'name': 'glaucoma_cookie'
    }
}

authenticator = stauth.Authenticate(
    config['credentials'], config['cookie']['name'],
    config['cookie']['key'], config['cookie']['expiry_days']
)


# --------------- Load Models --------------------
@st.cache_resource
def load_resources():
    processor = AutoImageProcessor.from_pretrained("pamixsun/segformer_for_optic_disc_cup_segmentation")
    seg_model = SegformerForSemanticSegmentation.from_pretrained(
        "pamixsun/segformer_for_optic_disc_cup_segmentation"
    )
    seg_model.eval()

    try:
        clf = joblib.load("xgboost_new.pkl")  # <-- Corrected
    except FileNotFoundError:
        st.error("🔴 Missing xgboost_new.pkl file.")
        clf = None

    return processor, seg_model, clf


# --------------- Image Processing --------------------
def process_image(image, processor, seg_model, clf):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Segmentation
    inputs = processor(rgb, return_tensors="pt")
    with torch.no_grad():
        logits = seg_model(**inputs).logits.cpu()
    upsampled = nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear")
    pred = upsampled.argmax(dim=1)[0].numpy().astype(np.uint8)

    disc_mask, cup_mask = (pred == 1), (pred == 2)
    overlay = rgb.copy()
    overlay[disc_mask] = [255, 255, 0]
    overlay[cup_mask] = [255, 0, 0]

    # Extract Features
    features_df = extract_m3_features(rgb)

    pred_label, confidence = "N/A", np.nan

    if clf is not None:
        # Align features correctly
        try:
            expected_features = clf.feature_names_in_
        except AttributeError:
            try:
                expected_features = clf.best_estimator_.feature_names_in_
            except Exception:
                expected_features = list(features_df.columns[:24])

        model_input = features_df.reindex(columns=expected_features, fill_value=0)

        prob = clf.predict_proba(model_input)[0]
        idx = np.argmax(prob)
        pred_label = "Glaucoma" if idx == 1 else "Normal"
        confidence = prob[idx]

    return rgb, overlay, features_df, pred_label, confidence


# ------------------- Main App --------------------
if not st.session_state.get("authentication_status"):
    authenticator.login()
else:
    processor, seg_model, clf = load_resources()

    st.title("🩺 AI-Based Glaucoma Detection and Retinal Analysis")
    authenticator.logout("Logout", "sidebar")

    st.subheader("Patient Details")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Patient Name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col2:
        age = st.number_input("Age", min_value=1, max_value=100)
        comorbidities = st.multiselect("Comorbidities", ["Diabetes", "Hypertension", "BP"])

    st.subheader("Upload Retinal Fundus Image")
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("🔍 Analyzing Image..."):
            original, overlay, features_df, pred_label, confidence = process_image(
                image, processor, seg_model, clf
            )

        col1, col2 = st.columns(2)
        col1.image(original, caption="Original Fundus Image", use_container_width=True)
        col2.image(overlay, caption="Segmented Disc & Cup", use_container_width=True)

        st.subheader(f"🧠 Prediction: **{pred_label}**")
        if isinstance(confidence, (float, np.floating)):
            st.write(f"📊 Confidence: **{confidence*100:.2f}%**")

        st.subheader("Extracted Features")
        st.dataframe(features_df.style.format("{:.3f}"))

        # Feature Chart
        top_features = features_df.iloc[0].sort_values(ascending=False)[:15]
        fig = go.Figure(data=[go.Bar(x=top_features.index, y=top_features.values)])
        fig.update_layout(title="Top 15 Feature Values", xaxis_tickangle=-45)
        st.plotly_chart(fig)

        patient_info = {
            "Name": name,
            "Age": age,
            "Gender": gender,
            "Comorbidities": ", ".join(comorbidities) if comorbidities else "None",
        }

        pdf_bytes = create_report_pdf(
            patient_info,
            original,
            overlay,
            features_df,
            fig,
            pred_label,
            confidence
        )

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"Glaucoma_Report_{name}.pdf",
            mime="application/pdf"
        )
