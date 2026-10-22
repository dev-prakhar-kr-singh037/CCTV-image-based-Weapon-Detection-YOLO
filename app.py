import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Weapon Detection System",
    page_icon="🔫",
    layout="wide"
)

# ---------------------------
# Load YOLO Model
# ---------------------------
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

# ---------------------------
# Title
# ---------------------------
st.title("🔫 Weapon Detection System")
st.markdown("Upload an image to detect weapons using YOLOv8.")

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------
# Prediction Section
# ---------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("Detect Weapon"):

        with st.spinner("Running Detection..."):

            results = model(image)

            annotated_image = results[0].plot()

        with col2:
            st.subheader("Detection Result")
            st.image(
                annotated_image,
                use_container_width=True
            )

        boxes = results[0].boxes

        st.markdown("---")
        st.subheader("Detection Summary")

        if len(boxes) == 0:

            st.success("✅ No Weapon Detected")

        else:

            st.error(
                f"⚠️ Object Detected: {len(boxes)}"
            )

            for i, box in enumerate(boxes, start=1):

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                st.write(
                    f"""
                    Detection {i}
                    - Class: **{class_name}**
                    - Confidence: **{confidence:.2%}**
                    """
                )