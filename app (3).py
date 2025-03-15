
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

# ✅ Fix: Image Preprocessing Function
def preprocess_image_for_model(img):
    img_size = 224  # Match model input size
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return [img, img]  # Return list for both CNN and ViT inputs

# ------------------- Load the Model Safely -------------------
@st.cache_resource
def load_kidney_model():
    try:
        if os.path.exists("/content/kidney_model.keras"):
            model = load_model("/content/kidney_model.keras", compile=False)
        elif os.path.exists("/content/kidney_model.h5"):
            model = load_model("/content/kidney_model.h5", compile=False)
        else:
            st.error("❌ Model file not found! Please check the file path.")
            return None
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_kidney_model()

# ------------------- Function to Make Predictions -------------------
def predict_image(img):
    try:
        # Process image for both CNN and ViT
        inputs = preprocess_image_for_model(img)  

        # Ensure model is loaded
        if model is not None:
            prediction = model.predict(inputs)  # Pass both inputs
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)

            # Class labels
            classes = ["Cyst", "Normal", "Stone", "Tumor"]

            return classes[predicted_class], confidence
        else:
            return None, None
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
        return None, None

# ------------------- Streamlit App UI -------------------
st.title("🩺 Kidney Condition Classification")
st.write("Upload a kidney CT scan image to classify its condition.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        label, confidence = predict_image(img)
        if label:
            st.success(f"🩺 Prediction: {label} ({confidence * 100:.2f}%)")
        else:
            st.error("❌ Model not loaded. Please check the error messages above.")
