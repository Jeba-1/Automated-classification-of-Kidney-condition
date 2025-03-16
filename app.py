
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import gdown
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

# Google Drive file IDs
file_id1 = "1-F1b1rqhcwwyOJoOgkrDi4EAfBSb17rB"
# Local model file paths
model_path = "kidney_model.keras"
# ✅ Download models if not already present
if not os.path.exists(model_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id1}", model_path, quiet=False)
  
# ------------------- Load the Model Safely -------------------
@st.cache_resource
def load_kidney_model():
    try:
        if os.path.exists(model_path):
            model = load_model(model_path, compile=False)
        else:
            st.error("❌ Model file not found! Please check the file path.")
            return None
        st.success("")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_kidney_model()

# ------------------- Function to Make Predictions -------------------
def predict_image(img):
    try:
        inputs = preprocess_image_for_model(img)

        if model is not None:
            prediction = model.predict(inputs)
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

# ------------------- Class Information -------------------
CLASS_INFO = {
    "Cyst": {
        "description": "Cystic kidney disease involves fluid-filled sacs in the kidney that may require monitoring or treatment.",
        "symptoms": [
            "Pain in the back or side",
            "High blood pressure",
            "Frequent urination",
            "Blood in urine"
        ],
        "diagnosis": [
            "Ultrasound",
            "CT scan",
            "MRI",
            "Kidney function tests"
        ],
        "treatment": [
            "Regular monitoring with imaging tests",
            "Medications for pain relief and blood pressure control",
            "Drainage procedures for large cysts",
            "Surgery in severe cases"
        ]
    },
    "Normal": {
        "description": "The kidney appears normal with no visible abnormalities.",
        "symptoms": ["No symptoms (healthy kidney function)"],
        "diagnosis": ["Routine medical checkup"],
        "treatment": ["Maintain a healthy lifestyle", "Drink plenty of water", "Regular medical checkups"]
    },
    "Stone": {
        "description": "Kidney stones are mineral deposits that may cause pain and require treatment.",
        "symptoms": [
            "Severe lower back or abdominal pain",
            "Blood in urine",
            "Frequent urge to urinate",
            "Nausea and vomiting"
        ],
        "diagnosis": [
            "CT scan",
            "X-ray",
            "Urine tests",
            "Ultrasound"
        ],
        "treatment": [
            "Increased water intake to help flush out small stones",
            "Pain relievers",
            "Medications to break down or pass stones",
            "Shock wave therapy (ESWL) for larger stones",
            "Surgical removal in severe cases"
        ]
    },
    "Tumor": {
        "description": "A kidney tumor might indicate malignancy or benign growth. Further testing is needed to determine the severity.",
        "symptoms": [
            "Blood in urine",
            "Abdominal pain",
            "Unexplained weight loss",
            "Fatigue",
            "Fever"
        ],
        "diagnosis": [
            "CT scan",
            "MRI",
            "Biopsy",
            "Blood tests"
        ],
        "treatment": [
            "Surgical removal (nephrectomy for malignant tumors)",
            "Targeted therapy or immunotherapy for cancerous tumors",
            "Radiation therapy in some cases",
            "Regular follow-up imaging"
        ]
    }
}

# ------------------- Streamlit App UI -------------------
st.title("🩺 Automated Classification of Kidney Condition")
st.write("Upload kidney CT scan images to classify their condition.")


# ✅ Process each uploaded image separately
# Initialize session state for buttons and predictions per image
if "predictions" not in st.session_state:
    st.session_state.predictions = {}

# Upload Images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(img, caption=f"Uploaded Image {i+1}: {uploaded_file.name}", use_column_width=True)

        # Unique session key for each image
        img_key = f"image_{i}"

        # Initialize session state for this image
        if img_key not in st.session_state.predictions:
            st.session_state.predictions[img_key] = {"label": None, "selected_section": None}

        # Prediction Button
        if st.button(f"🔍 Predict Image {i+1}", key=f"predict_{i}"):
            label, confidence = predict_image(img)
            if label:
                st.session_state.predictions[img_key]["label"] = label  # Store predicted label
                st.session_state.predictions[img_key]["selected_section"] = None  # Reset selection
                st.success(f"🩺 Prediction: {label} ({confidence * 100:.2f}%)")
                st.write(f"**Description:** {CLASS_INFO[label]['description']}")
            else:
                st.error("❌ Model not loaded. Please check the error messages above.")

        # ✅ Show extra information only when a prediction is made
        if st.session_state.predictions[img_key]["label"]:
            label = st.session_state.predictions[img_key]["label"]

            # Buttons for extra information
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"🛑 Symptoms {i+1}", key=f"symptoms_{i}"):
                    st.session_state.predictions[img_key]["selected_section"] = "symptoms"
            with col2:
                if st.button(f"🩻 Diagnosis {i+1}", key=f"diagnosis_{i}"):
                    st.session_state.predictions[img_key]["selected_section"] = "diagnosis"
            with col3:
                if st.button(f"💊 Treatment {i+1}", key=f"treatment_{i}"):
                    st.session_state.predictions[img_key]["selected_section"] = "treatment"

            # ✅ Show selected section
            if st.session_state.predictions[img_key]["selected_section"]:
                section = st.session_state.predictions[img_key]["selected_section"]
                
                if section == "symptoms":
                    st.write("### Symptoms:")
                    for symptom in CLASS_INFO[label]["symptoms"]:
                        st.write(f"- {symptom}")

                elif section == "diagnosis":
                    st.write("### Diagnosis Measures:")
                    for measure in CLASS_INFO[label]["diagnosis"]:
                        st.write(f"- {measure}")

                elif section == "treatment":
                    st.write("### Treatment Suggestions:")
                    for treatment in CLASS_INFO[label]["treatment"]:
                        st.write(f"- {treatment}")

                st.write("---")  # Separator for clarity
                
