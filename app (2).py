
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

# Download model from Google Drive if not exists
model_path = "Custom_CNN (1).h5"
file_id = "1c9Rsky1DmCUsHO-rTRfzejvZQ5nV8Ukh"
url = f"https://drive.google.com/uc?id={file_id}"
if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

# Load the trained model
model = load_model(model_path)

# Define class information
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

# Streamlit UI
st.title("Kidney Condition Classification")
st.write("Upload up to 4 kidney CT scan images to classify their condition.")

uploaded_files = st.file_uploader("Choose up to 4 images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Load and display the image
        img = Image.open(uploaded_file)
        st.image(img, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

        # Preprocess the image
        def preprocess_image(img):
            img = img.resize((224, 224))  # Resize to model input size
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            img = img / 255.0  # Normalize pixel values
            return img

        img_array = preprocess_image(img)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = max(CLASS_INFO.keys(), key=lambda c: prediction[0][list(CLASS_INFO.keys()).index(c)])
        confidence = np.max(prediction) * 100  # Convert to percentage

        # Display classification results
        st.write(f"### Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Display condition details
        condition_info = CLASS_INFO[predicted_class]
        st.write(f"**Description:** {condition_info['description']}")

        # Symptoms
        st.write("**Symptoms:")
        for symptom in condition_info["symptoms"]:
            st.write(f"- {symptom}")

        # Diagnosis methods
        st.write("**Diagnosis Measures:**")
        for measure in condition_info["diagnosis"]:
            st.write(f"- {measure}")

        # Treatment suggestions
        st.write("**Treatment Suggestions:**")
        for treatment in condition_info["treatment"]:
            st.write(f"- {treatment}")
        st.write("---")
        