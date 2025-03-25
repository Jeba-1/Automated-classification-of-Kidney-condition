import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os
from fpdf import FPDF

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
st.title("🩺 Automated Classification of Kidney Condition ")
st.write("Upload a kidney CT scan image to classify its condition.")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to model input size
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values
    return img

# Process each uploaded image
if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file)
        st.image(img, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)

        if f"prediction_{i}" not in st.session_state:
            st.session_state[f"prediction_{i}"] = None

        if st.button("🔍 Predict", key=f"predict_{i}"):
            img_array = preprocess_image(img)
            prediction = model.predict(img_array)
            predicted_class = max(CLASS_INFO.keys(), key=lambda c: prediction[0][list(CLASS_INFO.keys()).index(c)])
            confidence = np.max(prediction) * 100  # Convert to percentage
            st.session_state[f"prediction_{i}"] = (predicted_class, confidence)

        if st.session_state[f"prediction_{i}"]:
            predicted_class, confidence = st.session_state[f"prediction_{i}"]
            st.write(f"### Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}%")

            if st.button("📜 Description", key=f"description_{i}"):
                st.write(f"**Description:** {CLASS_INFO[predicted_class]['description']}")

            if st.button("🛑 Symptoms", key=f"symptoms_{i}"):
                st.write("**Symptoms:**")
                for symptom in CLASS_INFO[predicted_class]["symptoms"]:
                    st.write(f"* {symptom}")

            if st.button("🩻 Diagnosis", key=f"diagnosis_{i}"):
                st.write("**Diagnosis Measures:**")
                for measure in CLASS_INFO[predicted_class]["diagnosis"]:
                    st.write(f"* {measure}")

            if st.button("💊 Treatment", key=f"treatment_{i}"):
                st.write("**Treatment Suggestions:**")
                for treatment in CLASS_INFO[predicted_class]["treatment"]:
                    st.write(f"* {treatment}")

            class PDF(FPDF):
                def footer(self):
                    # Draw border on every page
                    self.rect(5, 5, 200, 287)
            def generate_pdf():
                pdf = PDF()
                pdf.add_page()

                # Title
                pdf.set_font("Times", style='B', size=16)
                pdf.cell(200, 10, "Kidney Condition Report", ln=True, align='C')
                pdf.ln(10)

                # Add Image
                image_path = f"uploaded_image_{i}.jpg"
                img.save(image_path)
                pdf.image(image_path, x=60, y=30, w=90)
                pdf.ln(80)

                # Prediction Result
                pdf.set_font("Times", style='B', size=14)
                pdf.cell(200, 10, "Prediction Condition", ln=True)
                pdf.set_font("Times", size=12)
                pdf.cell(200, 10, f"Prediction: {predicted_class}", ln=True)
                pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True)
                pdf.ln(4)

                # Add sections with horizontal lines
                sections = [
                    ("Description", CLASS_INFO[predicted_class]['description']),
                    ("Symptoms", CLASS_INFO[predicted_class]['symptoms']),
                    ("Diagnosis Measures", CLASS_INFO[predicted_class]['diagnosis']),
                    ("Treatment Suggestions", CLASS_INFO[predicted_class]['treatment']),
                ]

                for title, content in sections:
                    pdf.set_font("Times", style='B', size=14)
                    pdf.cell(200, 10, title, ln=True)
                    pdf.set_font("Times", size=12)
                   
                    if isinstance(content, list):  # Symptoms, diagnosis, treatment are lists
                        for item in content:
                            pdf.cell(200, 10, f"- {item}", ln=True)
                    else:  # Description is a string
                        pdf.multi_cell(0, 10, content)

                    pdf.ln(4)
                    pdf.line(10, pdf.get_y(), 200, pdf.get_y())  # Draw horizontal line
                    pdf.ln(5)

                # Save and return the PDF file
                pdf_path = f"report_{i}.pdf"
                pdf.output(pdf_path)
                return pdf_path
                
            if st.button("📥 Download Report", key=f"download_{i}"):
                pdf_path = generate_pdf()
                with open(pdf_path, "rb") as file:
                    st.download_button(
                        label="Download Report as PDF",
                        data=file,
                        file_name=f"Kidney_Condition_Report_{i}.pdf",
                        mime="application/pdf"
                    )

