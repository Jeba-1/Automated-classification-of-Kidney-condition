# Automated-classification-of-Kidney-condition-using-deep-learning-with-real-time-API-deployment
# Overview
This project focuses on automated classification of kidney conditions using deep learning. The model is trained on CT scan images to classify kidney conditions into four categories:
Cyst (3,709 images)
Normal (5,077 images)
Tumor (2,283 images)
Stone (1,377 images)

The system is deployed as a real-time web app using Streamlit, enabling users to upload kidney CT scan images and receive condition predictions instantly.
# Features
✔ Multi-Class Classification: Distinguishes between four kidney conditions.
✔ Hybrid CNN Model: Achieves high accuracy with optimized architecture.
✔ Performance Metrics: Includes accuracy, confusion matrix, classification report, and loss/accuracy plots.
✔ Real-Time App: Deployed on Streamlit Cloud for easy accessibility.

# Dataset & Preprocessing
The dataset includes kidney CT scan images labeled into four categories.
Images are preprocessed, augmented, and normalized to improve model performance.
The dataset is split into training, validation, and test sets for robust evaluation.
# Model Architecture
The model is based on a Hybrid CNN (Convolutional Neural Network) approach, optimized for kidney condition classification. Five different deep learning models were implemented for comparison, and the best-performing model was selected.

# Evaluation Metrics
Accuracy: Over 97.22% achieved.
Confusion Matrix: Shows class-wise performance.
Classification Report: Precision, Recall, F1-score for each class.
Loss & Accuracy Plot: Visualizes model training progress.

# Deployment
The model is deployed using Streamlit for real-time inference.
Accessible at: https://automated-classification-of-kidney-condition-jebapriya.streamlit.app/

# Installation & Usage
1. Clone the repository:
git clone https://github.com/your-repo/kidney-condition-classification.git
cd kidney-condition-classification
2. Install dependencies:
pip install -r requirements.txt
3. Run the Streamlit app:
streamlit run app.py
4. Upload a kidney CT scan image & get predictions!
   
# Results & Insights
Achieved a high classification accuracy (>90%).
The system can assist radiologists and healthcare professionals in diagnosis.

# Future Enhancements
🔹 Fine-tuning the model with Vision Transformers (ViTs).
🔹 Expanding the dataset for higher generalization.
🔹 Integrating a real-time API for cloud-based classification.
