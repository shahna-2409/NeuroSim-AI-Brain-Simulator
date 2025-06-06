
# NeuroSim: AI-Enhanced Simulation of Human Brain Activity

This repository contains the full source code, assets, and web application files for the academic machine learning project titled **"NeuroSim – AI-Enhanced Simulation of Human Brain Activity"** submitted by Shahna Shahul Hameed as part of the MSc in Artificial Intelligence at the National College of Ireland.

---

## 🧠 Project Overview

**NeuroSim** is a deep learning-based medical diagnostic tool designed to predict and classify neurological conditions like **Alzheimer’s** and **Parkinson’s Disease** using multi-modal data including MRI brain scans and patient-generated drawing patterns. The project applies a **VGG19 Convolutional Neural Network** using transfer learning to achieve classification across multiple stages of both diseases.

---

## 🎯 Objectives

- Predict the **stage** of Alzheimer's and Parkinson's diseases using medical imaging.
- Use **transfer learning** with VGG19 to reduce training time and improve accuracy.
- Build a functional **Flask-based web interface** to allow real-time image upload and disease prediction.
- Explore data preprocessing techniques, augmentation, and overfitting management in CNNs.

---

## 🗂️ Project Structure

```
AI_Brain_Simulation/
├── app.py                         # Flask web app launcher
├── predict.py                     # Prediction logic using AI_Simulation.h5
├── AI_Simulation.h5               # Pre-trained CNN model
├── AccVal_acc.png                 # Training accuracy curve
├── LossVal_loss.png               # Training loss curve
│
├── templates/                     # HTML interface for Flask
│   ├── index.html
│   └── login.html
│
├── static/                        # CSS, JS, assets
│   └── assets/
│       ├── vendor/
│       └── images/
│
├── uploads/                       # Uploaded test images
└── attachments/                   # Sample input image sets
```

---

## 🧬 Datasets Used

| Dataset           | Source                      | Type            | Size         |
|------------------|-----------------------------|------------------|--------------|
| Alzheimer's MRI  | OASIS-3 (open access)       | 80,000+ MRI slices | 461 subjects |
| Parkinson's Drawings | Kaggle spiral & wave test images | 2,000+ images | 100+ patients |

Each image was resized to **224x224**, normalized, and augmented to improve generalizability.

---

## 🏗️ Model Architecture

- **Base Model:** VGG19 (pre-trained on ImageNet)
- **Transfer Learning:** All convolutional layers frozen
- **Top Layers:**
  - Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(8, Softmax)
- **Optimizer:** Adam (`lr=0.001`)
- **Loss Function:** Categorical Crossentropy
- **Epochs:** 10
- **Batch Size:** 32
- **Tuning Tool:** Keras Tuner (for hyperparameter optimization)

---

## 📊 Model Performance

| Metric               | Training Set | Validation Set |
|----------------------|--------------|----------------|
| Accuracy (Final)     | **94.48%**   | **39.43%**     |
| Loss (Final)         | **0.1787**   | **6.3181**     |

> ⚠️ Note: The model shows signs of **overfitting**, with high training accuracy but low generalization on the validation set. Future work should explore regularization and dataset balancing.

---


## 🌐 Web Application

The app allows a user to upload an image (e.g., MRI slice or drawing) and receive a predicted classification using the trained CNN.

### How to Run:

1. Clone the repository
2. Place the trained model `AI_Simulation.h5` in the root directory  
3. Install requirements:
```bash
pip install flask keras tensorflow numpy opencv-python
```
4. Launch the app:
```bash
python app.py
```
5. Visit `http://127.0.0.1:5000/` in your browser


---

## ⚠️ Limitations

- Validation accuracy and loss show that **generalization is limited**
- Dataset size for Parkinson's was relatively small
- No GPU was available; training was conducted on CPU
- Web app is **functional** but basic — future improvement could include confidence scores, user accounts, or DICOM support

---

## 🔮 Future Improvements

- Improve generalization via stronger augmentation or regularization (L2, Dropout)
- Add explainability tools like Grad-CAM or SHAP
- Include multimodal data (e.g., EEG, PET) for broader diagnostics
- Expand to additional neurological diseases

---

## 📄 Project Report

You can view the full detailed report here:  
📘 [View/Download Final report (PDF)](FINAL-REPORT.pdf)

---

## 👩‍💻 Author

**Shahna Shahul Hameed**  
MSc in Artificial Intelligence  
Dublin, Ireland 

📧 shahna.s.hameed24@gmail.com

🖇️ https://www.linkedin.com/in/shahna-shahul-hameed/

---

## 📄 License

This project is licensed under the MIT License.
