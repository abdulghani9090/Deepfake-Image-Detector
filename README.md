# 🛡️ Deepfake Forensics Lab

**Enterprise-Grade Deepfake Detection System**

A professional forensic tool designed to detect AI-generated facial manipulations in images. Powered by Deep Learning (ResNet18) and explainable AI (Grad-CAM), wrapped in a high-fidelity "Cyber-Forensics" dashboard.

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)

## 🌟 Key Features

### 🖥️ Professional Frontend
- **Forensic Dashboard**: A modular, bento-grid layout for analyzing evidence.
- **Cyber-Forensics Theme**: Immersive dark mode with neon accents (`#0f172a` base), glassmorphism, and responsive micro-interactions.
- **Session History**: Tracks your recent analysis sessions and results in the sidebar.
- **UX Pipeline Simulation**: Visualizes the 4-stage analysis process (Detection -> Preprocessing -> TTA Inference -> Forensics).

### 🧠 Robust Backend
- **Multi-View Inference (TTA)**: Uses **Test Time Augmentation** to analyze image variations (original + flipped), reducing false positives.
- **Smart Explainability**: Enhanced **Grad-CAM** heatmaps with noise reduction and thresholding to pinpoint specific artifact regions.
- **Resilient Pipeline**: Robust Face detection (MTCNN) with error handling and smart padding.

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (Recommended for speed, but runs on CPU)

### Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/deepfake-detector.git
   cd deepfake-detector
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard**
   ```bash
   streamlit run app.py
   ```

---

## 🛠️ Technical Architecture

### 1. Face Detection (MTCNN)
The system uses Multi-task Cascaded Convolutional Networks (MTCNN) to isolate faces from the input image.
- **Enhancement**: Added 20% context padding to capture chin/forehead artifacts often missed by tight crops.

### 2. Deepfake Classification (ResNet18)
- **Backbone**: ResNet18 (Pretrained on ImageNet).
- **Fine-tuning**: The final fully connected layer is fine-tuned for binary classification (Real vs. Fake).
- **Inference Strategy**: 
    - `Input -> Original -> Score_A`
    - `Input -> Flip -> Score_B`
    - `Final Score = (Score_A + Score_B) / 2`

### 3. Explainability (Grad-CAM)
Visualizes *why* the model made a decision.
- **Layer**: Targeting the final convolutional layer (`layer4`).
- **Post-Processing**: Gaussian Blur + Thresholding (<0.2 cut) to remove background noise.

---

## 📂 Project Structure

```
deepfake_detector/
├── app.py                 # Main Streamlit Application (The "Dashboard")
├── scripts/
│   ├── model.py           # PyTorch Model Definition (ResNet18)
│   └── utils.py           # Core Logic: Face Detection, Grad-CAM, TTA
├── models/
│   └── deepfake_model.pth # Trained Model Weights
├── requirements.txt       # Python Dependencies
└── README.md              # Documentation
```

## 🛡️ Disclaimer
This tool is for educational and research purposes. While high-performing, no detector is 100% accurate. Always verify results with manual inspection.
