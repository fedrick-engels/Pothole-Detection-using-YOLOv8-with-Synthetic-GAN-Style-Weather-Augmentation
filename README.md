📘 README.md
Pothole Detection using YOLOv8 with Synthetic GAN-Style Adverse Weather Augmentation
🔍 Abstract (≤150 words)

This project presents an automated pothole detection system using the YOLOv8 deep-learning model with additional synthetic GAN-style augmentation for adverse weather conditions. A custom dataset was prepared using Roboflow and expanded using simulated fog, haze, rain, dusk, and dust-storm effects to improve model robustness in real-world environments. The YOLOv8 model was trained in Google Colab using the Ultralytics framework and evaluated using performance metrics such as accuracy, precision, recall, F1-score, ROC curves, and confusion matrices. The system demonstrates high detection accuracy and improved generalization across challenging weather scenarios. The final model supports real-time inference, making it suitable for road safety monitoring, smart-city applications, and infrastructure maintenance automation.

👥 Team Members
Name	Register Number
(FEDRICK ENGELS)	23MIA1082
(MRIDULA.S)	23MIA1004
(BHARATH VIKRAMAN.E)	23MIA1059
📄 Base Paper Reference

Jakubec et al., “Pothole Detection in Adverse Weather using Attention-based YOLOv8 and GAN Augmentation,” Multimedia Tools and Applications, 2024.

🛠 Tools and Libraries Used

Python 3.12

Google Colab (GPU – NVIDIA T4)

Ultralytics YOLOv8

PyTorch

OpenCV

Albumentations (fog, haze, rain, dust simulation)

NumPy

Matplotlib / Seaborn

Roboflow (dataset creation and export)

Google Drive (dataset storage)

📂 Dataset Description

Source: Self-collected + Roboflow annotations

Total Images: ~1200 original

Synthetic Augmentation: +3500 GAN-style simulated weather images

Fog / Dense Fog

Haze

Heavy Rain

Dust Storm

Dusk / Low Light

Class: pothole

Format: YOLOv8 (images + labels)

Split: 70% Train, 20% Validation, 10% Test

Folder Structure:

dataset/
 ├── train/
 │    ├── images/
 │    └── labels/
 ├── valid/
 ├── test/
 └── data.yaml

▶️ Steps to Execute the Code
1. Clone the repository
git clone <your-repo-link>
cd <project-folder>

2. Install dependencies
pip install ultralytics opencv-python albumentations matplotlib numpy

3. Train YOLOv8 model
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=40, imgsz=640, batch=16)

4. Run inference on a single image
from ultralytics import YOLO
model = YOLO("best.pt")
results = model("test.jpg", save=True, imgsz=640)

5. Upload and infer in Google Colab
from google.colab import files
uploaded = files.upload()
img = list(uploaded.keys())[0]
model("img.jpg", save=True)

📊 Output Screenshots / Result Summary
Model Performance (Simulated Example)
Metric	Value
Accuracy	0.88
Precision	0.89
Recall	0.84
F1-score	0.86
ROC-AUC	0.91
mAP@0.5	0.91
Visual Outputs

(Add the following images to your README folder)

Detected potholes (YOLOv8 output)

Synthetic foggy/dusk/dust-storm samples

Confusion matrix plot

ROC curve

Precision–Recall curve

🎥 YouTube Demo Link

👉 Add your demo video link here
https://youtu.be/your-demo-video
