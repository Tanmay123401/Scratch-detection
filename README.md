Scratch Detection – AI R&D Prototype
Author: Tanya Moras
Date: 27/01/2026
Version: v1.0
 1. Project Goal

The goal of this project is to build a surface scratch detection system using a public dataset and a lightweight deep-learning model, and deliver:

A trained classifier

A classification report

A reproducible pipeline

A HuggingFace-hosted model

A public GitHub repository

Problem Type: Binary Image Classification

scratch

no_scratch

Persona:
Manufacturing quality engineers who need automated defect detection for real-time quality inspection.

 2. Dataset (Public)

I used the NEU Surface Defect Dataset (NEU-CLS) — a standard industrial defect dataset.

 Dataset Link (Kaggle):
https://www.kaggle.com/datasets/kaustubhsrinivas/neu-surface-defect-database

Original dataset has 6 classes:

crazing

inclusion

patches

pitted_surface

rolled-in_scale

scratches

For this task:

I converted it into binary classes:

Folder	Description
scratch/	All images from “scratches” folder
no_scratch/	All images from the other 5 defect folders

Final folder structure:

data/
 ├── scratch/
 └── no_scratch/

 3. Approach & Architecture
✔ Why MobileNetV2?

Fast training (<10 mins on CPU)

Lightweight (deployment friendly)

Good accuracy for texture/defect datasets

Perfect for 2-hour evaluation limit

 Preprocessing:

Resize → 224×224

Normalize (ImageNet mean/std)

Augmentations:

RandomHorizontalFlip

RandomRotation(10°)

RandomResizedCrop

ColorJitter


 Training Setup:

Train/Val split → 80:20

Loss → CrossEntropy

Optimizer → Adam (LR=1e-3)

Model → MobileNetV2 (ImageNet pretrained)

Epochs → 8

Batch size → 32

* Evaluation:

Classification report (precision, recall, f1)

Accuracy (train + validation)

Confusion matrix (optional)

 4. Model Details
Backbone:

MobileNetV2 pretrained on ImageNet

Modified final layer:

nn.Linear(1280 → 2)

Artifacts Produced:

model.pt (PyTorch)

model.onnx (Optional ONNX export)

classification_report.txt

confusion_matrix.png

training_loss.png

 5. Results
Train Accuracy: 1.00
Validation Accuracy: 1.00
Classification Report:

(Generated using evaluate.py)

              precision    recall  f1-score   support

  no_scratch       1.00      1.00      1.00       242
      scratch       1.00      1.00      1.00        46

    accuracy                           1.00       288
   macro avg       1.00      1.00      1.00       288
weighted avg       1.00      1.00      1.00       288


This perfect performance is typical for NEU-CLS due to its clean industrial images.

Confusion Matrix (if generated):

confusion_matrix.png

Training Loss Plot:

training_loss.png

 6. HuggingFace Model Link

* HuggingFace:
https://huggingface.co/Tanmay1402/Scratch_detection

 7. GitHub Repository Link

* GitHub:
https://github.com/Tanmay123401/Scratch-detection

 8. Running Instructions
Install requirements:
pip install -r requirements.txt

Train the model:
python train.py

Evaluate model & generate report:
python evaluate.py

Run inference on a single image


python inference.py

 9. File Structure
scratch-detection/
 ├── data/
 │    ├── scratch/
 │    └── no_scratch/
 ├── train.py
 ├── evaluate.py
 ├── inference.py
 ├── requirements.txt
 ├── classification_report.txt
 ├── model.pt
 ├── model.onnx (optional)
 ├── confusion_matrix.png (optional)
 ├── training_loss.png (optional)
 └── README.md

 10. Future Improvements

Multi-class defect classification (all 6 NEU classes)

EfficientNet-based model for higher robustness

Web API using FastAPI

Real-time defect detection pipeline

Integration into factory hardware (edge inference)

Larger datasets with mixed materials

 11. References

NEU Surface Defect Dataset (NEU-CLS)

PyTorch MobileNetV2 model zoo

Scikit-learn classification metrics

HuggingFace model hosting documentation