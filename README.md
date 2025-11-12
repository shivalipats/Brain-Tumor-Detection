# Brain-Tumor-Detection

## Brain Tumor MRI Classification Using Deep Learning
### Overview

This project focuses on detecting and classifying brain tumors from MRI images using deep learning.
The model uses transfer learning with ResNet18, fine-tuned on a medical imaging dataset to distinguish between tumor types such as glioma, meningioma, pituitary, and no tumor.

The goal is to assist early diagnosis and support medical professionals through a fast, reliable, and interpretable AI system.

### Problem
Brain tumors are life-threatening and require early, accurate detection for effective treatment.
However, MRI scans can be complex to analyze manually and are subject to human error.
This project addresses that challenge by building an AI-based diagnostic model that can classify tumors with high accuracy.

### Dataset 
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
Classes:
- Glioma
- Meningioma
- Pituitary
- No Tumor

Data Split:

80% training

20% validation/testing

## Training and Evaluation
Training Details:

Optimizer: Adam

Learning Rate: 0.001

Batch Size: 32

Epochs: 25

Loss Function: Cross-Entropy

Evaluation Metrics:

Confusion Matrix

