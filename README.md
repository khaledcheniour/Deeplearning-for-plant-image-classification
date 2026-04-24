# Tree Image Segmentation using U-Net and Attention Mechanisms

## Overview
This project focuses on **tree image segmentation** using deep learning models based on **U-Net** architecture.  
The objective was to compare different U-Net variants with and without attention mechanisms to improve segmentation accuracy on natural tree images.

## Models Tested
Six models were implemented and evaluated:

- U-Net with **4 encoder layers**
  - Without Attention
  - With Self-Attention
  - With Multi-Head Attention

- U-Net with **5 encoder layers**
  - Without Attention
  - With Self-Attention
  - With Multi-Head Attention

## Technologies Used
- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib

## Dataset
Annotated tree images with pixel-level masks.

Dataset split:
- Training: 70%
- Validation: 15%
- Testing: 15%

## Preprocessing
- Image resizing
- Pixel normalization
- Data augmentation:
  - Rotation
  - Flip
  - Zoom

## Training Setup
- Loss Function: Dice Loss
- Optimizer: Adam
- Evaluation Metrics:
  - Accuracy
  - Recall
  - Precision
  - F1-Score
  - IoU

## Results Summary

### 4 Layers Encoder

| Model | Accuracy | F1-Score |
|------|----------|----------|
| Without Attention | 0.7200 | 0.7612 |
| Self-Attention | 0.8520 | 0.8559 |
| Multi-Head Attention | 0.8468 | 0.8512 |

### 5 Layers Encoder

| Model | Accuracy | F1-Score |
|------|----------|----------|
| Without Attention | 0.8504 | 0.8471 |
| Self-Attention | 0.8603 | 0.8621 |
| Multi-Head Attention | 0.8481 | 0.8492 |

## Key Findings
- Attention mechanisms significantly improved segmentation performance.
- **Self-Attention** achieved the best overall results.
- Deeper models (5 layers) performed better than shallow models.

## Applications
- Forest monitoring
- Smart agriculture
- Environmental analysis
- Vegetation mapping

## Authors
- Yessmine Chaabouni  
- Khaled Cheniour  
- Sahar Feki

## Supervisor
Prof. Abdelaziz Kallel

## Internship Organization
Centre de Recherche Numérique de Sfax  
Remote Sensing for Smart Agriculture Department

## Internship Period
July 1, 2025 – July 31, 2025
