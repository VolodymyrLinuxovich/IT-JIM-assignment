# Artifact Detection for Generated Images

## Overview

This project tackles the problem of detecting artifacts in generated images as part of an automatic quality assurance system. The goal is to build a binary image classification model that distinguishes between images with artifacts (e.g., text overlays, distorted facial features, misplaced hands/fingers, etc.) and those without. The dataset comprises training and test images, each following the naming format: `image_<frame_index>_<class label>.png`, where the class label is either 0 (artifact) or 1 (artifactless).

## Description of the Task

- **Objective:** Develop and train a binary image classification model to identify artifacts in generated images.
- **Evaluation Metric:** Micro F1 score (required). Additional metrics are optional.
- **Approach:** The project includes functions for training, validation, and inference. Advanced users can develop and combine different approaches for an ensemble solution.

# Project Structure

project_root/
├── data/
│   ├── train/
│   └── test/
├── main.py
├── requirements.txt
└── README.md

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- Pillow (PIL)
- Pandas
- NumPy
- Scikit-learn
- tqdm

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd project_root

2. Install Dependencies: Install all required packages using pip:
# 
# pip install -r requirements.txt
# 
3. Directory Setup: Ensure that the project directory is organized as described above and that the input images are placed in the respective train and test folders under the data directory.
# 
# Process Flow
# Input Loading:
# The dataset is loaded using a custom Dataset class (ArtifactDataset), which reads image files from the specified directory and extracts the class label from the file name.
# 
# Data Transformation:
# Different transformation pipelines are used for training (with augmentations), validation, and testing. Transformations include resizing, random horizontal flipping, rotation, and normalization (using ImageNet statistics).
# 
# Model Architecture & Initialization:
# Transfer learning is applied using one or more pretrained models (such as ResNet18 and EfficientNet-B0). The final classification layer is modified to output the required two classes.
# 
# Training & Validation:
# The model is trained using a cross-entropy loss function and the Adam optimizer. After each epoch, the model is validated on the validation set, and the micro F1 score is computed.
# 
# Inference:
# Inference functions are provided to generate predictions on the test dataset. An ensemble method is optionally implemented by averaging the probability outputs from multiple models.
# 
# Metric Calculation:
# The primary evaluation metric is the micro F1 score, which is computed using Scikit-learn's f1_score function.
# 
# Code Modules
# Global Dataset Loading (dataset.py):
# - ArtifactDataset: A custom PyTorch Dataset that loads images and extracts labels from file names.
# 
# Data Transformations (within dataset.py or separate module):
# - get_train_transforms: Returns a composition of transformations for training images.
# - get_valid_transforms: Returns a standardized transformation pipeline for validation images.
# - get_test_transforms: Returns transformations for test images (same as validation).
# 
# Model Initialization (model.py):
# - get_resnet18_model: Initializes a ResNet18 model with a modified final layer.
# - get_efficientnet_b0_model: Initializes an EfficientNet-B0 model with the classifier layer adjusted for two classes.
# 
# Training & Validation (train.py):
# - train_one_epoch: Trains the model for one epoch and computes training loss and micro F1 score.
# - validate_model: Evaluates the model on the validation dataset.
# 
# Inference (inference.py):
# - inference: Generates predictions on test images.
# - inference_ensemble: Combines outputs from multiple models by averaging probabilities.
# 
# Main Execution (main.py):
# Coordinates the overall pipeline: dataset loading, model initialization, training, validation, inference, and CSV submission generation.
# 
# Usage
# Training and Inference:
# Run the main script to execute the entire pipeline:
# 
# python src/main.py --mode train --train_dir data/train --test_dir data/test --epochs 10 --batch_size 32
# (Adjust the command-line arguments as per your configuration.)
# 
# Ensembling:
# The ensemble function in inference.py aggregates predictions from different models, which can further boost overall performance.
# 
# Output:
# The final submission is saved as submission.csv, which contains the filename and predicted label (0 for artifact, 1 for artifactless).