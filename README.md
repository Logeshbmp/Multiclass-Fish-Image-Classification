# Multiclass Fish Image Classification

## Project Overview
This project classifies fish images into multiple species categories using deep learning.
It includes:
- A CNN model trained from scratch.
- Multiple transfer learning models fine-tuned on the fish dataset.
- Model performance comparison.
- A Streamlit app for real-time fish species prediction from uploaded images.

## Domain
Image Classification

## Problem Statement
Classify fish images into multiple categories and compare model architectures to identify the best-performing model for deployment.

## Business Use Cases
- Enhanced Accuracy: Select the best model architecture based on evaluation metrics.
- Deployment Ready: Enable real-time predictions through a user-friendly web app.
- Model Comparison: Compare models using consistent performance metrics.

## Skills Demonstrated
- Deep Learning
- Python
- TensorFlow / Keras
- Streamlit
- Data Preprocessing and Augmentation
- Transfer Learning
- Model Evaluation
- Visualization
- Model Deployment

## Dataset
The dataset contains fish images organized by class folders.
Training/validation/testing pipelines use `ImageDataGenerator` and resized inputs (`224x224`).

## Approach
1. Data preprocessing and augmentation
- Rescale pixel values to `[0, 1]`.
- Apply rotation, zoom, and flip augmentations on training data.

2. Model training
- Train a custom CNN from scratch.
- Fine-tune pre-trained models:
  - VGG16
  - ResNet50
  - MobileNet
  - InceptionV3
  - EfficientNetB0
- Save trained models as `.h5` files.

3. Model evaluation
- Compare models using test accuracy (and can be extended to precision/recall/F1/confusion matrix).
- Visualize model comparison in the Streamlit app.

4. Deployment
- Streamlit app accepts uploaded image files.
- Predicts fish class and confidence score.
- Displays top predictions and model comparison chart.

## Repository Files
- `fish_app.py`: Streamlit inference app.
- `Multiclas_fish.ipynb`: Training, fine-tuning, and evaluation workflow.
- `class_labels.json`: Class label mapping.
- `model_comparison.csv`: Test accuracy comparison across models.
- `*_fish_model.h5`: Saved trained model checkpoints.
- `requirements.txt`: Python dependencies.

## Model Performance (from `model_comparison.csv`)

| Model | TestAccuracy |
|---|---:|
| MobileNet_FineTuned | 0.9984 |
| InceptionV3_FineTuned | 0.9972 |
| VGG16_FineTuned | 0.9925 |
| CNN | 0.9642 |
| ResNet50_FineTuned | 0.6727 |
| EfficientNetB0_FineTuned | 0.1632 |

## How to Run
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```bash
   streamlit run fish_app.py
   ```
4. Upload a fish image and select a model for prediction.

## Notes
- `fish_best_model.h5` is configured as the default model option in the app.
- Keep all model `.h5` files and `class_labels.json` in the project root for seamless loading.
- For production use, consider exporting to the latest `.keras` format and adding robust input validation/logging.
