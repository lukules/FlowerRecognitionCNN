# Flower Recognition using Machine Learning
### University Project for the Course: Systems with Machine Learning

# Project Overview

This project aims to develop a machine learning model capable of recognizing five types of flowers based on image input. The system classifies each image into one of the following categories:

    Tulips
    Roses
    Dandelions
    Sunflowers
    Daisies

The system is designed to be robust, accounting for variations in lighting, background clutter, and flower appearance. The model processes RGB images and outputs a classification label indicating the recognized type of flower.
Problem Definition
Goal

The objective is to build a model that can accurately classify flower images into one of five categories. This task falls under image classification.
Inputs

    RGB images in formats such as JPEG, PNG, and BMP.
    Minimum resolution: 400x400 pixels.

Outputs

    Classification label representing the recognized flower type.

# Data Description
Dataset

    Source: Data was collected from three different datasets on Kaggle.

    Total Images: 13,080 images across five categories:
        Daisy: 2,146 images (16.4%)
        Dandelion: 2,666 images (20.4%)
        Sunflower: 2,758 images (21.2%)
        Tulip: 2,739 images (21.1%)
        Rose: 2,771 images (20.9%)

    Images vary in size and quality, providing a diverse dataset. They were resized to 64x64, 128x128, and 256x256 pixels to facilitate model evaluation and comparison.

# Data Cleaning

    SHA256 hashes were used to remove duplicates and ensure unique images in each class.
    Some images containing multiple types of flowers or people in the background were filtered out to improve classification performance.

# Data Representation

    Images are stored as three-dimensional NumPy arrays (height, width, color channels).
    Each image is associated with a label indicating its class (Daisy, Dandelion, Sunflower, Tulip, Rose).

# Data Splits

The dataset was split into three configurations:

    SPLIT 1: Original class distribution without augmentation or normalization.
    SPLIT 2: Uniform class distribution with augmented and normalized data.
    SPLIT 3: Similar to SPLIT 2 but with validation data extracted from the training set.

# Machine Learning Approach
## Problem Type

This is a supervised classification problem. The goal is to train a model to classify images into one of five flower categories.
Chosen Model: Convolutional Neural Networks (CNNs)

CNNs were chosen due to their strong performance in image classification tasks. Their ability to automatically detect important features like edges, textures, and patterns makes them ideal for this project.
Loss Function

    Categorical Cross-Entropy: This function is optimal for multi-class classification problems, comparing the predicted probabilities to the actual class labels.

# Training and Testing
SPLIT 1

    The model was trained on the SPLIT 1 dataset.
    Training Time: ~43 seconds per epoch.
    Overfitting was attempted by removing regularization, batch normalization, and dropout. This allowed the model to memorize training data, which reduced generalization.

SPLIT 2

    Data augmentation techniques like rotation, inversion, and scaling were used to expand the dataset.
    Performance on the SPLIT 2 dataset showed improved generalization, with less fluctuation in validation metrics.

SPLIT 3

    Similar performance trends were observed with SPLIT 3, where validation and test accuracies remained consistent across different datasets.

# Testing Metrics

The following metrics were used to evaluate the models:

    Accuracy: Overall correctness of predictions.
    Precision: Correctness of positive predictions.
    Recall: Ability to capture all relevant instances.
    F1-Score: Harmonic mean of precision and recall, balancing both metrics.
    Confusion Matrix: Shows classification performance for each class.

# Results
MODEL 1

    Achieved around 84-87% accuracy, with strong performance on Sunflowers but confusion between Tulips and Roses.

MODEL 2

    Achieved 83-84% accuracy, though slightly lower than MODEL 1 due to normalization and augmentation.

MODEL 3

    Consistent performance across all splits, with high accuracy for Sunflowers.

This project was completed as part of the Systems with Machine Learning course, aiming to apply machine learning techniques to solve real-world classification problems. The project demonstrates the effectiveness of convolutional neural networks in image classification tasks, particularly when dealing with complex and varied data such as flower images.
