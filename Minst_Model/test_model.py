#!/usr/bin/env python3
"""This script loads the pre-trained model and tests it using a small subset of the MNIST dataset."""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the MNIST dataset
print("Loading MNIST dataset...")
mnist = pd.read_csv('dataset.csv')

# Separate features and labels
data = mnist.drop("target", axis=1)  # Features (X)
target = mnist["target"]             # Labels (y)

# Load the pre-trained model
print("Loading pre-trained model...")
ovo_clf = joblib.load('minst_model.pkl')

# Take a small subset from the dataset (for example, 20 samples)
X_test_small = data[:20]  # First 20 samples
y_test_small = target[:20]

# Predict using the loaded model
print("Testing the model on a small subset of the MNIST dataset...")
y_pred_small = ovo_clf.predict(X_test_small)

# Calculate precision, recall, and F1-score for the small test set
precision = precision_score(y_test_small, y_pred_small, average='macro')
recall = recall_score(y_test_small, y_pred_small, average='macro')
f1 = f1_score(y_test_small, y_pred_small, average='macro')

# Display results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("Testing completed.")

