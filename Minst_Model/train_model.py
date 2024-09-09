#!/usr/bin/env python3
"""This module has a model that predict handwriting digit from the minst dataset"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import SGDClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from PIL import Image, ImageDraw, ImageFont
import random
import joblib


def preprocess_image_to_df(image_path):
    """Function to convert image to dataframe that will be compatible with minst dataset"""
    # Load the image and convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # Invert the image colors: black becomes white, and white becomes black
    img = Image.eval(img, lambda x: 255 - x)
    
    # Resize to 28x28 pixels
    img = img.resize((28, 28))
    
    # Convert the image to a numpy array and normalize pixel values
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize to match the training data
    
    # Flatten the 28x28 image into a single row (1, 784) for DataFrame compatibility
    img_flattened = img_array.flatten()
    
    # Convert the flattened image to a pandas DataFrame
    img_df = pd.DataFrame([img_flattened], columns=COLUMNS)
    
    return img_df

# A function to plot a digit
def plot_digits(X_new, images_per_row):
    """Function to plot digits from the minst dataset"""
    X_new = np.array(X_new) # np.c_[[data]][0]
    for index in range(len(X_new)):
        some_digit = X_new[index]
        some_digit_image = some_digit.reshape(28, 28)
        plt.subplot(images_per_row, images_per_row, index + 1)
        plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
        plt.axis("off")
    plt.show()

print("BEGINNING........")
# import the dataset
minst = pd.read_csv('dataset.csv')

# drop the label
data = minst.drop("target", axis=1) # X
target = minst["target"] # y(label)

# get the X and y(label)
X_ = np.array(data) # np.c_[[data]][0]
y = np.array(minst["target"])

# Split the dataset into training and testing sets
print("Splitting the dataset into training and testing..............")
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

COLUMNS = X_train.columns

# Train a KNeighbor Classifier model with one versus one
ovo_clf = OneVsOneClassifier(KNeighborsClassifier())
print("Training the model.......................")
ovo_clf.fit(X_train, y_train)
joblib.dump(ovo_clf, "minst_model.pkl")

# Evaluate ovo_clf model
print("Testing the model......................")
# Check the score
ovo_scores = cross_val_score(ovo_clf, X_test, y_test, cv=3, scoring='accuracy')
print("Cross validation scores from 3 trained models: ", ovo_scores)

# Predict using cross validation
ovo_predictions = cross_val_predict(ovo_clf, X_test, y_test, cv=3)

# Check the precision, recall, f1_score
ovo_precision = precision_score(y_test, ovo_predictions, average='macro')
ovo_recall = recall_score(y_test, ovo_predictions, average='macro')
f1_score_res = f1_score(y_test, ovo_predictions, average='macro')

print("recall score -- ", ovo_recall)
print("precision score -- ", ovo_precision)
print("F1 Score --", f1_score_res)
print("END........")
