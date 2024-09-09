Here's an updated version of the `README.md` file that includes the instructions for testing the model with a pre-loaded script.

```markdown
# Handwriting Digit Recognition using MNIST Dataset

This project uses the MNIST dataset to train a machine learning model that predicts handwritten digits (0-9). The model is trained using a `KNeighborsClassifier` wrapped in a `OneVsOneClassifier`. After training, the model is saved as a file (`minst_model.pkl`) for easy loading and testing.

## Project Files

- **train_model.py**: Script that trains the model using the MNIST dataset and saves the trained model as `minst_model.pkl`.
- **test_model.py**: Script to test the saved model on a small subset of the MNIST dataset.
- **mnist_sample.csv**: A sample of the MNIST dataset used for both training and testing.
- **minst_model.pkl**: Pre-trained model that can predict handwritten digits.

## How to Use

### 1. Install the Required Libraries

Make sure you have Python installed on your system. Then, install the necessary dependencies using `pip`:

```bash
pip install pandas scikit-learn joblib matplotlib numpy pillow
```

### 2. Training the Model

If you want to train the model from scratch, run the `train_model.py` script:

```bash
python fetch_minst_dataset.py
python train_model.py
```

This will train the model using the MNIST dataset and save it as `minst_model.pkl` in the same directory.

### 3. Testing the Pre-trained Model

To test the pre-trained model, follow these steps:

1. Make sure the files `mnist_sample.csv` and `minst_model.pkl` are in the same directory.
2. Run the `test_model.py` script:

```bash
python test_model.py
```

### Testing Details

- The `test_model.py` script loads the pre-trained model (`minst_model.pkl`) and tests it on a small subset of the MNIST dataset (20 samples).
- It calculates the precision, recall, and F1-score of the predictions.

The output will show:

- **Precision**: The fraction of relevant instances among the retrieved instances.
- **Recall**: The fraction of relevant instances that were successfully retrieved.
- **F1 Score**: The weighted average of precision and recall.

**Note**: The model can only be tested using the provided MNIST dataset (`mnist_sample.csv`) and not with custom images.

### Example Output

When you run the `test_model.py` script, you should see something like this:

```bash
Loading MNIST dataset...
Loading pre-trained model...
Testing the model on a small subset of the MNIST dataset...
Precision: 0.8800
Recall: 0.88000
F1 Score: 0.88
Testing completed.
```

This shows that the model has been successfully tested and the metrics provide insight into its performance.

## Model and Dataset

- **Model**: KNeighborsClassifier (wrapped in OneVsOneClassifier)
- **Dataset**: A sample of the MNIST dataset (`mnist_sample.csv`), used for both training and testing.

## License

This project is open-source and available for use under the [MIT License](LICENSE).
```

This `README.md` file explains the project structure, how to install dependencies, and how to test the model using the provided script. It also clarifies that the model is meant to be tested with the provided MNIST dataset only.
