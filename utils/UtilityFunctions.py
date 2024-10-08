# utils for ML & DL algorithms using numpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# setting the seed for reproducibility
np.random.seed(42)

# normalize the data
def normalize(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_norm = (X - X_min) / (X_max - X_min)
    return X_norm

# standardize the data
def standardize(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    return (X - X_mean) / X_std + 1e-7

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# softmax function
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def train_test_split(X, y, test_size=0.2, shuffle=True):
    # Ensure that X is a DataFrame and y is a Series or DataFrame
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise ValueError("y must be a pandas Series or DataFrame")
    
    # Shuffle the data if required
    if shuffle:
        shuffled_indices = np.random.permutation(len(X))
        X = X.iloc[shuffled_indices].reset_index(drop=True)
        y = y.iloc[shuffled_indices].reset_index(drop=True)

    # Determine the split index
    split_index = int(len(X) * (1 - test_size))

    # Split the data into train and test sets
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, X_test, y_train, y_test

# TODO: Implement cross_val_score function for a numpy array
def cross_val_score(model, X, y, cv=5):
    """
    Evaluate a model using cross-validation.

    Parameters
    ----------
    model : object
        The model to evaluate.

    X : array-like of shape (n_samples, n_features)
        The input data.

    y : array-like of shape (n_samples,)
        The target values.

    cv : int, default=5
        The number of folds.

    Returns
    -------
    scores : array-like of shape (cv,)
        The score of the model for each
    """
    n = X.shape[0]  # Number of samples in the dataset
    indices = np.random.permutation(n)  # Shuffle the data randomly
    X = X[indices]  # Shuffle X based on random indices
    y = y[indices]  # Shuffle y based on random indices
    scores = np.zeros(cv)  # Initialize array to store scores for each fold

    for i in range(cv):
        split = int(n * (i / cv))  # Define the starting index for the test set
        # Train set is all data except for the current fold
        X_train = np.concatenate([X[:split], X[split + int(n / cv):]])
        X_test = X[split:split + int(n / cv)]  # Current fold is used as the test set
        y_train = np.concatenate([y[:split], y[split + int(n / cv):]])
        y_test = y[split:split + int(n / cv)]

        model.fit(X_train, y_train)  # Fit the model on the training data
        scores[i] = model.score(X_test, y_test)  # Evaluate model on the test data

    return scores  # Return the scores (e.g., accuracy) for each fold

# plot cross validation scores
def plot_cross_val_score(scores, k):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=np.arange(k), y=scores, palette='viridis')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Cross Validation Scores')
    plt.show()


def confusion_matrix(y_true, y_pred):
    '''
    Computes the confusion matrix to evaluate the accuracy of a classification.
    
    Parameters
    ----------
    y_true : array-like
        True labels of the data.
    y_pred : array-like
        Predicted labels.
    
    Returns
    -------
    matrix : numpy.ndarray
        Confusion matrix where each entry (i, j) is the count of samples with true label i and predicted label j.
    '''
    # Ensure y_true and y_pred have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("The length of y_true and y_pred must be the same.")
    
    # Get the number of unique classes
    n_classes = len(np.unique(y_true))
    
    # Initialize the confusion matrix with zeros
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    # Fill the confusion matrix
    for i in range(len(y_true)):
        matrix[int(y_true[i]), int(y_pred[i])] += 1

    return matrix


def plot_confusion_matrix(matrix, labels, normalize=False):
    '''
    Plots the confusion matrix using a heatmap.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        Confusion matrix to plot.
    labels : list
        List of label names.
    normalize : bool, optional
        If True, normalize the confusion matrix by row (true labels).
    '''
    if normalize:
        # Normalize the matrix by dividing each row by its sum
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    
    # Plot the heatmap
    sns.heatmap(matrix, annot=True, fmt='.2f' if normalize else 'g', cmap='Blues',
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Count' if not normalize else 'Proportion'})

    # Add labels and title
    plt.xlabel('Predicted labels', fontsize=12)
    plt.ylabel('True labels', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=45)  # Rotate x-tick labels for better readability
    plt.yticks(rotation=0)
    plt.show()


# accuracy score
def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# mean squared error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# mean absolute error
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# cross entropy error
def cross_entropy_error(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-7))

