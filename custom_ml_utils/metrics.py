import numpy as np

# Accuracy score
def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Mean squared error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Mean absolute error
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Cross entropy error
def cross_entropy_error(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-7))

# Confusion matrix
def confusion_matrix(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        matrix[int(y_true[i]), int(y_pred[i])] += 1
    return matrix