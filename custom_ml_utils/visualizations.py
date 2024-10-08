import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Plot cross-validation scores
def plot_cross_val_score(scores, k):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=np.arange(k), y=scores, palette='viridis')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Cross Validation Scores')
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(matrix, labels, normalize=False):
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='.2f' if normalize else 'g', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()
