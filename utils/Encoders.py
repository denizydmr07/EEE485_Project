import numpy as np
import pandas as pd

class LabelEncoder:
    def __init__(self):
        self.classes_ = {}
        self.inverse_classes_ = {}

    # Fit the encoder with the unique labels
    def fit(self, y):
        unique_labels = np.unique(y)
        self.classes_ = {label: idx for idx, label in enumerate(unique_labels)}
        self.inverse_classes_ = {idx: label for idx, label in enumerate(unique_labels)}
        return self

    # Transform categorical labels into integers
    def transform(self, y):
        return np.array([self.classes_[label] for label in y])

    # Inverse transform integers back into the original labels
    def inverse_transform(self, y):
        return np.array([self.inverse_classes_[idx] for idx in y])

    # Fit and transform in one step
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

class OneHotEncoder:
    def __init__(self):
        self.label_encoder = LabelEncoder()  # Use the LabelEncoder you wrote
        self.n_classes = None  # To store the number of unique classes
        self.categories_ = []  # To store the categories for column names
    
    # Fit the encoder to the unique labels and prepare for one-hot encoding
    def fit(self, y, column_prefix="column"):
        # set the column prefix
        self.column_prefix = column_prefix
        # First, apply label encoding
        y_encoded = self.label_encoder.fit_transform(y)
        self.n_classes = len(np.unique(y_encoded))  # Get the number of unique classes
        self.categories_ = self.label_encoder.classes_  # Store the original category names
        return self

    # Transform the labels into one-hot encoded vectors and return a DataFrame
    def transform(self, y):
        # Convert y to encoded labels using the label encoder
        y_encoded = self.label_encoder.transform(y)
        
        # Create one-hot encoded matrix using np.eye
        one_hot_encoded = np.eye(self.n_classes)[y_encoded]
        
        # Create column names for the one-hot encoded DataFrame
        column_names = [f"{self.column_prefix}_{category}" for category in self.categories_]
        
        # Return as a DataFrame
        return pd.DataFrame(one_hot_encoded, columns=column_names)

    # Fit and transform in one step
    def fit_transform(self, y, column_prefix="column"):
        self.fit(y, column_prefix)
        return self.transform(y)
    
    # Inverse transform one-hot vectors back to the original labels
    def inverse_transform(self, one_hot_encoded):
        # Convert one-hot encoded vectors back to label encoded integers
        y_encoded = np.argmax(one_hot_encoded, axis=1)
        
        # Inverse transform to get the original labels then return as a DataFrame
        return pd.DataFrame(self.label_encoder.inverse_transform(y_encoded), columns=[self.column_prefix])

class OrdinalEncoder:
    def __init__(self):
        self.classes_ = {}  # To store the ordinal mapping
        self.inverse_classes_ = {}  # To store the reverse mapping
        self.column_prefix = None  # To store the column prefix
    
    # Fit the encoder to the unique labels in a column of the DataFrame
    def fit(self, df, column_prefix = "column", categories=None):
        # Set the column prefix
        self.column_prefix = column_prefix

        # set the ordered categories
        self.ordered_categories = categories

        # Extract the column from the DataFrame
        y = df.values
        
        # If predefined categories are provided, use them to assign ordinal labels
        if self.ordered_categories:
            unique_labels = self.ordered_categories
        else:
            unique_labels = np.unique(y)
        
        # Create mapping from category to integer
        self.classes_ = {label: idx for idx, label in enumerate(unique_labels)}
        self.inverse_classes_ = {idx: label for idx, label in enumerate(unique_labels)}
        return self
    
    # Transform the column of the DataFrame and return as a pandas Series
    def transform(self, s):
        # Get the column data, df is a Series
        y = s.values
        
        # Transform using the stored mapping
        encoded_values = np.array([self.classes_[label] for label in y])
        
        # Return the encoded values as a pandas Series
        return pd.Series(encoded_values, name=f"{self.column_prefix}_encoded")
    
    # Fit and transform in one step
    def fit_transform(self, df, column_prefix, categories=None):
        self.fit(df, column_prefix, categories)
        return self.transform(df)
    
    # Inverse transform the ordinal values back to the original labels
    def inverse_transform(self, encoded_series):
        # Convert the encoded values back to the original labels
        original_values = np.array([self.inverse_classes_[idx] for idx in encoded_series])
        return pd.Series(original_values, name=encoded_series.name.replace('_encoded', ''))


