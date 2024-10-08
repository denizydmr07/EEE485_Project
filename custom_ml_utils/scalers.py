import numpy as np

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.min_ = None
        self.max_ = None
        self.feature_range = feature_range  # Set the feature range (default is (0, 1))

    # Fit the scaler to the columns based on their min and max values
    def fit(self, X, columns):
        # Ensure the selected columns are treated as float for correct division
        self.min_ = np.min(X[:, columns].astype(float), axis=0)
        self.max_ = np.max(X[:, columns].astype(float), axis=0)
        return self

    # Transform the specified columns using the fitted min and max and scale to feature_range
    def transform(self, X, columns):
        X_scaled = X.astype(float).copy()  # Ensure that operations are done in float
        X_min, X_max = self.feature_range
        X_scaled[:, columns] = ((X[:, columns].astype(float) - self.min_) / 
                                (self.max_ - self.min_)) * (X_max - X_min) + X_min
        return X_scaled

    # Fit and transform in one step
    def fit_transform(self, X, columns):
        self.fit(X, columns)
        return self.transform(X, columns)

    # Inverse transform the scaled data back to the original values
    def inverse_transform(self, X_scaled, columns):
        X_min, X_max = self.feature_range
        X_original = X_scaled.astype(float).copy()
        X_original[:, columns] = ((X_scaled[:, columns] - X_min) / 
                                  (X_max - X_min)) * (self.max_ - self.min_) + self.min_
        return X_original


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    # Fit the scaler to the columns based on their mean and standard deviation
    def fit(self, X, columns):
        # Ensure the selected columns are treated as float for correct calculations
        self.mean_ = np.mean(X[:, columns].astype(float), axis=0)
        self.std_ = np.std(X[:, columns].astype(float), axis=0)
        return self

    # Transform the specified columns using the fitted mean and std
    def transform(self, X, columns):
        X_scaled = X.astype(float).copy()  # Ensure that operations are done in float
        X_scaled[:, columns] = (X[:, columns].astype(float) - self.mean_) / self.std_
        return X_scaled

    # Fit and transform in one step
    def fit_transform(self, X, columns):
        self.fit(X, columns)
        return self.transform(X, columns)

    # Inverse transform the scaled data back to the original values
    def inverse_transform(self, X_scaled, columns):
        X_original = X_scaled.astype(float).copy()
        X_original[:, columns] = X_scaled[:, columns] * self.std_ + self.mean_
        return X_original

