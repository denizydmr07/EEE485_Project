import numpy as np

class PCA():
    ''' 
    A class to perform PCA on a general dataset.

    Attributes
    ----------
    fitted : bool
        A flag indicating whether the PCA object has been fitted.
    mean : numpy.ndarray
        The mean of the data.
    eigenvalues : numpy.ndarray
        The eigenvalues of the covariance matrix of the data.
    eigenvectors : numpy.ndarray
        The eigenvectors of the covariance matrix of the data.
    pve : numpy.ndarray
        The proportion of variance explained by each principal component.
    '''
    def __init__(self):
        self.fitted = False
        self.mean = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.pve = None

    def fit(self, X):
        '''
        Fits the PCA object to the data.

        Parameters
        ----------
        X : numpy.ndarray
            The dataset to fit the PCA object to. 
        '''
        # I don't want to alter the original dataset, so I make a copy
        X = X.copy()

        # getting the mean of the data
        self.mean = X.mean(axis=0)

        # subtracting the mean from the data
        X = X - self.mean

        # calculating the covariance matrix
        cov_mat = X.T @ X

        # calculating the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

        # calculating the proportion of variance explained
        pve = eigenvalues / np.sum(eigenvalues)

        # sorting the eigenvalues and eigenvectors
        idx = pve.argsort()[::-1]

        # storing the sorted eigenvalues, eigenvectors, and pve
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        self.pve = pve[idx]

        # deleting reference of X to free up memory
        del X

        # setting the fitted flag to True
        self.fitted = True

    def transform(self, data, n_components):
        '''
        Transforms the data into the principal component space.

        Parameters
        ----------
        data : numpy.ndarray
            The data to transform.
        n_components : int
            The number of principal components to use.

        Returns
        -------
        numpy.ndarray
            The transformed data.
        '''
        # check if self.fitted
        if not self.fitted:
            raise Exception('PCA object has not been fitted.')
        
        # Center the data (subtract the mean)
        data_centered = data - self.mean

        # Project data into the reduced principal component space
        projection = data_centered @ self.eigenvectors[: ,:n_components]

        return projection
    

    def inverse_transform(self, data_transformed, n_components):
        '''
        Reconstructs the original data from the transformed data.

        Parameters
        ----------
        data_transformed : numpy.ndarray
            The data in the principal component space.
        n_components : int
            The number of principal components used.

        Returns
        -------
        numpy.ndarray
            The reconstructed data in the original space.
        '''
        # Reconstruct the data from principal components
        reconstructed_data = data_transformed @ self.eigenvectors[:, :n_components].T
        
        # Add the mean back to get the uncentered data
        reconstructed_data = reconstructed_data + self.mean

        return reconstructed_data
    
    def explained_variance(self, n_components):
        '''
        Returns the proportion of variance explained by the first n_components.

        Parameters
        ----------
        n_components : int
            The number of principal components to consider.

        Returns
        -------
        float
            The proportion of variance explained.
        '''
        return np.sum(self.pve[:n_components])
    
    def plot_explained_variance(self):
        '''
        Plots the proportion of variance explained by each principal component as a bar graph using Seaborn.
        '''
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Convert the explained variance to percentage
        explained_variance_percentage = self.pve * 100

        # Set the seaborn style
        sns.set(style="whitegrid")

        # Create a bar plot using seaborn
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(range(1, len(explained_variance_percentage) + 1)), y=explained_variance_percentage, palette="muted")

        # Add labels and title
        plt.xlabel('Principal Component', fontsize=12)
        plt.ylabel('Proportion of Variance Explained (%)', fontsize=12)
        plt.title('Variance Explained by Each Principal Component', fontsize=14)
        
        # Ensure the y-axis goes from 0 to 100%
        plt.ylim(0, 100)

        plt.show()


