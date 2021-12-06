import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean
        self.mean = np.mean(X, axis=0)
        X -= self.mean
        # covariance matrix
        # data: column=feature, row=sample НО! cov в np использует наоборот строки-факторы и наблюдения в колонках
        cov = np.cov(X.T)  # поэтому транспонируем сразу
        # eigenvectors, eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        X -= self.mean
        return np.dot(X, self.components.T)


if __name__ == "__main__":
    from sklearn import datasets
    import matplotlib.pyplot as plt

    data = datasets.load_iris()
    X = data.data
    y = data.target

    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X", X.shape)
    print("Shape of transformed X", X_projected.shape)

    X1 = X_projected[:, 0]
    X2 = X_projected[:, 1]

    plt.scatter(X1, X2, c=y, edgecolors='none', alpha=0.8, cmap=plt.cm.get_cmap('viridis', 3))

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()