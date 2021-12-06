import numpy as np


class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # S_W, S_B
        mean_overall = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            # (4, n_c) * (n_c, 4) = (4, 4)
            S_W += (X_c - mean_c).T.dot((X_c - mean_c))
            n_c = X_c.shape[0]
            # (4, 1) * (4, 1).T = (4, 4)
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)  # (4,1)
            S_B += n_c * (mean_diff.dot(mean_diff.T))

        A = np.linalg.inv(S_W).dot(S_B)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]  # последняя конструкция развернёт вектор в одбратную сторону
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.linear_discriminants = eigenvectors[0:self.n_components]

    def transform(self, X):
        return np.dot(X, self.linear_discriminants.T)


if __name__ == "__main__":
    from sklearn import datasets
    import matplotlib.pyplot as plt

    data = datasets.load_iris()
    X = data.data
    y = data.target

    lda = LDA(2)
    lda.fit(X, y)
    X_projected = lda.transform(X)

    print("Shape of X", X.shape)
    print("Shape of transformed X", X_projected.shape)

    X1 = X_projected[:, 0]
    X2 = X_projected[:, 1]

    plt.scatter(X1, X2, c=y, edgecolors='none', alpha=0.8, cmap=plt.cm.get_cmap('viridis', 3))

    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Linear Discriminant 2')
    plt.colorbar()
    plt.show()

    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='none', alpha=0.8, cmap=plt.cm.get_cmap('viridis', 3))

    plt.xlabel('Axis 1')
    plt.ylabel('Axis 2')
    plt.colorbar()
    plt.show()