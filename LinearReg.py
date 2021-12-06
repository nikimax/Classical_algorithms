import numpy as np


def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1500):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # градиент
            dw = (2/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2/n_samples) * np.sum(y_predicted - y)
            # обновление весов
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self.weights, self.bias

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=2, noise=20, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X_numpy, y_numpy, test_size=0.2)

    model = LinearRegression()
    w, b = model.fit(X_train, y_train)
    y_predict2 = model.predict(X_test)

    print("MSE on testing", MSE(y_test, y_predict2))
