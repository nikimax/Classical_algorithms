import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)  # отсортирует уникальные значения классов
        n_classes = len(self._classes)  # взяли число классов

        # создаём пустые матрицы вероятнстей и средних для каждого класса
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # выделяем набор Х-ов для каждого класса У и считаем для каждого набора среднее, var, priors
        for idx, c in enumerate(self._classes):
            X_c = X[c == y]
            if __name__ == '__main__':
                self._mean[idx, :] = X_c.mean(axis=0)
                self._var[idx, :] = X_c.var(axis=0)
                self._priors[idx] = X_c.shape[0]/float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._normal_destrib(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]

    # пишем функцию плотности для нрмальнго распределения
    def _normal_destrib(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x-mean)**2 / (2 * var))
        denominator = np.sqrt(2*np.pi * var)
        return numerator/denominator


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy


if __name__ == "__main__":
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20)
    plt.show()

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    print("Naive Bayes accuracy:", accuracy(y_test, predictions))

