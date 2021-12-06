import numpy as np
from KNN import euclidean_distance
import matplotlib.pyplot as plt

np.random.seed(42)

class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample idx for each cluster
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # optimization
        for _ in range(self.max_iters):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()
            # update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self.plot_steps:
                self.plot()
            # check if converged
            if self._is_converged(centroids_old, self.centroids):
                break
        # return cluster label
        return self._gen_cluster_labels(self.clusters)

    def _gen_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for clusters_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = clusters_idx
        return labels

    # Если центройды на новом шаге не изменили положение - прерываем цикл
    def _is_converged(self, centroids_old, centroids_new):
        distances = [euclidean_distance(centroids_old[i], centroids_new[i]) for i in range(self.K)]
        return sum(distances) == 0

    # для каждой точки пространства считаем ближайший центройд и определяем точку в соответствующий кластер
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    # расчёт расстояния от точки до каждого центройда и потом берём меньшее расстоние
    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    # просто заменяем центройд на среднее по кластру
    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)  #  creates scatter plot, where each cluster has its own color
        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)
        plt.show()


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y = make_blobs(centers=4, n_samples=150, n_features=2, shuffle=True, random_state=1234)
    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(K=clusters, max_iters=150, plot_steps=True)
    y_pred = k.predict(X)
