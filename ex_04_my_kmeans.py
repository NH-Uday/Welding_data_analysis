from typing import Literal
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from dtaidistance import dtw

DISTANCE_METRICS = Literal['euclidean', 'manhattan', 'dtw']
INIT_METHOD = Literal['random', 'kmeans++']


class MyKMeans:
    def __init__(self, k: int, max_iter: int = 100,
                 distance_metric: DISTANCE_METRICS = 'euclidean',
                 init_method: INIT_METHOD = 'kmeans++'):
        self.k: int = k
        self.max_iter: int = max_iter
        self.centroids: np.ndarray | None = None
        self.distance_metric: DISTANCE_METRICS = distance_metric
        self.inertia_: float | None = None
        self.init_method: INIT_METHOD = init_method

    def fit(self, x: np.ndarray | pd.DataFrame):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        elif not isinstance(x, np.ndarray):
            raise ValueError('Input data must be a numpy array or a pandas DataFrame')

        if x.ndim not in (2, 3):
            raise ValueError('Input data must be a 2D or 3D array')
        x = np.asarray(x, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Input data must contain only finite values')
        if not isinstance(self.k, (int, np.integer)) or not 1 <= self.k <= len(x):
            raise ValueError('k must be between 1 and the number of samples')
        if self.max_iter < 1:
            raise ValueError('max_iter must be at least 1')
        if self.distance_metric not in ('euclidean', 'manhattan', 'dtw'):
            raise ValueError(f'Invalid distance metric: {self.distance_metric}')
        if self.init_method not in ('random', 'kmeans++'):
            raise ValueError(f'Invalid initialization method: {self.init_method}')

        self.centroids = self._initialize_centroids(x)
        for _ in range(self.max_iter):
            distances = self._compute_distance(x, self.centroids)
            labels = np.argmin(distances, axis=1)
            updated = self.centroids.copy()
            for j in range(self.k):
                members = x[labels == j]
                if len(members):
                    updated[j] = members.mean(axis=0)
                else:
                    # Re-seed an empty cluster with a poorly represented sample.
                    nearest = distances.min(axis=1)
                    updated[j] = x[np.argmax(nearest)]
            if np.allclose(updated, self.centroids, rtol=1e-6, atol=1e-8):
                self.centroids = updated
                break
            self.centroids = updated

        distances = self._compute_distance(x, self.centroids)
        closest = distances.min(axis=1)
        self.inertia_ = float(np.sum(closest ** 2))
        return self

    def fit_predict(self, x: np.ndarray | pd.DataFrame):
        if isinstance(x, pd.DataFrame):
            x = x.values
        elif not isinstance(x, np.ndarray):
            raise ValueError('Input data must be a numpy array or a pandas DataFrame')
        self.fit(x)
        return self.predict(x)

    def predict(self, x: np.ndarray | pd.DataFrame):
        if self.centroids is None:
            raise ValueError('Fit the model before calling predict')
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        distances = self._compute_distance(np.asarray(x, dtype=float), self.centroids)
        return np.argmin(distances, axis=1)

    def _initialize_centroids(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        if not 1 <= self.k <= n:
            raise ValueError('k must be between 1 and the number of samples')
        if self.init_method == 'random':
            return x[np.random.choice(n, self.k, replace=False)].copy()
        if self.init_method != 'kmeans++':
            raise ValueError(f'Invalid initialization method: {self.init_method}')

        chosen = [int(np.random.randint(n))]
        while len(chosen) < self.k:
            distances = self._compute_distance(x, x[chosen])
            weights = np.min(distances, axis=1) ** 2
            weights[chosen] = 0.0
            total = weights.sum()
            if total <= 0 or not np.isfinite(total):
                remaining = np.setdiff1d(np.arange(n), chosen)
                nxt = int(np.random.choice(remaining))
            else:
                nxt = int(np.random.choice(n, p=weights / total))
            chosen.append(nxt)
        return x[chosen].copy()

    def _compute_distance(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        centroids = np.asarray(centroids, dtype=np.float64)
        if x.ndim not in (2, 3) or centroids.ndim != x.ndim:
            raise ValueError('Input data must be a 2D or 3D array with matching dimensions')
        if self.distance_metric == 'dtw':
            return self._dtw(x, centroids)
        if x.shape[1:] != centroids.shape[1:]:
            raise ValueError('Samples and centroids must have matching feature shapes')
        a = x.reshape(len(x), -1)
        b = centroids.reshape(len(centroids), -1)
        delta = a[:, None, :] - b[None, :, :]
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum(delta ** 2, axis=2))
        if self.distance_metric == 'manhattan':
            return np.sum(np.abs(delta), axis=2)
        raise ValueError(f'Invalid distance metric: {self.distance_metric}')

    def _dtw(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        result = np.empty((len(x), len(centroids)), dtype=float)
        for i, sample in enumerate(x):
            for j, centroid in enumerate(centroids):
                if x.ndim == 2:
                    result[i, j] = dtw.distance_fast(
                        np.ascontiguousarray(sample, dtype=np.double),
                        np.ascontiguousarray(centroid, dtype=np.double))
                else:
                    if sample.shape[1] != centroid.shape[1]:
                        raise ValueError('DTW samples and centroids must have the same channel count')
                    channel_dists = [
                        dtw.distance_fast(
                            np.ascontiguousarray(sample[:, c], dtype=np.double),
                            np.ascontiguousarray(centroid[:, c], dtype=np.double))
                        for c in range(sample.shape[1])
                    ]
                    result[i, j] = np.linalg.norm(channel_dists)
        return result