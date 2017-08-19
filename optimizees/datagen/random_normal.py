import numpy as np
from .dataset import Dataset


class RandomNormal:
    def __init__(self, min_data_size=100, max_data_size=1000, min_features=1, max_features=100):
        self.min_data_size = min_data_size
        self.max_data_size = max_data_size
        self.min_features = min_features
        self.max_features = max_features


    def sample_dataset(self, classification=True, data_size=None, num_features=None):
        data_size = data_size or np.random.randint(low=self.min_data_size, high=self.max_data_size + 1)
        num_features = num_features or np.random.randint(low=self.min_features, high=self.max_features + 1)

        X = np.random.normal(size=(data_size, num_features))

        w = np.random.normal(size=num_features)
        w0 = np.random.normal(size=1, scale=0.1)

        y = X.dot(w) + w0
        if classification:
            y = y > 0

        return Dataset(X.astype(np.float32), y, w=w, w0=w0)


    def sample_dataset_batch(self, batch_size, classification=True, data_size=None, num_features=None):
        data_size = data_size or np.random.randint(low=self.min_data_size, high=self.max_data_size + 1)
        num_features = num_features or np.random.randint(low=self.min_features, high=self.max_features + 1)

        X = np.empty((batch_size, data_size, num_features), dtype=np.float32)
        y = np.empty((batch_size, data_size))
        w = np.empty((batch_size, num_features), dtype=np.float32)
        w0 = np.empty((batch_size, 1), dtype=np.float32)

        for i in range(batch_size):
            d = self.sample_dataset(classification=classification, data_size=data_size, num_features=num_features)
            X[i], y[i] = d.X, d.y
            w[i], w0[i] = d.w, d.w0

        return Dataset(X, y, w=w, w0=w0)
