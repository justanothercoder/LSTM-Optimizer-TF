import numpy as np


class Dataset:
    def __init__(self, X, y, **kwargs):
        self.X = X
        self.y = y

        for k, v in kwargs.items():
            setattr(self, k, v)


    @property
    def data_size(self):
        return self.X.shape[-2]


    @property
    def num_features(self):
        return self.X.shape[-1]


    def batch_iterator(self, n_batches, batch_size, shuffle=True):
        if shuffle:
            indices = np.arange(self.data_size)
            np.random.shuffle(indices)

        s = 0

        for i in range(n_batches):
            if s + batch_size > self.data_size:
                s = 0

            if shuffle:
                ind = indices[s:s + batch_size]
            else:
                ind = slice(s, s + batch_size)

            s += batch_size
            yield self[ind]


    def random_batch_iterator(self, n_batches, batch_size):
        for _ in range(n_batches):
            s = np.random.randint(low=0, high=self.data_size - batch_size)
            #yield self.X[s:s + batch_size], self.y[s:s + batch_size]
            yield self[s:s + batch_size]



    def sample_batch(self, batch_size):
        s = np.random.randint(low=0, high=self.data_size - batch_size + 1)
        return self[s:s + batch_size]


    def __getitem__(self, s):
        if len(self.X.shape) == 3:
            return self.X[:, s], self.y[:, s]
        else:
            return self.X[s], self.y[s]
