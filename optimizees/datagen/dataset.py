import numpy as np


class Dataset:
    def __init__(self, X, y, batched=False, **kwargs):
        self.X = X
        self.y = y

        self.s = 0
        self.batched = batched

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

        for i in range(n_batches):
            if self.s + batch_size > self.data_size:
                self.s = 0

            ind = slice(self.s, self.s + batch_size)
            if shuffle:
                ind = indices[ind]

            self.s += batch_size
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
