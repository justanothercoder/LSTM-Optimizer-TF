import numpy as np
from sklearn import utils
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler


class Digits:
    def __init__(self):
        self.digits = load_digits(n_class=10)


    def sample_dataset(self, shuffle=True):
        X = self.digits.data
        y = self.digits.target

        if shuffle:
            utils.shuffle(X, y)

        if transform:
            X = StandardScaler().fit_transform(X.astype(np.float32))

        return Dataset(X, y)
