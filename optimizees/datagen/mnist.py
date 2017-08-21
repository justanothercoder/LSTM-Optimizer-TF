import numpy as np
from sklearn import utils
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler

from .dataset import Dataset


class MNIST:
    def __init__(self):
        self.mnist = fetch_mldata('MNIST original', data_home='/srv/hd1/data/vyanush/')


    def sample_dataset(self, shuffle=True, transform=True):
        X = self.mnist.data
        y = self.mnist.target.astype(np.int32)

        if shuffle:
            utils.shuffle(X, y)
        
        if transform:
            X = StandardScaler().fit_transform(X.astype(np.float32)).astype(np.float32)

        return Dataset(X, y)