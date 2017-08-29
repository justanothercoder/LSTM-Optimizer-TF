import numpy as np
import tensorflow as tf
from . import optimizee
from . import datagen


class LogisticRegression(optimizee.Optimizee):
    name = 'logistic_regression'

    def __init__(self, max_data_size=300, max_features=100, min_data_size=100, min_features=1):
        super(LogisticRegression, self).__init__()
        self.datagen = datagen.RandomNormal(min_data_size, max_data_size, min_features, max_features)


    def get_x_dim(self):
        return self.dim


    def build(self):
        with tf.variable_scope('logistic_regression'):
            self.dim = tf.placeholder(tf.int32, [], name='dim')
            self.X = tf.placeholder(tf.float32, [None, None, None], name='X')
            self.y = tf.placeholder(tf.int32, [None, None], name='y')

    
    def loss(self, x, i):
        num_features = tf.shape(self.X)[2]

        w = x[:, :-1]
        w0 = x[:, -1]

        w = tf.expand_dims(w, axis=-2)
        w0 = tf.expand_dims(w0, axis=-1)

        XT = tf.transpose(self.X, perm=[0, 2, 1])
        score = tf.squeeze(tf.matmul(w, XT), axis=-2) + w0

        p = tf.clip_by_value(tf.sigmoid(score), 1e-5, 1 - 1e-5)

        y = tf.cast(self.y, tf.float32)
        f = -tf.reduce_mean(y * tf.log(p) + (1 - y) * tf.log(1 - p), axis=-1)
        g = self.grad(x, f)
        return f, g


    def sample_problem(self, batch_size=1):
        self.dataset = self.datagen.sample_dataset_batch(batch_size, classification=True)
        init = np.concatenate([self.dataset.w, self.dataset.w0], axis=1)
        params = {
            self.X: self.dataset.X,
            self.y: self.dataset.y,
            self.dim: num_features + 1
        }
        
        return optimizee.SimpleNonStochProblem(init, params)

