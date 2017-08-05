import numpy as np
import tensorflow as tf
from . import optimizee


class LogisticRegression(optimizee.Optimizee):
    name = 'logistic_regression'

    def __init__(self, max_data_size=300, max_features=100):
        super(LogisticRegression, self).__init__()
        self.max_data_size = max_data_size
        self.max_features = max_features


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


    def get_initial_x(self, batch_size=1):
        self.num_features = np.random.randint(low=1, high=self.max_features)
        self.data_size    = np.random.randint(low=1, high=self.max_data_size)
    
        self.w  = np.random.normal(size=(batch_size, self.num_features))
        self.w0 = np.random.normal(size=(batch_size, 1))

        return np.concatenate([self.w, self.w0], axis=-1)
        

    def get_new_params(self, batch_size=1):
        X = np.random.normal(size=(batch_size, self.data_size, self.num_features))
        #y = np.random.randint(0, 2, size=(batch_size, self.data_size))
        #y = (np.dot(self.w, X.transpose(0, 2, 1)) + self.w0) > 0
        y = np.einsum('ai,aji->aj', self.w, X) + self.w0 > 0

        return {
            self.X: X,
            self.y: y,
            self.dim: self.num_features + 1
        }
                
        
    def get_next_dict(self, n_bptt_steps, batch_size=1):
        return { } 

