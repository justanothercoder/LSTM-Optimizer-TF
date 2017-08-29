import numpy as np
import tensorflow as tf
from . import optimizee


class Quadratic(optimizee.Optimizee):
    name = 'quadratic'

    def __init__(self, low=20, high=100):
        super(Quadratic, self).__init__()
        self.low = low
        self.high = high


    def get_x_dim(self):
        return self.dim


    def build(self):
        with tf.variable_scope('quadratic'):
            self.W = tf.placeholder(tf.float32, [None, None, None], name='W')
            self.b = tf.placeholder(tf.float32, [None, None], name='b')
            self.dim = tf.placeholder(tf.int32, [], name='dim')

    
    def loss(self, x, i):
        b = tf.expand_dims(self.b, -1)

        f = tf.square(tf.matmul(self.W, tf.expand_dims(x, -1)) - b)
        f = tf.reduce_mean(tf.squeeze(f, axis=-1), axis=-1)
        g = self.grad(x, f)
        return f, g


    def sample_problem(self, batch_size=1):
        D = np.random.randint(low=self.low, high=self.high)
        init = np.random.normal(0, 0.1, size=(batch_size, D))

        W = np.random.normal(0, 0.1, size=(batch_size, D, D))
        b = np.random.normal(0, 0.1, size=(batch_size, D))

        params = {self.W: W, self.b: b, self.dim: D}
        return optimizee.SimpleNonStochProblem(init, params)
