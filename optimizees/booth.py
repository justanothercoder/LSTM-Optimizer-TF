import numpy as np
import tensorflow as tf
from . import optimizee


class Booth(optimizee.Optimizee):
    name = 'beale'

    def __init__(self, low=2, high=10):
        self.low = low
        self.high = high


    def get_x_dim(self):
        return self.dim


    def build(self):
        with tf.variable_scope('booth'):
            self.dim = tf.placeholder(tf.int32, [], name='dim')
            self.a = tf.placeholder(tf.float32, [None, None], name='a')
            self.b = tf.placeholder(tf.float32, [None, None], name='b')

    
    def loss(self, x, i):
        t0 = x[..., ::2]
        t1 = x[..., 1::2]

        s = tf.reduce_sum(
            (t0 + 2 * t1 - self.a - 2 * self.b)**2
            + (2 * t0 + t1 - 2 * self.a - self.b)**2, axis=-1)
        g = self.grad(x, s)
        return s, g


    def get_initial_x(self, batch_size=1):
        self.D = np.random.randint(low=self.low, high=self.high)
    
        x = np.random.normal(1, 0.1, size=(batch_size, self.D, 1))
        y = np.random.normal(3, 0.1, size=(batch_size, self.D, 1))

        self.t = np.concatenate([x, y], axis=-1).reshape(batch_size, -1)
        return self.t


    def get_new_params(self, batch_size=1):
        return {
            self.a: np.random.normal(self.t[..., ::2], 0.1, size=(batch_size, self.D)),
            self.b: np.random.normal(self.t[..., 1::2], 0.1, size=(batch_size, self.D)),
            self.dim: self.D * 2
        }
                
        
    def get_next_dict(self, n_bptt_steps, batch_size=1):
        return { } 


