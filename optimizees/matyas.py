import numpy as np
import tensorflow as tf
from . import optimizee


class Matyas(optimizee.Optimizee):
    name = 'matyas'

    def __init__(self, low=2, high=10):
        self.low = low
        self.high = high


    def get_x_dim(self):
        return self.dim


    def build(self):
        with tf.variable_scope('matyas'):
            self.dim = tf.placeholder(tf.int32, [], name='dim')

    
    def loss(self, x, i):
        t0 = x[..., ::2]
        t1 = x[..., 1::2]

        s = tf.reduce_sum(0.26 * (t0**2 + t1**2) - 0.48 * t0 * t1, axis=-1)
        g = self.grad(x, s)
        return s, g


    def get_initial_x(self, batch_size=1):
        self.D = np.random.randint(low=self.low, high=self.high)
    
        x = np.random.normal(0, 2, size=(batch_size, self.D, 1))
        y = np.random.normal(0, 2, size=(batch_size, self.D, 1))

        self.t = np.concatenate([x, y], axis=-1).reshape(batch_size, -1)
        return self.t


    def get_new_params(self, batch_size=1):
        return {
            self.dim: self.D * 2
        }
                
        
    def get_next_dict(self, n_bptt_steps, batch_size=1):
        return { } 

