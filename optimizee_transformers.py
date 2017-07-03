import numpy as np
import tensorflow as tf


class ClipByValue:
    def __init__(self, optimizee, clip_low, clip_high):
        self.optim = optimizee
        self.clip_low = clip_low
        self.clip_high = clip_high


    def build(self):
        self.optim.build()


    def loss(self, x, i):
        f = self.optim.loss(x, i)
        return tf.clip_by_value(f, self.clip_low, self.clip_high)

    
    def get_initial_x(self, batch_size=1):
        return self.optim.get_initial_x(batch_size)


    def get_new_params(self, batch_size=1):
        return self.optim.get_new_params(batch_size)


    def get_next_dict(self, n_bptt_steps):
        return self.optim.get_next_dict(n_bptt_steps)


class UniformRandomScaling:
    def __init__(self, optimizee, r=3.0):
        self.optim = optimizee
        self.r = r


    def build(self):
        self.optim.build()

        with tf.variable_scope('random_scaling'):
            self.c = tf.placeholder(tf.float32, [None, None], name='c')


    def loss(self, x, i):
        f = self.optim.loss(self.c * x, i)
        return f

    
    def get_initial_x(self, batch_size=1):
        x = self.optim.get_initial_x(batch_size)
        self.coef = np.random.uniform(-self.r, self.r, size=x.shape)
        return x / self.coef


    def get_new_params(self, batch_size=1):
        d = self.optim.get_new_params(batch_size)
        d[self.c] = self.coef
        return d


    def get_next_dict(self, n_bptt_steps):
        return self.optim.get_next_dict(n_bptt_steps)
