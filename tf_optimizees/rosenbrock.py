import numpy as np
import tensorflow as tf


class Rosenbrock
    def __init__(self, low=2, high=10):
        self.low = low
        self.high = high


    def build(self):
        a, b, dim = self.inputs()
        loss = self.loss(a, b, dim)


    def inputs(self):
        a = tf.placeholder(tf.float32, [None], name='a')
        b = tf.placeholder(tf.float32, [None], name='b')
        dim = tf.placeholder(tf.int32, [], name='dim')

        return a, b, dim


    def loss(self, x, i):
        t0 = x[..., ::2]
        t1 = x[..., 1::2]

        s = tf.reduce_sum(tf.square(self.a - t0) + self.b * tf.square(t1 - tf.square(t0)), axis=-1)
        g = self.grad(x, s)
        return s, g


    def get_initial_x(self, batch_size=1):
        self.D = np.random.randint(low=self.low, high=self.high)
    
        x = np.random.normal(0, 0.1, size=(batch_size, self.D, 1))
        y = np.random.normal(0, 0.01, size=(batch_size, self.D, 1))

        self.t = np.concatenate([x, y], axis=-1).reshape(batch_size, -1)
        return self.t


    def get_new_params(self, batch_size=1):
        return {
            self.a: np.random.normal(self.t[..., ::2], 0.1, size=(batch_size, self.D)),
            self.b: np.random.uniform(10, 100, size=(batch_size, self.D)),
            self.dim: self.D * 2
        }
