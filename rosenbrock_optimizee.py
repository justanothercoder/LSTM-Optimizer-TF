import numpy as np
import tensorflow as tf


class Rosenbrock:
    name = 'rosenbrock'

    def __init__(self, low=2, high=10):
        self.low = low
        self.high = high


    def build(self):
        with tf.variable_scope('rosenbrock'):
            self.a = tf.placeholder(tf.float32, [None], name='a')
            self.b = tf.placeholder(tf.float32, [None], name='b')

    
    def loss(self, x, i):
        t0 = x[::2]
        t1 = x[1::2]

        s = tf.reduce_sum((self.a - t0)**2 + self.b * (t1 - t0**2)**2)
        return tf.clip_by_value(s, 0, 10**10)


    def get_initial_x(self):
        self.D = np.random.randint(low=self.low, high=self.high)
    
        x = np.random.normal(0, 0.1, size=(self.D, 1))
        y = np.random.normal(0, 0.01, size=(self.D, 1))

        t = np.hstack([x, y]).reshape(-1)

        return t


    def get_new_params(self):
        return {
            self.a: np.random.normal(0, 1, size=self.D),
            self.b: np.random.uniform(10, 100, size=self.D),
        }
                
        
    def get_next_dict(self, n_bptt_steps):
        return { } 
