import numpy as np
import tensorflow as tf


class Quadratic:
    name = 'quadratic'

    def __init__(self, low=20, high=100):
        self.low = low
        self.high = high


    def build(self):
        with tf.variable_scope('quadratic'):
            self.W = tf.placeholder(tf.float32, [None, None], name='W')
            self.b = tf.placeholder(tf.float32, [None], name='b')

    
    def loss(self, x, i):
        return tf.reduce_mean((tf.matmul(self.W, tf.expand_dims(x, 1)) - self.b)**2)


    def get_initial_x(self):
        self.D = np.random.randint(low=self.low, high=self.high)
        return np.random.normal(0, 0.1, size=self.D)


    def get_new_params(self):
        return {
            self.W: np.random.normal(0, 0.1, size=(self.D, self.D)),
            self.b: np.random.normal(0, 0.1, size=self.D),
        }
                
        
    def get_next_dict(self, n_bptt_steps):
        return { } 
