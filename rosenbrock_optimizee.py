import numpy as np
import tensorflow as tf


class Rosenbrock:
    name = 'rosenbrock'

    def __init__(self, low=2, high=10):
        self.low = low
        self.high = high


    def get_x_dim(self):
        return self.dim


    def build(self):
        with tf.variable_scope('rosenbrock'):
            #self.a = tf.placeholder(tf.float32, [None], name='a')
            #self.b = tf.placeholder(tf.float32, [None], name='b')
            self.dim = tf.placeholder(tf.int32, [], name='dim')
            self.a = tf.placeholder(tf.float32, [None, None], name='a')
            self.b = tf.placeholder(tf.float32, [None, None], name='b')

    
    def loss(self, x, i):
        t0 = x[..., ::2]
        t1 = x[..., 1::2]

        s = tf.reduce_sum((self.a - t0)**2 + self.b * (t1 - t0**2)**2, axis=-1)
        return tf.clip_by_value(s, 0, 10**10)


    def get_initial_x(self, batch_size=1):
        self.D = np.random.randint(low=self.low, high=self.high)
    
        #x = np.random.normal(0, 0.1, size=(self.D, 1))
        #y = np.random.normal(0, 0.01, size=(self.D, 1))
        
        x = np.random.normal(0, 0.1, size=(batch_size, self.D, 1))
        y = np.random.normal(0, 0.01, size=(batch_size, self.D, 1))

        self.t = np.concatenate([x, y], axis=-1).reshape(batch_size, -1)
        return self.t


    def get_new_params(self, batch_size=1):
        return {
            #self.a: np.random.normal(0, 1, size=self.D),
            #self.b: np.random.uniform(10, 100, size=self.D),
            
            self.a: np.random.normal(self.t[..., ::2], 0.1, size=(batch_size, self.D)),
            self.b: np.random.uniform(10, 100, size=(batch_size, self.D)),
            self.dim: self.D * 2
        }
                
        
    def get_next_dict(self, n_bptt_steps):
        return { } 
