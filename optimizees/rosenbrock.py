import numpy as np
import tensorflow as tf
from . import optimizee


class Rosenbrock(optimizee.Optimizee):
    name = 'rosenbrock'

    def __init__(self, low=2, high=10):
        super(Rosenbrock, self).__init__()
        self.low = low
        self.high = high


    def get_x_dim(self):
        return self.dim


    def build(self):
        with tf.variable_scope('rosenbrock'):
            self.dim = tf.placeholder(tf.int32, [], name='dim')
            self.a = tf.placeholder(tf.float32, [None, None], name='a')
            self.b = tf.placeholder(tf.float32, [None, None], name='b')

    
    def loss(self, x, i):
        t0 = x[..., ::2]
        t1 = x[..., 1::2]

        s = tf.reduce_sum(tf.square(self.a - t0) + self.b * tf.square(t1 - tf.square(t0)), axis=-1)
        #return tf.clip_by_value(s, 0, 10**10)
        g = self.grad(x, s)
        return s, g


    def sample_problem(self, batch_size=1):
        D = np.random.randint(low=self.low, high=self.high)
    
        x = np.random.normal(0, 0.1, size=(batch_size, D, 1))
        y = np.random.normal(0, 0.01, size=(batch_size, D, 1))
        init = np.concatenate([x, y], axis=-1).reshape(batch_size, -1)

        a = np.random.normal(init[..., ::2], 0.1, size=(batch_size, D))
        b = np.random.uniform(10, 100, size=(batch_size, D))
        
        params = {self.a: a, self.b: b, self.dim: D * 2}
        return init, params 
                
        
    def get_next_dict(self, n_bptt_steps, batch_size=1):
        return { } 
