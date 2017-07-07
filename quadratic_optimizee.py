import numpy as np
import tensorflow as tf


class Quadratic:
    name = 'quadratic'

    def __init__(self, low=20, high=100):
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
        x = tf.expand_dims(x, -1)

        f = (tf.matmul(self.W, x) - b)**2
        return tf.reduce_mean(tf.squeeze(f, axis=-1), axis=-1)

        #return tf.squeeze(tf.reduce_mean(, axis=1), axis=-1)


    def get_initial_x(self, batch_size=1):
        self.D = np.random.randint(low=self.low, high=self.high)
        #return np.random.normal(0, 0.1, size=self.D)

        return np.random.normal(0, 0.1, size=(batch_size, self.D))


    def get_new_params(self, batch_size=1):
        return {
            #self.W: np.random.normal(0, 0.1, size=(self.D, self.D)),
            #self.b: np.random.normal(0, 0.1, size=self.D),

            self.W: np.random.normal(0, 0.1, size=(batch_size, self.D, self.D)),
            self.b: np.random.normal(0, 0.1, size=(batch_size, self.D)),
            self.dim: self.D
        }
                
        
    def get_next_dict(self, n_bptt_steps, batch_size=1):
        return { } 
