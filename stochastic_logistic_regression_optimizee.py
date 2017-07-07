import numpy as np
import tensorflow as tf


class StochasticLogisticRegression:
    name = 'stochastic_logistic_regression'

    def __init__(self, max_data_size=1000, max_features=100):
        self.max_data_size = max_data_size
        self.max_features = max_features


    def get_x_dim(self):
        return self.dim


    def build(self):
        with tf.variable_scope('rosenbrock'):
            self.dim = tf.placeholder(tf.int32, [], name='dim')
            self.x = tf.placeholder(tf.float32, [None, None, None, None], name='X')
            self.y = tf.placeholder(tf.int32, [None, None, None], name='y')

    
    def loss(self, x, i):
        num_features = tf.shape(self.x)[-1]

        w  = tf.expand_dims(x[:, :-1], axis=-2)
        w0 = tf.expand_dims(x[:,  -1], axis=-1)

        score = tf.matmul(w, tf.transpose(self.x[i], perm=[0, 2, 1])) + w0
        p = tf.clip_by_value(tf.sigmoid(score), 1e-5, 1 - 1e-5)

        y = tf.cast(self.y[i], tf.float32)
        return -tf.reduce_mean(y * tf.log(p) + (1 - y) * tf.log(1 - p), axis=-1)


    def get_initial_x(self, batch_size=1):
        self.num_features = np.random.randint(low=1, high=self.max_features)
        self.data_size    = np.random.randint(low=1, high=self.max_data_size)
        self.batch_size   = np.random.randint(low=1, high=self.data_size)
    
        self.w  = np.random.normal(size=(batch_size, self.num_features))
        self.w0 = np.random.normal(size=(batch_size, 1))
            
        self.X = np.random.normal(size=(batch_size, self.data_size, self.num_features)),
        self.Y = np.random.randint(0, 2, size=(batch_size, self.data_size)),
        self.s = 0

        return np.concatenate([self.w, self.w0], axis=-1)
        

    def get_new_params(self, batch_size=1):
        return {
            self.dim: self.num_features + 1
        }

        
    def get_next_dict(self, n_bptt_steps, batch_size=1):
        x = np.zeros((n_bptt_steps, batch_size, self.batch_size, self.num_features)) 
        y = np.zeros((n_bptt_steps, batch_size, self.batch_size)) 

        for _ in range(n_bptt_steps):
            if self.s + self.batch_size > self.data_size:
                self.s = 0
            pos_cur, pos_next = self.s, self.s + self.batch_size

            x[i] = self.X[:, pos_cur:pos_next]
            y[i] = self.Y[:, pos_cur:pos_next]

            self.s = pos_next

        return { 
            self.x: x,
            self.y: y,
        } 
