import numpy as np
import tensorflow as tf
from . import optimizee
from . import datagen


class BayesianLogreg(optimizee.Optimizee):
    name = 'bayesian_logistic_regression'

    def __init__(self, min_data_size=100, max_data_size=1000, min_features=1, max_features=100, max_batch_size=0.1):
        super(BayesianLogreg, self).__init__()
        self.datagen = datagen.RandomNormal(min_data_size, max_data_size, min_features, max_features)
        self.max_batch_size = max_batch_size


    def get_x_dim(self):
        return self.dim


    def build(self):
        with tf.variable_scope('bayesian_logistic_regression'):
            self.dim = tf.placeholder(tf.int32, [], name='dim')
            self.x = tf.placeholder(tf.float32, [None, None, None], name='x_batch') # n_bptt_steps * data_size * num_features
            self.y = tf.placeholder(tf.int32, [None, None], name='y_batch')
            self.logalpha = tf.placeholder(tf.float32, [None, None], name='alpha')


    def loss(self, x, i):
        theta = x[0, :self.dim // 2]
        logsigma = x[0, self.dim // 2: self.dim]

        eps = tf.random_normal(tf.shape(theta))
        w = theta + tf.exp(logsigma) * eps

        #score = tf.squeeze(tf.matmul(self.x[i], tf.expand_dims(w, -1)), axis=-2)
        score = tf.matmul(self.x[i], tf.expand_dims(w, -1))
        p = tf.clip_by_value(tf.sigmoid(score), 1e-5, 1 - 1e-5)

        y = tf.cast(self.y[i], tf.float32)
        f = -tf.reduce_mean(y * tf.log(p) + (1 - y) * tf.log(1 - p))

        n = tf.shape(w)[0]
        #n = tf.shape(w)[1]

        sigma = tf.exp(logsigma)
        alpha = tf.exp(self.logalpha)

        KL = 0.5 * tf.cast(n, tf.float32) * (tf.reduce_sum(self.logalpha - logsigma + sigma / alpha) - 1) + 0.5 * tf.reduce_sum(theta * theta / alpha)
        f = f + KL
        f = tf.expand_dims(f, 0)
        g = self.grad(x, f)

        return f, g


    def sample_problem(self, batch_size=1):
        self.dataset = self.datagen.sample_dataset(classification=True)
        self.batch_size = np.random.randint(low=1, high=int(self.dataset.data_size * self.max_batch_size) + 1)

        n_f = self.dataset.num_features
        
        init = np.random.normal(size=(batch_size, n_f * 2))
        logalpha = np.random.normal(size=(batch_size, n_f), scale=0.1)

        params = {self.dim: n_f * 2, self.logalpha: logalpha}
        return init, params

        
    def get_next_dict(self, n_bptt_steps, batch_size=1):
        x = np.zeros((n_bptt_steps, self.batch_size, self.dataset.num_features), dtype=np.float32) 
        y = np.zeros((n_bptt_steps, self.batch_size), dtype=np.int32) 

        random_batches = self.dataset.random_batch_iterator(n_bptt_steps, self.batch_size)

        for i, batch in enumerate(random_batches):
            x[i], y[i] = batch 

        return { 
            self.x: x,
            self.y: y,
        } 


    def sample_batch(self, batch_size):
        return self.datagen.sample_batch(batch_size)
