import numpy as np
import tensorflow as tf
from . import optimizee
from . import datagen


class CorrectStochLogreg(optimizee.Optimizee):
    name = 'stochastic_logistic_regression'

    def __init__(self, min_data_size=100, max_data_size=1000, min_features=1, max_features=100):
        super(CorrectStochLogreg, self).__init__()
        self.datagen = datagen.RandomNormal(min_data_size, max_data_size, min_features, max_features)


    def get_x_dim(self):
        return self.dim


    def build(self):
        with tf.variable_scope('stochastic_logistic_regression'):
            self.dim = tf.placeholder(tf.int32, [], name='dim')
            self.x = tf.placeholder(tf.float32, [None, None, None, None], name='x_batch') # n_bptt_steps * batch_size * data_size * num_features
            self.y = tf.placeholder(tf.int32, [None, None, None], name='y_batch')


    def loss(self, x, i):
        w  = tf.expand_dims(x[:, :-1], axis=-2)
        w0 = tf.expand_dims(x[:,  -1], axis=-1)

        xT = tf.transpose(self.x[i], perm=[0, 2, 1])
        score = tf.squeeze(tf.matmul(w, xT), axis=-2) + w0

        p = tf.clip_by_value(tf.sigmoid(score), 1e-5, 1 - 1e-5)

        y = tf.cast(self.y[i], tf.float32)
        f = -tf.reduce_mean(y * tf.log(p) + (1 - y) * tf.log(1 - p), axis=-1)
        g = self.grad(x, f)
        return f, g


    def sample_problem(self, batch_size=1):
        self.dataset = self.datagen.sample_dataset_batch(batch_size, classification=True)
        self.batch_size = np.random.randint(low=1, high=self.dataset.data_size // 10 + 1)

        self.w  = np.random.normal(size=(batch_size, self.num_features))
        self.w0 = np.random.normal(size=(batch_size, 1), scale=0.1)
        
        self.data_size    = np.random.randint(low=self.min_data_size, high=self.max_data_size)
        self.batch_size   = np.random.randint(low=1, high=self.data_size // 10 + 2)
            
        self.X = np.random.normal(size=(batch_size, self.data_size, self.num_features))
        self.Y = np.einsum('ai,aji->aj', self.w, self.X) + self.w0 > 0
        self.s = 0
        
        init = np.random.normal(size=(batch_size, self.dataset.num_features + 1))
        params = {self.dim: self.dataset.num_features + 1}
        return init, params

        
    def get_next_dict(self, n_bptt_steps, batch_size=1):
        x = np.zeros((n_bptt_steps, batch_size, self.batch_size, self.dataset.num_features), dtype=np.float32) 
        y = np.zeros((n_bptt_steps, batch_size, self.batch_size), dtype=np.int32) 

        random_batches = self.dataset.random_batch_iterator(n_bptt_steps, self.batch_size)

        for i, batch in enumerate(random_batches):
            x[i], y[i] = batch 

        return { 
            self.x: x,
            self.y: y,
        } 


    def sample_batch(self, batch_size):
        return self.datagen.sample_batch(batch_size)

        #ind = np.random.randint(low=0, high=self.X.shape[1] - batch_size + 1)
        #return self.X[:, ind:ind + batch_size].astype(np.float32), self.Y[:, ind:ind + batch_size].astype(np.float32)
