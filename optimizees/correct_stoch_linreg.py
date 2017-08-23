import numpy as np
import tensorflow as tf
from . import optimizee
from . import datagen


class CorrectStochLinreg(optimizee.Optimizee):
    name = 'stochastic_linear_regression'

    def __init__(self, min_data_size=1, max_data_size=1000, min_features=1, max_features=100):
        super(CorrectStochLinreg, self).__init__()
        self.datagen = datagen.RandomNormal(min_data_size=min_data_size, max_data_size=max_data_size, min_features=min_features, max_features=max_features)


    def get_x_dim(self):
        return self.dim


    def build(self):
        with tf.variable_scope('stochastic_linear_regression'):
            self.dim = tf.placeholder(tf.int32, [], name='dim')
            self.x = tf.placeholder(tf.float32, [None, None, None, None], name='x_batch') # n_bptt_steps * batch_size * data_size * num_features
            self.y = tf.placeholder(tf.float32, [None, None, None], name='y_batch')

    
    def loss(self, x, i):
        w  = tf.expand_dims(x[:, :-1], axis=-2)
        w0 = tf.expand_dims(x[:,  -1], axis=-1)

        xT = tf.transpose(self.x[i], perm=[0, 2, 1])
        score = tf.squeeze(tf.matmul(w, xT), axis=-2) + w0

        f = tf.reduce_mean(tf.square(score - self.y[i]), axis=-1) / tf.cast(tf.shape(w)[1], tf.float32) / 100
        g = self.grad(x, f)
        return f, g


    def sample_problem(self, batch_size=1):
        self.dataset = self.datagen.sample_dataset_batch(batch_size, classification=False)
        self.batch_size = np.random.randint(low=1, high=self.dataset.data_size + 1)
    
        init = np.random.normal(size=(batch_size, self.dataset.num_features + 1))
        params = {self.dim: self.dataset.num_features + 1}
        return init, params

        
    def get_next_dict(self, n_bptt_steps, batch_size=1):
        x = np.zeros((n_bptt_steps, batch_size, self.batch_size, self.dataset.num_features)) 
        y = np.zeros((n_bptt_steps, batch_size, self.batch_size)) 

        batches = self.dataset.batch_iterator(n_bptt_steps, self.batch_size, shuffle=False)

        for i, batch in enumerate(batches):
            x[i], y[i] = batch

        return { 
            self.x: x,
            self.y: y,
        } 
