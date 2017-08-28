import numpy as np
import tensorflow as tf
from . import optimizee
from . import datagen
from problem_producer import set_random_state


class CorrectStochLogreg(optimizee.Optimizee):
    name = 'stochastic_logistic_regression'

    def __init__(self, min_data_size=100, max_data_size=1000, min_features=1, max_features=100, max_batch_size=0.1):
        super(CorrectStochLogreg, self).__init__()
        self.datagen = datagen.RandomNormal(min_data_size, max_data_size, min_features, max_features)
        self.max_batch_size = max_batch_size


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
        self.batch_size = np.random.randint(low=1, high=int(self.dataset.data_size * self.max_batch_size) + 1)
        
        w = np.random.normal(size=(batch_size, self.dataset.num_features))
        w0 = np.random.normal(size=(batch_size, 1), scale=0.1)
        init = np.concatenate([w, w0], axis=1)

        params = {self.dim: self.dataset.num_features + 1}
        return self.Problem(init, params, self.dataset, self.batch_size, self.x, self.y)

    #def sample_batch(self, batch_size):
    #    return self.datagen.sample_batch(batch_size)
        
    class Problem(optimizee.Problem):
        def __init__(self, init, params, dataset, batch_size, x, y):
            super(CorrectStochLogreg.Problem, self).__init__(init, params, name='correct_stoch_logreg')
            self.dataset = dataset
            self.batch_size = batch_size
            self.x = x
            self.y = y
            
            self.np_random = np.random.RandomState()
            self.np_random.set_state(np.random.get_state())


        def get_next_dict(self, n_bptt_steps, batch_size):
            x = np.zeros((n_bptt_steps, batch_size, self.batch_size, self.dataset.num_features), dtype=np.float32) 
            y = np.zeros((n_bptt_steps, batch_size, self.batch_size), dtype=np.int32)

            with set_random_state(self.np_random):
                random_batches = self.dataset.random_batch_iterator(n_bptt_steps, self.batch_size)

                for i, batch in enumerate(random_batches):
                    x[i], y[i] = batch 

            return { 
                self.x: x,
                self.y: y,
            } 
