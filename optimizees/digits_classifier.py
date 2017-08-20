import numpy as np
import tensorflow as tf
from collections import namedtuple
from tensorflow.python.ops.init_ops import glorot_uniform_initializer

from . import optimizee
from . import datagen


class MLPClassifier(optimizee.Optimizee):
    name = 'mlp_classifier'

    def __init__(self, num_units=20, num_layers=1, dataset_name='digits', activation='sigmoid', return_func=False):
        super(MLPClassifier, self).__init__()

        if dataset_name == 'digits':
            self.datagen = datagen.Digits()
        elif dataset_name == 'mnist':
            self.datagen = datagen.MNIST()
        elif dataset_name == 'random':
            self.datagen = datagen.RandomNormal(min_data_size=100, max_data_size=1000, min_features=1, max_features=100)

        self.dataset_name = dataset_name
        self.dataset = self.datagen.sample_dataset()

        self.num_units = num_units
        self.num_layers = num_layers
        self.activation = activation
        self.return_func = return_func

        self.x_len = 0
        self.x_len_counted = False


    def get_x_dim(self):
        return self.dim


    def build(self):
        with tf.variable_scope('digits_classifier'):
            self.dim = tf.placeholder(tf.int32, [], name='dim')
            self.x = tf.placeholder(tf.float32, [None, 1, None, self.dataset.num_features], name='X') # n_bptt_steps * (batch_size=1) * data_size * num_features
            self.y = tf.placeholder(tf.int32, [None, 1, None], name='y')


    def loss(self, x, i):
        return self.loss_(x, self.x[i][0], self.y[i][0])

 
    def loss_(self, w, x, y):
        self.coord_pos = 0
        self.coord_vector = w
        dims = [self.num_units] * self.num_layers

        # self.x[i].shape == (batch_size, data_size, n_inputs)
        #pred = tf.transpose(self.x[i], perm=[0, 2, 1])
        pred = x
        activation = getattr(tf.nn, self.activation)

        with tf.variable_scope('nn_classifier/loss', custom_getter=self.custom_getter) as scope, tf.device('/gpu:0'):
            for n_outputs in dims:
                pred = tf.layers.dense(pred, n_outputs, activation=None)
                pred = tf.layers.batch_normalization(pred)
                pred = activation(pred)

            pred = tf.layers.dense(pred, 10)

            #pred = tf.transpose(pred, perm=[0, 2, 1]) # shape = (batch_size, data_size, n_classes)
            f = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred), axis=-1)

            p = tf.argmax(tf.nn.softmax(pred), axis=-1)
            p = tf.cast(p, tf.int32)
            acc = tf.reduce_mean(tf.cast(tf.equal(p, y), tf.float32), axis=-1)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                g = self.grad(w, f)

        if not self.x_len_counted:
            self.x_len = self.coord_pos
            self.x_len_counted = True
            self.vars_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

        if self.return_func:
            return f, g
        else:
            return acc, g


    def get_initial_x(self, batch_size=1):
        if self.dataset_name == 'mnist':
            self.batch_size = np.random.randint(low=1, high=256)
        else:
            self.batch_size = np.random.randint(low=1, high=self.dataset.data_size // 4 + 1)

        print("MLPClassifier; dataset: {}, batch_size: {}".format(self.dataset_name, self.batch_size))

        #w = np.random.normal(0, 0.01, size=(batch_size, self.x_len))
        w = np.zeros(self.x_len)
        print("x_len: ", w.shape)

        for name, d in self.coord_vars.items():
            start, end = d['pos']

            initializer = d['initializer'] or glorot_uniform_initializer(dtype=tf.float32)
            dummy = initializer(d['shape'])
            val = tf.get_default_session().run(dummy)

            w[start:end] = val.reshape(-1)

        return w[None, :]
        

    def get_new_params(self, batch_size=1):
        return {
            self.dim: self.x_len
        }

        
    def get_next_dict(self, n_bptt_steps, batch_size=1):
        x = np.zeros((n_bptt_steps, 1, self.batch_size, self.dataset.num_features)) 
        y = np.zeros((n_bptt_steps, 1, self.batch_size)) 

        random_batches = self.dataset.random_batch_iterator(n_bptt_steps, self.batch_size)

        for i, (x_, y_) in enumerate(random_batches):
            x[i] = x_[None]
            y[i] = y_[None]

        return { 
            self.x: x,
            self.y: y,
        } 

