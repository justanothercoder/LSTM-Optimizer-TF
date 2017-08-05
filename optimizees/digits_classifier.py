import numpy as np
import tensorflow as tf
from tensorflow.python.ops.init_ops import glorot_uniform_initializer

from sklearn.datasets import load_digits, fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn import utils
from . import optimizee

class DIGITSClassifier(optimizee.Optimizee):
    name = 'digits_classifier'

    def __init__(self, num_units=20, num_layers=1, dataset_name='digits', activation='sigmoid', return_func=False):
        super(DIGITSClassifier, self).__init__()
        self.dataset_name = dataset_name

        if dataset_name == 'digits':
            dataset = load_digits(n_class=10)
        elif dataset_name == 'mnist':
            dataset = fetch_mldata('MNIST original', data_home='/srv/hd1/data/vyanush/')

        self.X, self.Y = dataset.data, dataset.target
        self.X, self.Y = utils.shuffle(self.X, self.Y)

        self.X = StandardScaler().fit_transform(self.X.astype(np.float32))

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
            self.x = tf.placeholder(tf.float32, [None, None, None, self.X.shape[1]], name='X') # n_bptt_steps * batch_size * data_size * num_features
            self.y = tf.placeholder(tf.int32, [None, None, None], name='y')

 
    def loss(self, x, i):
        self.coord_pos = 0
        self.coord_vector = x
        dims = [self.num_units] * self.num_layers

        # self.x[i].shape == (batch_size, data_size, n_inputs)
        #pred = tf.transpose(self.x[i], perm=[0, 2, 1])
        pred = self.x[i][0]

        activation = getattr(tf.nn, self.activation)

        with tf.variable_scope('nn_classifier/loss', custom_getter=self.custom_getter) as scope, tf.device('/gpu:0'):
            for n_outputs in dims:
                pred = tf.layers.dense(pred, n_outputs, activation=None)
                pred = tf.layers.batch_normalization(pred)
                pred = activation(pred)

            pred = tf.layers.dense(pred, 10)

            #pred = tf.transpose(pred, perm=[0, 2, 1]) # shape = (batch_size, data_size, n_classes)
            f = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y[i][0], logits=pred), axis=-1)

            p = tf.argmax(tf.nn.softmax(pred), axis=-1)
            print(p.get_shape(), self.y[i][0].get_shape())
            #acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(p, tf.int32), self.y[i][0]), tf.float32), axis=1)
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(p, tf.int32), self.y[i][0]), tf.float32), axis=-1)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                g = self.grad(x, f)

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
            self.batch_size = np.random.randint(low=1, high=self.X.shape[0] // 4 + 1)
        self.s = 0

        print("{} classifier; batch_size: {}".format(self.dataset_name, self.batch_size))

        #w = np.random.normal(0, 0.01, size=(batch_size, self.x_len))
        w = np.zeros(self.x_len)

        print(w.shape)

        for name, d in self.coord_vars.items():
            start, end = d['pos']

            #with tf.variable_scope('dummy_{}'.format(name), reuse=False):
            #    shape = d['shape']
            #    #print(type(shape[0]), type(shape[1]))
            #    init = (d['initializer'] or glorot_uniform_initializer(dtype=tf.float32))(shape)
            #    dummy = tf.get_variable('dummy', initializer=init)
            #    dummy.initializer.run()
            dummy = (d['initializer'] or glorot_uniform_initializer(dtype=tf.float32))(d['shape'])

            val = tf.get_default_session().run(dummy)
            w[start:end] = val.reshape(-1)

        return w[None, :]
        

    def get_new_params(self, batch_size=1):
        return {
            self.dim: self.x_len
        }

        
    def get_next_dict(self, n_bptt_steps, batch_size=1):
        x = np.zeros((n_bptt_steps, 1, self.batch_size, self.X.shape[1])) 
        y = np.zeros((n_bptt_steps, 1, self.batch_size)) 

        for i in range(n_bptt_steps):
            if self.s + self.batch_size > self.X.shape[0]:
                self.s = 0
            pos_cur, pos_next = self.s, self.s + self.batch_size

            pos_cur = np.random.randint(low=0, high=self.X.shape[0] - self.batch_size)
            pos_next = pos_cur + self.batch_size

            x[i] = np.tile(self.X[None, pos_cur:pos_next], (batch_size, 1, 1))
            y[i] = np.tile(self.Y[None, pos_cur:pos_next], (batch_size, 1, 1))

            self.s = pos_next

        return { 
            self.x: x,
            self.y: y,
        } 

