import numpy as np
import tensorflow as tf
from . import optimizee

from sklearn.datasets import load_digits, fetch_mldata
from sklearn.preprocessing import StandardScaler


class ConvClassifier(optimizee.Optimizee):
    name = 'conv_classifier'

    def __init__(self, num_filters=20, num_layers=1, dataset_name='digits', activation='relu'):
        if dataset_name == 'digits':
            dataset = load_digits(n_class=10)
        elif dataset_name == 'mnist':
            dataset = fetch_mldata('MNIST original', data_home='/srv/hd1/data/vyanush/')

        self.X, self.Y = dataset.data, dataset.target
        self.X = StandardScaler().fit_transform(self.X).astype(np.float32)

        if dataset_name == 'digits':
            self.X = self.X.reshape(-1, 8, 8, 1)
        elif dataset_name == 'mnist':
            self.X = self.X.reshape(-1, 28, 28, 1)

        self.num_filters = num_filters
        self.num_layers = num_layers
        self.activation = activation
        self.filter_width = 3


    def get_x_dim(self):
        return self.dim


    def build(self):
        with tf.variable_scope('digits_classifier'):
            self.dim = tf.placeholder(tf.int32, [], name='dim')
            self.x = tf.placeholder(tf.float32, [None, None, None, None, None, None], name='X') # n_bptt_steps * batch_size * data_size * num_features
            self.y = tf.placeholder(tf.int32, [None, None, None], name='y')

 
    def loss(self, x, i):
        activation = getattr(tf.nn, self.activation)

        weights = []

        fs = []
        s = 0

        dims = [1] + [self.num_filters] * self.num_layers

        # self.x[i].shape == (data_size, w, h, n_filters)
        pred = self.x[i][0]

        for i in range(1, len(dims)):
            n_input_filters = dims[i - 1]
            n_output_filters = dims[i]
            dim = self.filter_width * self.filter_width * n_input_filters * n_output_filters

            W = x[0, s: s + dim]
            W = tf.reshape(W, [self.filter_width, self.filter_width, n_input_filters, n_output_filters])
            b = x[0, s + dim: s + dim + n_output_filters]

            pred = tf.nn.conv2d(pred, W, strides=[1, 1, 1, 1], padding='SAME')
            pred = tf.nn.bias_add(pred, b)
            pred = tf.nn.max_pool(pred, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pred = activation(pred)

            s += dim + n_output_filters

        pred = tf.reduce_mean(pred, axis=(1, 2))
        dim = n_output_filters * 10
        W = x[0, s: s + dim]
        W = tf.reshape(W, [n_output_filters, 10])
        b = x[0, s + dim: s + dim + 10]

        pred = tf.matmul(pred, W) + tf.expand_dims(b, 0)

        #f = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y[i][0], logits=pred), axis=-1)
        f = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y[i][0], logits=pred)
        f = tf.expand_dims(f, 0)
    
        p = tf.argmax(tf.nn.softmax(pred), axis=-1)
        #acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(p, tf.int32), self.y[i][0]), tf.float32), axis=1)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(p, tf.int32), self.y[i][0]), tf.float32), axis=-1)
        acc = tf.expand_dims(acc, 0)

        g = self.grad(x, f)


        #return f, g
        return acc, g


    def get_initial_x(self, batch_size=1):
        self.batch_size = np.random.randint(low=1, high=self.X.shape[0] // 4 + 1)
        print("Digits classifier; batch_size: ", self.batch_size)

        self.x_len = 0
    
        dims = [1] + [self.num_filters] * self.num_layers
        for i in range(1, len(dims)):
            n_input_filters = dims[i - 1]
            n_output_filters = dims[i]
            dim = self.filter_width * self.filter_width * n_input_filters * n_output_filters
            self.x_len += dim + n_output_filters

        self.x_len += n_output_filters * 10 + 10

        w = np.random.normal(0, 0.01, size=(batch_size, self.x_len))
        self.s = 0

        return w
        

    def get_new_params(self, batch_size=1):
        return {
            self.dim: self.x_len
        }

        
    def get_next_dict(self, n_bptt_steps, batch_size=1):
        x = np.zeros((n_bptt_steps, 1, self.batch_size,) + self.X.shape[1:]) 
        y = np.zeros((n_bptt_steps, 1, self.batch_size)) 

        for i in range(n_bptt_steps):
            if self.s + self.batch_size > self.X.shape[0]:
                self.s = 0
            pos_cur, pos_next = self.s, self.s + self.batch_size

            x[i] = self.X[None, pos_cur:pos_next]
            y[i] = self.Y[None, pos_cur:pos_next]

            self.s = pos_next


        return { 
            self.x: x,
            self.y: y,
        } 

