import pickle
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits, fetch_mldata
from sklearn.preprocessing import StandardScaler

from . import optimizee

HOME = '/srv/hd1/data/vyanush/'

def get_dataset(dataset_name):
    def standartize(X):
        return StandardScaler().fit_transform(X).astype(np.float32)

    if dataset_name == 'digits':
        dataset = load_digits(n_class=10)
        X, Y = dataset.data, dataset.target
        X = standartize(X)
        return X.reshape(-1, 8, 8, 1), np.array(Y)

    elif dataset_name == 'mnist':
        dataset = fetch_mldata('MNIST original', data_home=HOME)
        X, Y = dataset.data, dataset.target
        X = standartize(X)
        return X.reshape(-1, 28, 28, 1), np.array(Y)

    elif dataset_name == 'cifar-10':
        X = np.empty((10000, 3072))
        Y = np.empty((10000,))

        from pympler import asizeof
        print("Size in MB: ", asizeof.asizeof(X) / 1024**2)

        #for i in range(1, 2):
        #    with open(HOME + 'lstm_opt_tf/optimizees/' + 'cifar10/data_batch_{}'.format(i), 'rb') as f:
        #        data = pickle.load(f, encoding='bytes')
        #        X[i*10000:(i+1)*10000] = data[b'data']
        #        Y[i*10000:(i+1)*10000] = data[b'labels']
        
        with open(HOME + 'lstm_opt_tf/optimizees/' + 'cifar10/data_batch_1', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            X = data[b'data']
            Y = data[b'labels']

        X = standartize(X)
        return X.reshape(-1, 32, 32, 3), np.array(Y)


class ConvClassifier(optimizee.Optimizee):
    name = 'conv_classifier'

    def __init__(self, num_filters=20, num_layers=1, dataset_name='digits', activation='relu', filters_list=None, arch='custom'):
        self.X, self.Y = get_dataset(dataset_name)

        self.num_filters = num_filters
        self.num_layers = num_layers
        self.activation = getattr(tf.nn, activation)
        self.filter_width = 3
        self.arch = arch

        self.filters_list = filters_list or ( [self.X.shape[-1]] + [self.num_filters] * self.num_layers )

        self.x_len = 0
        self.counted_x_len = False


    def get_x_dim(self):
        return self.dim


    def build(self):
        with tf.variable_scope('conv_classifier'):
            self.dim = tf.placeholder(tf.int32, [], name='dim')
            self.x = tf.placeholder(tf.float32, [None, None, None, None, None, None], name='X') # n_bptt_steps * batch_size * data_size * num_features
            self.y = tf.placeholder(tf.int32, [None, None, None], name='y')


    def add_conv_params(self, x, n_input_filters, n_output_filters):
        dim = self.filter_width * self.filter_width * n_input_filters * n_output_filters

        W = x[0, self.s: self.s + dim]
        W = tf.reshape(W, [self.filter_width, self.filter_width, n_input_filters, n_output_filters])
        b = x[0, self.s + dim: self.s + dim + n_output_filters]

        self.s += dim + n_output_filters
        return W, b


    def conv(self, pred, x, n_input_filters, n_output_filters):
        W, b = self.add_conv_params(x, n_input_filters, n_output_filters)
            
        pred = tf.nn.conv2d(pred, W, strides=[1, 1, 1, 1], padding='SAME')
        pred = tf.nn.bias_add(pred, b)
        pred = self.activation(pred)

        return pred


    def pooling(self, pred, padding='SAME'):
        pred = tf.nn.max_pool(pred, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)
        return pred


    def add_fc(self, x, num_input_units, num_output_units):
        dim = num_input_units * num_output_units
        W = x[0, self.s: self.s + dim]
        W = tf.reshape(W, [num_input_units, num_output_units])
        b = x[0, self.s + dim: self.s + dim + num_output_units]

        self.s += dim + num_output_units
        return W, b


    def fc(self, pred, x, num_input_units, num_output_units):
        W, b = self.add_fc(x, num_input_units, num_output_units)
        pred = tf.matmul(pred, W) + tf.expand_dims(b, 0)
        return pred


    def loss(self, x, i):
        if self.arch == 'custom':
            return self.custom_arch(x, self.x[i][0], self.y[i][0])
        elif self.arch == 'vgg19':
            return self.vgg19_arch(x, self.x[i][0], self.y[i][0])


    def vgg19_arch(self, w, x, y):
        self.s = 0

        # self.x[i].shape == (data_size, w, h, n_filters)
        pred = self.conv(x   , w, self.X.shape[-1], 64)
        pred = self.conv(pred, w, 64, 64)
        pred = self.pooling(pred, padding='VALID')
        
        pred = self.conv(pred, w, 64, 128)
        pred = self.conv(pred, w, 128, 128)
        pred = self.pooling(pred, padding='VALID')
        
        pred = self.conv(pred, w, 128, 256)
        pred = self.conv(pred, w, 256, 256)
        pred = self.conv(pred, w, 256, 256)
        pred = self.conv(pred, w, 256, 256)
        pred = self.pooling(pred, padding='VALID')
        
        pred = self.conv(pred, w, 256, 512)
        pred = self.conv(pred, w, 512, 512)
        pred = self.conv(pred, w, 512, 512)
        pred = self.conv(pred, w, 512, 512)
        pred = self.pooling(pred, padding='VALID')
        
        pred = self.conv(pred, w, 512, 512)
        pred = self.conv(pred, w, 512, 512)
        pred = self.conv(pred, w, 512, 512)
        pred = self.conv(pred, w, 512, 512)
        pred = self.pooling(pred, padding='VALID')

        pred = tf.reduce_mean(pred, axis=(1, 2)) # global average pooling
        pred = self.fc(pred, w, 512, 10)

        if not self.counted_x_len:
            self.x_len = self.s
            self.counted_x_len = True

        #f = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y[i][0], logits=pred), axis=-1)
        f = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred)
        f = tf.expand_dims(f, 0)
    
        p = tf.argmax(tf.nn.softmax(pred), axis=-1)
        #acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(p, tf.int32), self.y[i][0]), tf.float32), axis=1)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(p, tf.int32), y), tf.float32), axis=-1)
        acc = tf.expand_dims(acc, 0)

        g = self.grad(w, f)
        return acc, g


    def custom_arch(self, w, x, y):
        self.s = 0

        dims = self.filters_list

        # self.x[i].shape == (data_size, w, h, n_filters)
        pred = x

        for i in range(1, len(dims)):
            n_input_filters = dims[i - 1]
            n_output_filters = dims[i]

            pred = self.conv(pred, w, n_input_filters, n_output_filters)
            pred = self.pooling(pred)

        pred = tf.reduce_mean(pred, axis=(1, 2)) # global average pooling
        pred = self.fc(pred, w, n_output_filters, 10)

        if not self.counted_x_len:
            self.x_len = self.s
            self.counted_x_len = True

        #f = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y[i][0], logits=pred), axis=-1)
        f = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred)
        f = tf.expand_dims(f, 0)
    
        p = tf.argmax(tf.nn.softmax(pred), axis=-1)
        #acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(p, tf.int32), self.y[i][0]), tf.float32), axis=1)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(p, tf.int32), y), tf.float32), axis=-1)
        acc = tf.expand_dims(acc, 0)

        g = self.grad(w, f)
        return acc, g


    def get_initial_x(self, batch_size=1):
        if self.arch == 'vgg19':
            self.batch_size = 64
        else:
            self.batch_size = np.random.randint(low=1, high=self.X.shape[0] // 4 + 1)
        print("Conv classifier; batch_size: ", self.batch_size)

        w = np.random.normal(0, 0.01, size=(batch_size, self.x_len))
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

