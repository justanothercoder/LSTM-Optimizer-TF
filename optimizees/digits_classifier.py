import numpy as np
import tensorflow as tf
from . import optimizee
from sklearn.datasets import load_digits


class DIGITSClassifier(optimizee.Optimizee):
    name = 'digits_classifier'

    def __init__(self, num_units=20, num_layers=1):
        digits = load_digits(n_class=10)
        self.X, self.Y = digits.data, digits.target
        self.num_units = num_units
        self.num_layers = num_layers


    def get_x_dim(self):
        return self.dim


    def build(self):
        with tf.variable_scope('digits_classifier'):
            self.dim = tf.placeholder(tf.int32, [], name='dim')
            self.x = tf.placeholder(tf.float32, [None, None, None, None], name='X') # n_bptt_steps * batch_size * data_size * num_features
            self.y = tf.placeholder(tf.int32, [None, None, None], name='y')

    
    def loss(self, x, i):
        batch_size = tf.shape(x)[0]

        weights = []
        n_inputs = 64

        fs = []
        s = 0

        dims = [64] + [self.num_units] * self.num_layers + [10]

        # self.x[i].shape == (batch_size, data_size, n_inputs)
        pred = tf.transpose(self.x[i], perm=[0, 2, 1])

        for i in range(1, len(dims)):
            n_inputs = dims[i - 1]
            n_outputs = dims[i]
            dim = n_inputs * n_outputs

            W = tf.slice(x, [0, s], [batch_size, dim])
            W = tf.reshape(W, [batch_size, n_outputs, n_inputs])
            b = tf.slice(x, [0, s + dim], [batch_size, n_outputs])
            b = tf.expand_dims(b, axis=-1)

            pred = tf.matmul(W, pred) + b

            if i + 1 < len(dims):
                pred = tf.nn.sigmoid(pred)

            s += (n_inputs + 1) * n_outputs

        pred = tf.transpose(pred, perm=[0, 2, 1]) # shape = (batch_size, data_size, n_classes)
        f = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y[i], logits=pred), axis=-1)
    
        p = tf.argmax(tf.nn.softmax(pred), axis=-1)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(p, tf.int32), self.y[i]), tf.float32), axis=1)

        g = self.grad(x, f)
        #return f, g
        return acc, g


    def get_initial_x(self, batch_size=1):
        self.batch_size = np.random.randint(low=1, high=self.X.shape[1] + 1)

        self.x_len = 0
    
        dims = [64] + [self.num_units] * self.num_layers + [10]
        for i in range(1, len(dims)):
            n_inputs = dims[i - 1]
            n_outputs = dims[i]
            self.x_len += (n_inputs + 1) * n_outputs

        w = np.random.normal(0, 0.01, size=(batch_size, self.x_len))
        self.s = 0

        return w
        

    def get_new_params(self, batch_size=1):
        return {
            self.dim: self.x_len
        }

        
    def get_next_dict(self, n_bptt_steps, batch_size=1):
        x = np.zeros((n_bptt_steps, batch_size, self.batch_size, self.X.shape[1])) 
        y = np.zeros((n_bptt_steps, batch_size, self.batch_size)) 

        for i in range(n_bptt_steps):
            if self.s + self.batch_size > self.X.shape[0]:
                self.s = 0
            pos_cur, pos_next = self.s, self.s + self.batch_size

            x[i] = np.tile(self.X[pos_cur:pos_next], (batch_size, 1))
            y[i] = np.tile(self.Y[pos_cur:pos_next], (batch_size, 1))

            self.s = pos_next

        return { 
            self.x: x,
            self.y: y,
        } 

