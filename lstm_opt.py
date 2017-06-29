import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, LayerNormBasicLSTMCell

import basic_model


def normalize(d, gamma):
    return d / (gamma * tf.norm(d, axis=-1, keep_dims=True))


class LSTMOpt(basic_model.BasicModel):
    def __init__(self, optimizee, num_units=20, num_layers=2, beta1=0.9, beta2=0.999, p_drop=0.0, layer_norm=True, stop_grad=True, add_skip=False, **kwargs):
        super(LSTMOpt, self).__init__(optimizee, **kwargs)

        self.num_units = num_units
        self.num_layers = num_layers

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1e-8

        self.p_drop = p_drop
        self.stop_grad = stop_grad
        self.add_skip = add_skip
        self.layer_norm = layer_norm


    def _build_pre(self):
        self.lstm = MultiRNNCell([LSTMCell(self.num_units) for _ in range(self.num_layers)])
        #self.lstm = MultiRNNCell([
        #    LayerNormBasicLSTMCell(self.num_units, layer_norm=self.layer_norm, dropout_keep_prob=1.0 - self.p_drop) 
        #    for _ in range(self.num_layers)
        #])
        

    def _build_input(self):
        self.x = tf.placeholder(tf.float32, [None, None], name='x') # shape = (n_functions, n_coords)
        self.m = tf.placeholder(tf.float32, [None, None], name='m')
        self.v = tf.placeholder(tf.float32, [None, None], name='v')
        self.b1t = tf.placeholder(tf.float32, [None], name='b1t')
        self.b2t = tf.placeholder(tf.float32, [None], name='b2t')
        self.loglr = tf.placeholder(tf.float32, [None, None], name='loglr')

        self.lstm_state = tuple(
            (tf.placeholder(tf.float32, [None, size.c]), tf.placeholder(tf.float32, [None, size.h])) # shape = (n_functions * n_coords, num_units)
            for size in self.lstm.state_size
        )
        self.input_state = [self.b1t, self.b2t, self.x, self.m, self.v, self.lstm_state, self.loglr]

    
    def _build_initial_state(self):
        x = self.x
        m = tf.zeros(shape=tf.shape(x))
        v = tf.zeros(shape=tf.shape(x))
        b1t = tf.ones([tf.shape(x)[0]])
        b2t = tf.ones([tf.shape(x)[0]])
        lstm_state = self.lstm.zero_state(tf.size(x), tf.float32)
        
        #loglr = tf.zeros(shape=tf.shape(x))
        loglr = tf.random_uniform(shape=tf.shape(x), minval=np.log(1e-6), maxval=np.log(1e-2))

        self.initial_state = [b1t, b2t, x, m, v, lstm_state, loglr]


    def _iter(self, f, i, state):
        b1t, b2t, x, m, v, lstm_state, loglr = state

        fx, g, g_norm = self._fg(f, x, i)
        x_shape = tf.shape(x)

        if self.stop_grad:
            g = tf.stop_gradient(g)

        m = self.beta1 * m + (1 - self.beta1) * g
        v = self.beta2 * v + (1 - self.beta2) * (g ** 2)

        b1t *= self.beta1
        b2t *= self.beta2

        a = tf.expand_dims(tf.sqrt(1 - b2t) / (1 - b1t), -1)
        s = a * m / (tf.sqrt(v) + self.eps)

        features = [g, (g ** 2), m, v, s]

        prep = tf.reshape(tf.stack(features, axis=-1), [-1, len(features)])
        last, lstm_state = self.lstm(prep, lstm_state)

        last = tf.layers.dense(last, 2, use_bias=False, name=self.loop_scope.name) #, kernel_initializer=tf.truncated_normal_initializer(0.1))
        #d, loglr = tf.unstack(last, axis=1)

        d, loglr_add = tf.unstack(last, axis=1)

        d = tf.reshape(d, x_shape)
        loglr_add = tf.reshape(loglr_add, x_shape)

        loglr += loglr_add

        n_coords = tf.cast(x_shape[0], tf.float32)
        
        #d = d / (tf.norm(d, axis=-1, keep_dims=True) * n_coords)
        d = normalize(d, 1. / n_coords)

        if self.add_skip:
            d += -s / (tf.norm(s, axis=-1, keep_dims=True) * n_coords)
            #d = s

        x += d * tf.exp(loglr)

        return [b1t, b2t, x, m, v, lstm_state, loglr], fx, g_norm
