import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell

import basic_model


class LSTMOpt(basic_model.BasicModel):
    def __init__(self, optimizee, num_units=20, num_layers=2, beta1=0.9, beta2=0.999, **kwargs):
        super(LSTMOpt, self).__init__(optimizee, **kwargs)

        self.num_units = num_units
        self.num_layers = num_layers

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = 1e-8


    def _build_pre(self):
        self.lstm = MultiRNNCell([LSTMCell(self.num_units) for _ in range(self.num_layers)])
        

    def _build_input(self):
        self.x = tf.placeholder(tf.float32, [None], name='x')
        self.m = tf.placeholder(tf.float32, [None], name='m')
        self.v = tf.placeholder(tf.float32, [None], name='v')
        self.b1t = tf.placeholder(tf.float32, [], name='b1t')
        self.b2t = tf.placeholder(tf.float32, [], name='b2t')
        self.sid = tf.placeholder(tf.int32, [], name='t')
        self.loglr = tf.placeholder(tf.float32, [None], name='loglr')

        self.lstm_state = tuple(
            (tf.placeholder(tf.float32, [None, size.c]), tf.placeholder(tf.float32, [None, size.h]))
            for size in self.lstm.state_size
        )
        self.input_state = [self.sid, self.b1t, self.b2t, self.x, self.m, self.v, self.lstm_state, self.loglr]

    
    def _build_initial_state(self):
        x = self.x
        m = tf.zeros(shape=tf.shape(x))
        v = tf.zeros(shape=tf.shape(x))
        b1t = tf.ones([])
        b2t = tf.ones([])
        lstm_state = self.lstm.zero_state(tf.size(x), tf.float32)
        loglr = tf.zeros(shape=tf.shape(x))

        self.initial_state = [tf.zeros([]), b1t, b2t, x, m, v, lstm_state, loglr]


    def _iter(self, f, i, state):
        sid, b1t, b2t, x, m, v, lstm_state, loglr = state

        fx, g = self._fg(f, x, i)
        g = tf.stop_gradient(g)

        g_norm = tf.reduce_sum(g**2)

        m = self.beta1 * m + (1 - self.beta1) * g
        v = self.beta2 * v + (1 - self.beta2) * (g ** 2)

        b1t *= self.beta1
        b2t *= self.beta2

        a = tf.sqrt(1 - b2t) / (1 - b1t)
        s = a * m / (tf.sqrt(v) + self.eps)

        prep = tf.stack([g, (g ** 2), m, v, s], axis=1)
        last, lstm_state = self.lstm(prep, lstm_state)

        last = tf.layers.dense(last, 2, use_bias=False, name=self.loop_scope.name) #, kernel_initializer=tf.truncated_normal_initializer(0.1))
        d, loglr_add = tf.unstack(last, axis=1)

        loglr += loglr_add
        
        d = d / (tf.cast(tf.shape(x)[0], tf.float32) * tf.sqrt(tf.reduce_sum(d ** 2)))
        x += d * tf.exp(loglr)

        return [sid + 1, b1t, b2t, x, m, v, lstm_state, loglr], fx, g_norm
