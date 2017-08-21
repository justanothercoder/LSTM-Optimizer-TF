import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell, LayerNormBasicLSTMCell, ResidualWrapper, LSTMBlockCell, LSTMStateTuple

from . import basic_model
from . import lstm_utils
from .adam_opt import AdamOpt


def custom_getter(getter, name, *args, **kwargs):
    shape = kwargs['shape']
    if name.find('bias') == -1 and shape is not None and kwargs.get('trainable', False):
        del kwargs['shape']
        initializer = kwargs.pop('initializer')
        g = getter(name + '_norm', shape=(shape[:-1] + [1]), initializer=tf.ones_initializer(), *args, **kwargs)
        v = getter(name + '_direction', shape=shape, initializer=initializer, *args, **kwargs)

        normed_v = v / tf.norm(v, axis=-1, keep_dims=True)
        return g * normed_v
    else:
        return getter(name, *args, **kwargs)


def normalize(d, gamma=1.0, eps=1e-8):
    return d / (gamma * tf.norm(d, axis=-1, keep_dims=True) + eps)


class LSTMOpt(basic_model.BasicModel):
    def __init__(self,
        num_units=20, num_layers=2,
        beta1=0.9, beta2=0.999, eps=1e-8,
        layer_norm=True, add_skip=False, clip_delta=100,
        rnn_type='lstm', residual=False,
        normalize_gradients=False,
        learn_init=False, use_both=False,
        weight_norm=False, only_adam_features=False,
        **kwargs):

        super(LSTMOpt, self).__init__(**kwargs)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.num_units = num_units
        self.num_layers = num_layers

        self.add_skip = add_skip
        self.residual = residual

        self.layer_norm = layer_norm
        self.weight_norm = weight_norm

        self.clip_delta = clip_delta
        self.rnn_type = rnn_type
        self.normalize_gradients = normalize_gradients
        self.learn_init = learn_init
        self.use_both = use_both

        self.only_adam_features = only_adam_features


    def build_pre(self):
        def make_cell(num_units, residual):
            if self.rnn_type == 'gru':
                print("GRU")
                cell = GRUCell(num_units)
            else:
                if self.layer_norm:
                    print("LSTM With layer norm")
                    cell = LayerNormBasicLSTMCell(num_units, layer_norm=True)
                else:
                    print("LSTM Without layer norm")
                    #cell = LSTMCell(num_units)
                    cell = LSTMBlockCell(num_units)

            if residual:
                cell = ResidualWrapper(cell)
            return cell

        self.lstm = MultiRNNCell([make_cell(self.num_units, self.residual and i > 0) for i in range(self.num_layers)])


    def build_inputs(self):
        self.m = tf.placeholder(tf.float32, [None, None], name='m')
        self.v = tf.placeholder(tf.float32, [None, None], name='v')
        self.b1t = tf.placeholder(tf.float32, [None], name='b1t')
        self.b2t = tf.placeholder(tf.float32, [None], name='b2t')
        self.loglr = tf.placeholder(tf.float32, [None, None], name='loglr')

        if self.rnn_type == 'gru':
            self.lstm_state = tuple(
                tf.placeholder(tf.float32, [None, size]) # shape = (n_functions * n_coords, num_units)
                for size in self.lstm.state_size
            )
        else:
            self.lstm_state = tuple(
                #(tf.placeholder(tf.float32, [None, size.c]), tf.placeholder(tf.float32, [None, size.h])) # shape = (n_functions * n_coords, num_units)
                LSTMStateTuple(tf.placeholder(tf.float32, [None, size.c]), tf.placeholder(tf.float32, [None, size.h])) # shape = (n_functions * n_coords, num_units)
                for size in self.lstm.state_size
            )

        self.input_state = {
            'b1t': self.b1t,
            'b2t': self.b2t,
            'm': self.m,
            'v': self.v,
            'lstm_state': self.lstm_state,
            'loglr': self.loglr
        }

        if self.use_both:
            self.m_norm = tf.placeholder(tf.float32, [None, None], name='m')
            self.v_norm = tf.placeholder(tf.float32, [None, None], name='v')
            self.input_state.update(m_norm=self.m_norm, v_norm=self.v_norm)

        return self.input_state

    
    def build_initial_state(self, x):
        m = tf.zeros(shape=tf.shape(x))
        v = tf.zeros(shape=tf.shape(x))
        b1t = tf.ones([tf.shape(x)[0]])
        b2t = tf.ones([tf.shape(x)[0]])

        if self.learn_init:
            lstm_state = lstm_utils.get_initial_cell_state(self.lstm, lstm_utils.make_variable_state_initializer(), tf.size(x), tf.float32)
        else:
            lstm_state = self.lstm.zero_state(tf.size(x), tf.float32)
        
        #loglr = tf.zeros(shape=tf.shape(x))
        loglr = tf.random_uniform(shape=tf.shape(x), minval=np.log(1e-6), maxval=np.log(1e-2))

        self.initial_state = {
            'b1t': b1t,
            'b2t': b2t,
            'm': m,
            'v': v,
            'lstm_state': lstm_state,
            'loglr': loglr
        }

        if self.use_both:
            m_norm = tf.zeros(shape=tf.shape(x))
            v_norm = tf.zeros(shape=tf.shape(x))
            self.initial_state.update(m_norm=m_norm, v_norm=v_norm)

        return self.initial_state


    def adam_update(self, g, m, v):
        new_m = self.beta1 * m + (1 - self.beta1) * g
        new_v = self.beta2 * v + (1 - self.beta2) * tf.square(g)
        return new_m, new_v


    def adam_step(self, m, v, a):
        s = a * m / (tf.sqrt(v, name='v_sqrt') + self.eps)
        return s


    def get_features(self, g, m, v, a):
        g2 = tf.square(g)
        s = self.adam_step(m, v, a)

        if self.only_adam_features:
            features = [s]
        else:
            features = [g, g2, m, v, s]

        return features


    def step(self, f, g, state):
        b1t, b2t, m, v, lstm_state, loglr = tuple(state[name] for name in ['b1t', 'b2t', 'm', 'v', 'lstm_state', 'loglr'])

        if self.use_both:
            m_norm = state['m_norm']
            v_norm = state['v_norm']

        g_shape = tf.shape(g)

        if self.normalize_gradients:
            g = normalize(g)

        m, v = self.adam_update(g, m, v)

        if self.use_both:
            g_norm = normalize(g)
            m_norm, v_norm = self.adam_update(g_norm, m_norm, v_norm)

        b1t *= self.beta1
        b2t *= self.beta2

        a = tf.expand_dims(tf.sqrt(1 - b2t) / (1 - b1t), -1)
        s = self.adam_step(m, v, a)

        features = self.get_features(g, m, v, a)
        if self.use_both:
            features += self.get_features(g_norm, m_norm, v_norm, a)
        
        prep = tf.reshape(tf.stack(features, axis=-1), [-1, len(features)])
        
        if self.weight_norm:
            scope = tf.get_variable_scope()
            scope.set_custom_getter(custom_getter)
        
        last, lstm_state = self.lstm(prep, lstm_state)
        last = tf.layers.dense(last, 2, use_bias=False)

        d, loglr_add = tf.unstack(last, axis=1)

        d = tf.reshape(d, g_shape)
        loglr_add = tf.reshape(loglr_add, g_shape)

        loglr = tf.minimum(loglr + loglr_add, np.log(self.clip_delta))
        n_coords = tf.cast(g_shape[0], tf.float32)

        if self.add_skip:
            d += -s
        
        if self.kwargs.get('adam_only', False):
            print("ADAMONLY")
            d = -s

        d = normalize(d, 1. / n_coords)

        lr = tf.exp(loglr, name='lr')
        lr = tf.clip_by_value(lr, 0, self.clip_delta)

        step = d * lr

        new_state = dict(b1t=b1t, b2t=b2t, m=m, v=v, lstm_state=lstm_state, loglr=loglr)
        if self.use_both:
            new_state.update(m_norm=m_norm, v_norm=v_norm)

        adam_normed = tf.nn.l2_normalize(-s, 1)
        step_normed = tf.nn.l2_normalize(d, 1)
                
        cos_ = tf.reduce_sum(adam_normed * step_normed, axis=1)

        return {
            'step': step,
            'state': new_state,
            'cos_step_adam': cos_
        }
