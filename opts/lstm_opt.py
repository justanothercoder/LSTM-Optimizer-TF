from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell, LayerNormBasicLSTMCell, ResidualWrapper, LSTMBlockCell, LSTMStateTuple

from . import basic_model
from . import lstm_utils
from .adam_opt import AdamOpt
import util


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


InitConfig = util.namedtuple_with_defaults(
    'InitConfig', [
        ('num_units', 20),
        ('num_layers', 2),
        ('beta1', 0.9),
        ('beta2', 0.999),
        ('eps', 1e-8),
        ('layer_norm', True),
        ('add_skip', False),
        ('clip_delta', 2),
        ('rnn_type', 'lstm'),
        ('residual', False),
        ('normalize_gradients', False),
        ('learn_init', False),
        ('use_both', False),
        ('weight_norm', False),
        ('only_adam_features', False)
])


class LSTMOpt(basic_model.BasicModel):
    def __init__(self, init_config, **kwargs):
        super(LSTMOpt, self).__init__(**kwargs)
        self.init_config = init_config

        states = ['m', 'v', 'b1t', 'b2t', 'loglr', 'lstm_state']
        if init_config.use_both:
            states += ['m_norm', 'v_norm']
        self.LSTMOptState = namedtuple('LSTMOptState', states)
        self.added_summary = False


    def build_pre(self):
        def make_cell(num_units, residual):
            if self.init_config.rnn_type == 'gru':
                print("GRU")
                cell = GRUCell(num_units)
            else:
                if self.init_config.layer_norm:
                    print("LSTM With layer norm")
                    cell = LayerNormBasicLSTMCell(num_units, layer_norm=True)
                else:
                    print("LSTM Without layer norm")
                    #cell = LSTMCell(num_units)
                    cell = LSTMBlockCell(num_units)

            if residual:
                cell = ResidualWrapper(cell)
            return cell

        self.lstm = MultiRNNCell([
                make_cell(self.init_config.num_units, self.init_config.residual and i > 0)
                for i in range(self.init_config.num_layers)
        ])


    def build_inputs(self):
        m = tf.placeholder(tf.float32, [None, None], name='m')
        v = tf.placeholder(tf.float32, [None, None], name='v')
        b1t = tf.placeholder(tf.float32, [None], name='b1t')
        b2t = tf.placeholder(tf.float32, [None], name='b2t')
        loglr = tf.placeholder(tf.float32, [None, None], name='loglr')

        if self.init_config.rnn_type == 'gru':
            lstm_state = tuple(
                tf.placeholder(tf.float32, [None, size]) # shape = (n_functions * n_coords, num_units)
                for size in self.lstm.state_size
            )
        else:
            lstm_state = tuple(
                LSTMStateTuple(
                    tf.placeholder(tf.float32, [None, size.c]),
                    tf.placeholder(tf.float32, [None, size.h])
                    ) # shape = (n_functions * n_coords, num_units)
                for size in self.lstm.state_size
            )

        state = (m, v, b1t, b2t, loglr, lstm_state)

        if self.init_config.use_both:
            m_norm = tf.placeholder(tf.float32, [None, None], name='m_norm')
            v_norm = tf.placeholder(tf.float32, [None, None], name='v_norm')
            state = state + (m_norm, v_norm)

        self.input_state = self.LSTMOptState(*state)
        return self.input_state

    
    def build_initial_state(self, x):
        m = tf.zeros(shape=tf.shape(x))
        v = tf.zeros(shape=tf.shape(x))
        b1t = tf.ones([tf.shape(x)[0]])
        b2t = tf.ones([tf.shape(x)[0]])
        loglr = tf.random_uniform(shape=tf.shape(x), minval=np.log(1e-6), maxval=np.log(1e-2), seed=util.get_seed())

        if self.init_config.learn_init:
            lstm_state = lstm_utils.get_initial_cell_state(self.lstm, lstm_utils.make_variable_state_initializer(), tf.size(x), tf.float32)
        else:
            lstm_state = self.lstm.zero_state(tf.size(x), tf.float32)

        state = (m, v, b1t, b2t, loglr, lstm_state)

        if self.init_config.use_both:
            m_norm = tf.zeros(shape=tf.shape(x))
            v_norm = tf.zeros(shape=tf.shape(x))
            state = state + (m_norm, v_norm)

        self.initial_state = self.LSTMOptState(*state)
        return self.initial_state


    def adam_prep(self, g, state):
        with tf.name_scope('adam_step'):
            b1 = self.init_config.beta1
            b2 = self.init_config.beta2

            m = b1 * state.m + (1 - b1) * g
            v = b2 * state.v + (1 - b2) * tf.square(g)

            b1t = state.b1t * b1
            b2t = state.b2t * b2

            a = tf.expand_dims(tf.sqrt(1 - b2t) / (1 - b1t), -1)
            s = a * m / (tf.sqrt(v) + self.init_config.eps)

            new_state = (m, v, b1t, b2t, state.loglr, state.lstm_state)

            if self.init_config.only_adam_features:
                features = [s]
            else:
                features = [g, tf.square(g), m, v, s]

            if self.init_config.use_both:
                g_norm = normalize(g)
                m_norm = b1 * state.m_norm + (1 - b1) * g_norm
                v_norm = b2 * state.v_norm + (1 - b2) * tf.square(g_norm)
                s_norm = a * m_norm / (tf.sqrt(v_norm) + self.init_config.eps)

                new_state = new_state + (m_norm, v_norm)

                if self.init_config.only_adam_features:
                    features += [s_norm]
                else:
                    features += [g_norm, tf.square(g_norm), m_norm, v_norm, s_norm]

            return self.LSTMOptState(*new_state), features, s


    def step(self, f, g, state):
        with tf.name_scope('lstm_step'):
            g_shape = tf.shape(g)

            new_state, features, s = self.adam_prep(g, state)
            prep = tf.reshape(tf.stack(features, axis=-1), [-1, len(features)])
            
            if self.init_config.weight_norm:
                scope = tf.get_variable_scope()
                scope.set_custom_getter(custom_getter)
            
            last, lstm_state = self.lstm(prep, state.lstm_state)

            if self.added_summary:
                tf.summary.histogram('lstm_activations', last)

            last = tf.layers.dense(last, 2, use_bias=False, name='dense')

            d, loglr_add = tf.unstack(last, axis=1)

            d = tf.reshape(d, g_shape)
            loglr_add = tf.reshape(loglr_add, g_shape)

            loglr = tf.minimum(state.loglr + loglr_add, np.log(self.init_config.clip_delta))
            n_coords = tf.cast(g_shape[1], tf.float32)

            if self.init_config.add_skip:
                d += -s
            
            if self.kwargs.get('adam_only', False):
                print("ADAMONLY")
                d = -s

            d = normalize(d, 1. / n_coords)

            lr = tf.exp(loglr, name='lr')
            lr = tf.clip_by_value(lr, 0, self.init_config.clip_delta)

            step = d * lr
            new_state = new_state._replace(lstm_state=lstm_state, loglr=loglr)

            if self.added_summary:
                tf.summary.histogram('loglr', loglr)
                tf.summary.scalar('max_loglr', tf.reduce_max(tf.reduce_mean(loglr, axis=0)))
                tf.summary.scalar('mean_loglr', tf.reduce_mean(loglr))
                self.added_summary = True

            #adam_normed = tf.nn.l2_normalize(-s, 1)
            #step_normed = tf.nn.l2_normalize(d, 1)
            #cos_ = tf.reduce_sum(adam_normed * step_normed, axis=1)

        return step, new_state

