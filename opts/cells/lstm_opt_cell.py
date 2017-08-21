from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell, LayerNormBasicLSTMCell, ResidualWrapper, LSTMBlockCell, LSTMStateTuple

from . import opt_cell
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


class LSTMOptCell(opt_cell.OptCell):
    def __init__(self, init_config):
        super(LSTMOptCell, self).__init__()
        self.init_config = init_config

        def make_cell(num_units, residual):
            if init_config.rnn_type == 'gru':
                cell = GRUCell(num_units)
            elif init_config.layer_norm:
                cell = LayerNormBasicLSTMCell(num_units, layer_norm=True)
            else:
                cell = LSTMBlockCell(num_units)

            if residual:
                cell = ResidualWrapper(cell)
            return cell

        self.cell = MultiRNNCell([
            make_cell(init_config.num_units, init_config.residual and i > 0)
            for i in range(init_config.num_layers)
        ])

        state_size = (1, 1, 1, 1, 1, self.cell.state_size)

        if init_config.use_both:
            state_size = state_size + (1, 1)

        self._state_size = state_size
        
        states = ['m', 'v', 'b1t', 'b2t', 'loglr', 'lstm_state']
        if init_config.use_both:
            states += ['m_norm', 'v_norm']
        self.LSTMOptState = namedtuple('LSTMOptState', states)


    def adam_prep(self, g, state):
        b1 = self.init_config.beta1
        b2 = self.init_config.beta2

        m = b1 * state.m + (1 - b1) * g
        v = b2 * state.v + (1 - b2) * tf.square(g)

        b1t = state.b1t * b1
        b2t = state.b2t * b2

        a = tf.sqrt(1 - b2t) / (1 - b1t)
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


    def get_features(self, g, m, v, a):
        g2 = tf.square(g)
        s = self.adam_step(m, v, a)

        if self.only_adam_features:
            features = [s]
        else:
            features = [g, g2, m, v, s]

        return features


    def zero_state(self, batch_size):
        m = tf.zeros([batch_size])
        v = tf.zeros([batch_size])
        b1t = tf.ones([batch_size])
        b2t = tf.ones([batch_size])
        loglr = tf.random_uniform(shape=[batch_size], minval=np.log(1e-6), maxval=np.log(1e-2), seed=util.get_seed())
        lstm_state = self.cell.zero_state(batch_size, tf.float32)

        state = m, v, b1t, b2t, loglr, lstm_state

        if self.init_config.use_both:
            state = state + (tf.zeros([batch_size]), tf.zeros([batch_size]))
        
        return self.LSTMOptState(*state)


    @property
    def state_size(self):
        return self._state_size


    def __call__(self, g, state):
        g_shape = tf.shape(g)
        g = tf.reshape(g, [-1])
        
        new_state, features, s = self.adam_prep(g, state)
        prep = tf.reshape(tf.stack(features, axis=-1), [-1, len(features)])

        if self.init_config.weight_norm:
            scope = tf.get_variable_scope()
            scope.set_custom_getter(custom_getter)
        
        last, cell_state = self.cell(prep, state.lstm_state)
        last = tf.layers.dense(last, 2, use_bias=False, name='dense')

        d, loglr_add = tf.unstack(last, axis=1)

        loglr = tf.minimum(state.loglr + loglr_add, np.log(self.init_config.clip_delta))

        if self.init_config.add_skip:
            d += -s
        
        #if self.kwargs.get('adam_only', False):
        #    print("ADAMONLY")
        #    d = -s

        d = tf.reshape(d, g_shape)

        n_coords = tf.cast(g_shape[0], tf.float32)
        d = normalize(d, 1. / n_coords)

        lr = tf.exp(loglr, name='lr')
        lr = tf.clip_by_value(lr, 0, self.init_config.clip_delta)
        lr = tf.reshape(lr, g_shape)

        step = d * lr
        new_state = new_state._replace(loglr=loglr, lstm_state=cell_state)

        #adam_normed = tf.nn.l2_normalize(tf.reshape(-s, g_shape), 1)
        #step_normed = tf.nn.l2_normalize(d, 1)
        #        
        #cos_step_adam = tf.reduce_sum(adam_normed * step_normed, axis=1)
        return step, new_state

