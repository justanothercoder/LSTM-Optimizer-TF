from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, LayerNormBasicLSTMCell, ResidualWrapper, LSTMBlockCell

from . import basic_model
from . import lstm_utils
from . import config
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


class InitConfig(config.Config):
    _default = {
        'num_units': 20,
        'num_layers': 2,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8,
        'layer_norm': True,
        'add_skip': False,
        'clip_delta': 2,
        'rnn_type': 'lstm',
        'residual': False,
        'normalize_gradients': False,
        'learn_init': False,
        'use_both': False,
        'weight_norm': False,
        'only_adam_features': False,
        'adam_only': False,
        'average_steps': False,
    }
    

class LSTMOpt(basic_model.BasicModel):
    def build_pre(self):
        def make_cell(num_units, residual):
            if self.init_config.rnn_type == 'gru':
                cell = GRUCell(num_units)
            else:
                if self.init_config.layer_norm:
                    cell = LayerNormBasicLSTMCell(num_units, layer_norm=True)
                else:
                    cell = LSTMBlockCell(num_units)

            if residual:
                cell = ResidualWrapper(cell)
            return cell

        self.cell = MultiRNNCell([
            make_cell(self.init_config.num_units, self.init_config.residual and i > 0)
            for i in range(self.init_config.num_layers)
        ])
        
        fields = ['m', 'v', 'b1t', 'b2t', 'loglr', 'cell_state']
        if self.init_config.use_both:
            fields += ['m_norm', 'v_norm']

        if self.init_config.average_steps:
            fields += ['old_step']

        self.LSTMOptState = namedtuple('LSTMOptState', fields)


    
    def init_state(self, x):
        m = tf.zeros(shape=tf.shape(x))
        v = tf.zeros(shape=tf.shape(x))
        b1t = tf.ones([tf.shape(x)[0]])
        b2t = tf.ones([tf.shape(x)[0]])

        b1t.set_shape([x.get_shape().as_list()[0]])
        b2t.set_shape([x.get_shape().as_list()[0]])

        loglr = tf.random_uniform(shape=tf.shape(x), minval=np.log(1e-6), maxval=np.log(1e-2), seed=util.get_seed())

        if self.init_config.learn_init:
            cell_state = lstm_utils.get_initial_cell_state(self.cell, lstm_utils.make_variable_state_initializer(), tf.size(x), tf.float32)
        else:
            cell_state = self.cell.zero_state(tf.size(x), tf.float32)

        state = (m, v, b1t, b2t, loglr, cell_state)

        if self.init_config.use_both:
            m_norm = tf.zeros(shape=tf.shape(x))
            v_norm = tf.zeros(shape=tf.shape(x))
            state = state + (m_norm, v_norm)

        if self.init_config.average_steps:
            old_step = tf.zeros(shape=tf.shape(x))
            state = state + (old_step,)

        return self.LSTMOptState(*state)


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

            new_state = state._replace(m=m, v=v, b1t=b1t)

            if self.init_config.only_adam_features:
                features = [s]
            else:
                features = [g, tf.square(g), m, v, s]

            if self.init_config.use_both:
                g_norm = normalize(g)
                m_norm = b1 * state.m_norm + (1 - b1) * g_norm
                v_norm = b2 * state.v_norm + (1 - b2) * tf.square(g_norm)
                s_norm = a * m_norm / (tf.sqrt(v_norm) + self.init_config.eps)

                new_state = new_state._replace(m_norm=m_norm, v_norm=v_norm)

                if self.init_config.only_adam_features:
                    features += [s_norm]
                else:
                    features += [g_norm, tf.square(g_norm), m_norm, v_norm, s_norm]

            return new_state, features, s


    def step(self, g, state):
        with tf.name_scope('lstm_step'):
            g_shape = tf.shape(g)

            new_state, features, s = self.adam_prep(g, state)
            prep = tf.reshape(tf.stack(features, axis=-1), [-1, len(features)])
            
            if self.init_config.weight_norm:
                scope = tf.get_variable_scope()
                scope.set_custom_getter(custom_getter)
            
            last, cell_state = self.cell(prep, state.cell_state)
    
            if self.init_config.adam_only:
                last = tf.layers.dense(last, 1, use_bias=False, name='dense')
                loglr_add = tf.unstack(last, axis=1)
                d = -s
            else:
                last = tf.layers.dense(last, 2, use_bias=False, name='dense')
                d, loglr_add = tf.unstack(last, axis=1)

                d = tf.reshape(d, g_shape)
            
                if self.init_config.add_skip:
                    d += -s
                
            loglr_add = tf.reshape(loglr_add, g_shape)
            loglr = tf.minimum(state.loglr + loglr_add, np.log(self.init_config.clip_delta))
            n_coords = tf.cast(g_shape[1], tf.float32)

            #d = normalize(d, 1. / n_coords)
            d = normalize(d, 1.)

            lr = tf.exp(loglr, name='lr')
            lr = tf.clip_by_value(lr, 0, self.init_config.clip_delta)

            step = d * lr
            if self.init_config.average_steps:
                gamma_logits = tf.get_variable('gamma_logits', shape=[], dtype=tf.float32, initializer=tf.zeros_initializer())
                gamma = tf.sigmoid(gamma_logits)
                step = gamma * state.old_step + step
                new_state = new_state._replace(old_step=step)

            new_state = new_state._replace(cell_state=cell_state, loglr=loglr)

        return step, new_state
