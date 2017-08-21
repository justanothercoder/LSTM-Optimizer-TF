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
    def __init__(self, num_units=20, num_layers=2,
            beta1=0.9, beta2=0.999, eps=1e-8,
            layer_norm=True, weight_norm=False,
            add_skip=False, residual=False,
            normalize_gradients=False, use_both=False, only_adam_features=False,
            clip_delta=2, rnn_type='lstm', learn_init=False):

        super(LSTMOptCell, self).__init__()

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_norm = weight_norm
        self.add_skip = add_skip
        self.residual = residual
        self.normalize_gradients = normalize_gradients
        self.use_both = use_both
        self.clip_delta = clip_delta
        self.learn_init = learn_init
        self.only_adam_features = only_adam_features

        def make_cell(num_units, residual):
            if rnn_type == 'gru':
                cell = GRUCell(num_units)
            elif layer_norm:
                cell = LayerNormBasicLSTMCell(num_units, layer_norm=True)
            else:
                cell = LSTMBlockCell(num_units)

            if residual:
                cell = ResidualWrapper(cell)
            return cell

        self.cell = MultiRNNCell([
            make_cell(num_units, residual and i > 0)
            for i in range(num_layers)
        ])

        state_size = (1, 1, 1, 1, 1, self.cell.state_size)

        if use_both:
            state_size = state_size + (1, 1)

        self._state_size = state_size


    def adam_update(self, g, m, v):
        b1 = self.beta1
        b2 = self.beta2

        new_m = b1 * m + (1 - b1) * g
        new_v = b2 * v + (1 - b2) * tf.square(g)
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


    def zero_state(self, batch_size):
        m = tf.zeros([batch_size])
        v = tf.zeros([batch_size])
        b1t = tf.ones([batch_size])
        b2t = tf.ones([batch_size])
        loglr = tf.random_uniform(shape=[batch_size], minval=np.log(1e-6), maxval=np.log(1e-2), seed=util.get_seed())
        lstm_state = self.cell.zero_state(batch_size, tf.float32)

        state = m, v, loglr, b1t, b2t, lstm_state

        if self.use_both:
            state = state + (tf.zeros([batch_size]), tf.zeros([batch_size]))
        
        return state


    @property
    def state_size(self):
        return self._state_size


    def __call__(self, g, state):
        m, v, loglr, b1t, b2t, cell_state, *args = state
        g_shape = tf.shape(g)
        
        if self.use_both:
            g_norm = normalize(g)
            g_norm = tf.reshape(g_norm, [-1])
            
            m_norm, v_norm = args
            m_norm, v_norm = self.adam_update(g_norm, m_norm, v_norm)

            g = tf.reshape(g, [-1])
            m, v = self.adam_update(g, m, v)

        elif self.normalize_gradients:
            g = normalize(g)
            g = tf.reshape(g, [-1])

            m, v = self.adam_update(g, m, v)
        else:
            g = tf.reshape(g, [-1])
            m, v = self.adam_update(g, m, v)

        b1t *= self.beta1
        b2t *= self.beta2

        a = tf.sqrt(1 - b2t) / (1 - b1t)
        s = self.adam_step(m, v, a)

        features = self.get_features(g, m, v, a)
        if self.use_both:
            features += self.get_features(g_norm, m_norm, v_norm, a)

        prep = tf.reshape(tf.stack(features, axis=-1), [-1, len(features)])

        if self.weight_norm:
            scope = tf.get_variable_scope()
            scope.set_custom_getter(custom_getter)
        
        last, cell_state = self.cell(prep, cell_state)
        last = tf.layers.dense(last, 2, use_bias=False, name='dense')

        d, loglr_add = tf.unstack(last, axis=1)

        loglr = tf.minimum(loglr + loglr_add, np.log(self.clip_delta))

        if self.add_skip:
            d += -s
        
        #if self.kwargs.get('adam_only', False):
        #    print("ADAMONLY")
        #    d = -s

        d = tf.reshape(d, g_shape)

        n_coords = tf.cast(g_shape[0], tf.float32)
        d = normalize(d, 1. / n_coords)

        lr = tf.exp(loglr, name='lr')
        lr = tf.clip_by_value(lr, 0, self.clip_delta)
        lr = tf.reshape(lr, g_shape)

        step = d * lr

        new_state = m, v, loglr, b1t, b2t, cell_state

        if self.use_both:
            new_state = new_state + (m_norm, v_norm)
        
        adam_normed = tf.nn.l2_normalize(tf.reshape(-s, g_shape), 1)
        step_normed = tf.nn.l2_normalize(d, 1)
                
        cos_step_adam = tf.reduce_sum(adam_normed * step_normed, axis=1)
        return step, new_state

