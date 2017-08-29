from collections import namedtuple
import tensorflow as tf
from .basic_model import BasicModel
from .lstm_opt import normalize
from . import config


class InitConfig(config.Config):
    _default = {
        'lr': 1e-4,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8,
        'enable_reduce': False,
        'factor': 0.5,
        'patience_max': 10,
        'epsilon': 1e-4,
        'normalize': False
    }


class AdamOpt(BasicModel):
    def __init__(self, init_config, **kwargs):
        super(AdamOpt, self).__init__(init_config, **kwargs)

        fields = ['m', 'v', 'b1t', 'b2t']
        if init_config.enable_reduce:
            fields += ['lr', 'f_best', 'f_ma', 'patience', 'sid']

        self.AdamOptState = namedtuple('AdamOptState', fields)
    
    
    def init_state(self, x):
        m = tf.zeros(tf.shape(x))
        v = tf.zeros(tf.shape(x))
        b1t = tf.ones([tf.shape(x)[0]])
        b2t = tf.ones([tf.shape(x)[0]])

        state = (m, v, b1t, b2t)

        if self.init_config.enable_reduce:
            lr = self.init_config.lr * tf.ones([tf.shape(x)[0]])
            f_best = tf.zeros([tf.shape(x)[0]])
            f_ma = tf.zeros([tf.shape(x)[0]])
            patience = tf.zeros([tf.shape(x)[0]])
            sid = tf.zeros([])

            state = state + (lr, f_best, f_ma, patience, sid)

        initial_state = self.AdamOptState(*state)
        return initial_state


    def build_pre(self):
        pass


    def reduce_lr_on_plateau(self, state, f):

        def true_fn():
            return f, f, state.lr, state.patience


        def false_fn():
            new_f_ma = 0.95 * state['f_ma'] + 0.05 * f
            new_f_best = tf.maximum(new_f_ma, state['f_best'])

            # (old - new) / old < eps

            patience = tf.where(
                        tf.less((state.f_best - new_f_best) / state.f_best, self.epsilon),
                        state.patience + 1,
                        state.patience
                    )

            new_lr = tf.where(
                        tf.equal(patience, self.init_config.patience_max),
                        state.lr * self.factor,
                        state.lr
                    )

            patience = tf.where(
                        tf.equal(patience, self.init_config.patience_max),
                        tf.zeros_like(patience),
                        patience
                    )
            return new_f_ma, new_f_best, new_lr, patience

        new_f_ma, new_f_best, new_lr, patience = tf.cond(tf.equal(state.sid, 0), true_fn, false_fn)
        return state._replace(lr=new_lr, f_best=new_f_best, f_ma=new_f_ma, patience=patience, sid=state.sid + 1)


    def step(self, g, state):
        if self.init_config.normalize:
            g = normalize(g)

        b1 = self.init_config.beta1
        b2 = self.init_config.beta2

        m = b1 * state.m + (1 - b1) * g
        v = b2 * state.v + (1 - b2) * tf.square(g)

        b1t = b1 * state.b1t
        b2t = b2 * state.b2t

        a = tf.expand_dims(tf.sqrt(1 - b2t) / (1 - b1t), -1)

        if self.init_config.enable_reduce:
            lr_state = self.reduce_lr_on_plateau(state, f)

        if not self.init_config.enable_reduce:
            s = self.init_config.lr * a * m / (tf.sqrt(v) + self.init_config.eps)
        else:
            s = lr_state.lr * a * m / (tf.sqrt(v) + self.init_config.eps)

        step = -s
        state = (m, v, b1t, b2t)

        if self.init_config.enable_reduce:
            state += (lr_state.lr, lr_state.f_best, lr_state.f_ma, lr_state.patience, lr_state.sid)

        return step, self.AdamOptState(*state)
    
    
    def restore(self, eid):
        pass


    def save(self, eid):
        pass
