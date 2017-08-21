import numpy as np
import tensorflow as tf

from . import opt_cell


class AdamOptCell(opt_cell.OptCell):
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8, enable_reduce=False, factor=0.5, patience_max=10, epsilon=1e-4, **kwargs):

        super(AdamOptCell, self).__init__()

        self.lr_init = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.factor = factor
        self.patience_max = patience_max
        self.enable_reduce = enable_reduce
        self.epsilon = epsilon
    
        state_size = (1, 1, 1, 1)

        if enable_reduce:
            state_size = state_size + (1, 1, 1, 1, 1)

        self._state_size = state_size

    def zero_state(self, batch_size):
        m = tf.zeros([batch_size])
        v = tf.zeros([batch_size])
        b1t = tf.ones([batch_size])
        b2t = tf.ones([batch_size])

        state = m, v, b1t, b2t

        if self.enable_reduce:
            lr = self.lr_init * tf.ones([batch_size])
            f_best = tf.zeros([batch_size])
            f_ma = tf.zeros([batch_size])
            patience = tf.zeros([batch_size])

            sid = tf.zeros([batch_size])
            state = state + (lr, f_best, f_ma, patience, sid)

        return state
        

    @property
    def state_size(self):
        return self._state_size


    def __call__(self, g, state):
        m, v, b1t, b2t, *args = state

        g_shape = tf.shape(g)
        g = tf.reshape(g, [-1])

        m = self.beta1 * m + (1 - self.beta1) * g
        v = self.beta2 * v + (1 - self.beta2) * tf.square(g)

        b1t *= self.beta1
        b2t *= self.beta2

        a = tf.sqrt(1 - b2t) / (1 - b1t)

        if self.enable_reduce:
            lr_state = self.reduce_lr_on_plateau(args, f)

        if not self.enable_reduce:
            s = self.lr_init * a * m / (tf.sqrt(v) + self.eps)
        else:
            lr = lr_state[0]
            s = lr * a * m / (tf.sqrt(v) + self.eps)

        new_state = m, v, b1t, b2t
        if self.enable_reduce:
            new_state = new_state + self.lr_update(lr_state)

        return -s, new_state


    def reduce_lr_on_plateau(self, state, f):
        lr, f_best, f_ma, patience, sid = state

        def true_fn():
            return f, f, lr, patience

        def false_fn():
            new_f_ma = 0.95 * f_ma + 0.05 * f
            new_f_best = tf.maximum(new_f_ma, f_best)

            new_patience = tf.where(
                        #tf.greater(new_f_best, state['f_best']),
                        tf.greater(new_f_best, f_best - self.epsilon),
                        patience + 1,
                        patience
                    )

            new_lr = tf.where(
                        tf.equal(new_patience, self.patience_max),
                        lr * self.factor,
                        lr
                    )

            new_patience = tf.where(
                        tf.equal(new_patience, self.patience_max),
                        tf.zeros_like(new_patience),
                        new_patience
                    )
            return new_f_ma, new_f_best, new_lr, new_patience

        new_f_ma, new_f_best, new_lr, patience = tf.cond(tf.equal(sid, 0), true_fn, false_fn)
        return new_lr, new_f_best, new_f_ma, new_patience, sid + 1
