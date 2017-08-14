import tensorflow as tf
from . import basic_model


class AdamOpt(basic_model.BasicModel):
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8, enable_reduce=False, factor=0.5, patience_max=10, epsilon=1e-4, **kwargs):
        super(AdamOpt, self).__init__(save_tf_data=False, **kwargs)

        self.lr_init = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.factor = factor
        self.patience_max = patience_max
        self.enable_reduce = enable_reduce
        self.epsilon = epsilon
    
    
    def build_inputs(self):
        self.x = tf.placeholder(tf.float32, [None, None], name='x')
        self.m = tf.placeholder(tf.float32, [None, None], name='v')
        self.v = tf.placeholder(tf.float32, [None, None], name='v')
        self.b1t = tf.placeholder(tf.float32, [None], name='beta1')
        self.b2t = tf.placeholder(tf.float32, [None], name='beta1')

        self.input_state = dict(x=self.x, m=self.m, v=self.v, b1t=self.b1t, b2t=self.b2t)
        if self.enable_reduce:
            self.lr = tf.placeholder(tf.float32, [None], name='lr')
            self.f_best = tf.placeholder(tf.float32, [None], name='f_best')
            self.f_ma = tf.placeholder(tf.float32, [None], name='f_ma')
            self.patience = tf.placeholder(tf.int32, [None], name='patience')
            self.sid = tf.placeholder(tf.int32, [], name='sid')

            self.input_state.update(dict(lr=self.lr, f_best=self.f_best, f_ma=self.f_ma, patience=self.patience, sid=self.sid))

        return self.input_state
    
    
    def build_initial_state(self):
        x = self.x
        m = tf.zeros(tf.shape(x))
        v = tf.zeros(tf.shape(x))
        b1t = tf.ones([tf.shape(x)[0]])
        b2t = tf.ones([tf.shape(x)[0]])

        self.initial_state = dict(x=x, m=m, v=v, b1t=b1t, b2t=b2t)
        if self.enable_reduce:
            lr = self.lr_init * tf.ones([tf.shape(x)[0]])
            f_best = tf.zeros([tf.shape(x)[0]])
            f_ma = tf.zeros([tf.shape(x)[0]])
            patience = tf.zeros([tf.shape(x)[0]])

            sid = tf.zeros([])

            self.initial_state.update(dict(lr=lr, f_best=f_best, f_ma=f_ma, patience=patience, sid=sid))

        return self.initial_state


    def build_pre(self):
        pass


    def reduce_lr_on_plateau(self, state, f):

        def true_fn():
            new_f_ma = f
            new_f_best = f
            new_lr = state['lr']
            patience = state['patience']
            return new_f_ma, new_f_best, new_lr, patience

        def false_fn():
            new_f_ma = 0.95 * state['f_ma'] + 0.05 * f
            new_f_best = tf.maximum(new_f_ma, state['f_best'])

            # (old - new) / old < eps

            patience = tf.where(
                        #tf.greater(new_f_best, state['f_best']),
                        #tf.greater(new_f_best, state['f_best'] - self.epsilon),
                        tf.less((state['f_best'] - new_f_best) / state['f_best'], self.epsilon),
                        state['patience'] + 1,
                        state['patience']
                    )

            new_lr = tf.where(
                        tf.equal(patience, self.patience_max),
                        state['lr'] * self.factor,
                        state['lr']
                    )

            patience = tf.where(
                        tf.equal(patience, self.patience_max),
                        tf.zeros_like(patience),
                        patience
                    )
            return new_f_ma, new_f_best, new_lr, patience

        new_f_ma, new_f_best, new_lr, patience = tf.cond(tf.equal(state['sid'], 0), true_fn, false_fn)

        return dict(f_ma=new_f_ma, f_best=new_f_best, lr=new_lr, patience=patience, sid=state['sid'] + 1)


    def step(self, f, g, state):
        x, m, v, b1t, b2t = tuple(state[name] for name in ['x', 'm', 'v', 'b1t', 'b2t'])

        m = self.beta1 * m + (1 - self.beta1) * g
        v = self.beta2 * v + (1 - self.beta2) * tf.square(g)

        b1t *= self.beta1
        b2t *= self.beta2

        a = tf.expand_dims(tf.sqrt(1 - b2t) / (1 - b1t), -1)

        if self.enable_reduce:
            lr_state = self.reduce_lr_on_plateau(state, f)

        if not self.enable_reduce:
            s = self.lr_init * a * m / (tf.sqrt(v) + self.eps)
        else:
            s = lr_state['lr'] * a * m / (tf.sqrt(v) + self.eps)

        x -= s
        new_state = dict(x=x, m=m, v=v, b1t=b1t, b2t=b2t)

        if self.enable_reduce:
            new_state.update(lr_state)

        return dict(state=new_state)
    
    
    def restore(self, eid):
        pass


    def save(self, eid):
        pass

