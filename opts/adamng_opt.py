import tensorflow as tf
from . import basic_model


class AdamNGOpt(basic_model.BasicModel):
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super(AdamNGOpt, self).__init__(save_tf_data=False, **kwargs)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
    
    
    def build_inputs(self):
        self.x = tf.placeholder(tf.float32, [None, None], name='x')
        self.m = tf.placeholder(tf.float32, [None, None], name='v')
        self.v = tf.placeholder(tf.float32, [None, None], name='v')
        self.b1t = tf.placeholder(tf.float32, [None], name='beta1')
        self.b2t = tf.placeholder(tf.float32, [None], name='beta1')
        self.input_state = dict(x=self.x, m=self.m, v=self.v, b1t=self.b1t, b2t=self.b2t)
        return self.input_state
    
    
    def build_initial_state(self):
        x = self.x
        m = tf.zeros(tf.shape(x))
        v = tf.zeros(tf.shape(x))
        b1t = tf.ones([tf.shape(x)[0]])
        b2t = tf.ones([tf.shape(x)[0]])
        self.initial_state = dict(x=x, m=m, v=v, b1t=b1t, b2t=b2t)
        return self.initial_state


    def build_pre(self):
        pass
        

    def step(self, g, state):
        x, m, v, b1t, b2t = tuple(state[name] for name in ['x', 'm', 'v', 'b1t', 'b2t'])

        g = g / tf.norm(g, 2)

        m = self.beta1 * m + (1 - self.beta1) * g
        v = self.beta2 * v + (1 - self.beta2) * tf.square(g)

        b1t *= self.beta1
        b2t *= self.beta2

        a = tf.expand_dims(tf.sqrt(1 - b2t) / (1 - b1t), -1)
        s = self.lr * a * m / (tf.sqrt(v) + self.eps)

        x -= s

        return {
            'state': dict(x=x, m=m, v=v, b1t=b1t, b2t=b2t),
        }
    
    
    def restore(self, eid):
        pass


    def save(self, eid):
        pass

