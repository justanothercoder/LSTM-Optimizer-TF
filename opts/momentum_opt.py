import tensorflow as tf
from . import basic_model


class MomentumOpt(basic_model.BasicModel):
    def __init__(self, lr, mu=0.9, **kwargs):
        super(MomentumOpt, self).__init__(save_tf_data=False, **kwargs)
        self.lr = lr
        self.mu = mu
    
    
    def build_inputs(self):
        self.x = tf.placeholder(tf.float32, [None, None], name='x')
        self.v = tf.placeholder(tf.float32, [None, None], name='v')
        self.input_state = dict(x=self.x, v=self.v)
        return self.input_state
    
    
    def build_initial_state(self):
        x = self.x
        v = tf.zeros(shape=tf.shape(x))
        self.initial_state = dict(x=x, v=v)
        return self.initial_state


    def build_pre(self):
        pass
        

    def step(self, f, g, state):
        x, v = state['x'], state['v']

        v = self.mu * v - self.lr * g
        x += v

        return {
            'state': dict(x=x, v=v),
        }
    
    
    def restore(self, eid):
        pass


    def save(self, eid):
        pass
