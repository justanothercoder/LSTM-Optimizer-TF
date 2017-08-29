import tensorflow as tf
from . import basic_model
from .config import Config


class MomentumOpt(basic_model.BasicModel):
    def __init__(self, lr, mu=0.9, **kwargs):
        super(MomentumOpt, self).__init__(Config(), **kwargs)
        self.lr = lr
        self.mu = mu
    
    
    def init_state(self, x):
        v = tf.zeros(shape=tf.shape(x))
        return (v,)


    def build_pre(self):
        pass
        

    def step(self, g, state):
        v, = state

        v = self.mu * v - self.lr * g
        step = v

        return step, (v,)
    
    
    def restore(self, eid):
        pass


    def save(self, eid):
        pass
