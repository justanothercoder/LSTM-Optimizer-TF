import tensorflow as tf
from . import basic_model


class SgdOpt(basic_model.BasicModel):
    def __init__(self, lr, beta1=0.9, beta2=0.999, **kwargs):
        super(SgdOpt, self).__init__(save_tf_data=False, **kwargs)
        self.lr = lr


    def build_pre(self):
        pass
        
    def build_inputs(self):
        x = tf.placeholder(tf.float32, shape=[None, None])
        return dict(x=x)


    def step(self, f, g, state):
        x = state['x']
        x -= self.lr * g

        return {
            'state': dict(x=x),
        }

    
    def restore(self, eid):
        pass


    def save(self, eid):
        pass
