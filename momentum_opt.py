import tensorflow as tf
import basic_model


class MomentumOpt(basic_model.BasicModel):
    def __init__(self, optimizee, lr, mu=0.9, **kwargs):
        super(MomentumOpt, self).__init__(optimizee, **kwargs)

        self.lr = lr
        self.mu = mu
    
    
    def _build_input(self):
        self.x = tf.placeholder(tf.float32, [None], name='x')
        self.v = tf.placeholder(tf.float32, [None], name='v')
        self.input_state = [self.x, self.v]
    
    
    def _build_initial_state(self):
        x = self.x
        v = tf.zeros(shape=tf.shape(x))
        self.initial_state = [x, v]


    def _build_pre(self):
        pass
        

    def _iter(self, f, i, state):
        x, v = state

        fx, g = self._fg(f, x, i)
        g = tf.stop_gradient(g)

        g_norm = tf.reduce_sum(g**2)

        v = self.mu * v - self.lr * g
        x += v

        return [x, v], fx, g_norm
