import tensorflow as tf
import basic_model


class SgdOpt(basic_model.BasicModel):
    def __init__(self, optimizee, lr, beta1=0.9, beta2=0.999, **kwargs):
        super(SgdOpt, self).__init__(optimizee, **kwargs)
        self.lr = lr


    def _build_pre(self):
        pass
        

    def _iter(self, f, i, state):
        x, = state

        fx, g = self._fg(f, x, i)
        g = tf.stop_gradient(g)

        x -= self.lr * g

        return [x], fx

