import tensorflow as tf

class Optimizee:
    def build(self):
        raise NotImplementedError


    def get_x_dim(self):
        raise NotImplementedError


    def loss(self, x, i):
        raise NotImplementedError


    def grad(self, x, f):
        g = tf.gradients(tf.reduce_sum(f), x)[0]
        return g


    def get_initial_x(self, batch_size=1):
        raise NotImplementedError


    def get_new_params(self, batch_size=1):
        raise NotImplementedError


    def get_next_dict(self, n_bptt_steps, batch_size=1):
        raise NotImplementedError
