from collections import namedtuple
import numpy as np
import tensorflow as tf


class Problem:
    def __init__(self, init, params):
        self.init = init
        self.params = params


    def get_next_dict(self, n_bptt_steps, batch_size):
        raise NotImplementedError


class Optimizee:
    def __init__(self):
        self.coord_vector = None
        self.coord_pos = 0
        self.coord_vars = {}
        self.vars_ = set()
    
    
    def custom_getter(self, getter, name, shape=None, *args, **kwargs):
        def dim_to_int(dim):
            if isinstance(dim, tf.Dimension):
                dim = dim.value
            return dim

        if kwargs.get('trainable', True):
            shape = [dim_to_int(d) for d in shape]
            dim = np.prod(shape)

            var = tf.reshape(self.coord_vector[0, self.coord_pos: self.coord_pos + dim], shape)

            pos = (self.coord_pos, self.coord_pos + dim)
            self.coord_pos += dim
            var = tf.identity(var, name=name)

            self.coord_vars[name] = {'var': var, 'initializer': kwargs['initializer'], 'pos': pos, 'shape': shape}
            return var
        else:
            return getter(name, shape, *args, **kwargs)


    def build(self):
        raise NotImplementedError


    def get_x_dim(self):
        raise NotImplementedError


    def loss(self, x, i):
        raise NotImplementedError


    def grad(self, x, f):
        g = tf.gradients(f, x)[0]
        return g


    def sample_problem(self, batch_size=1):
        raise NotImplementedError
