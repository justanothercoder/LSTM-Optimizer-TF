from collections import namedtuple
import numpy as np
import tensorflow as tf
from problem_producer import set_random_state


class Problem:
    def __init__(self, init, params):
        self.init = init
        self.params = params


    def get_next_dict(self, n_bptt_steps, batch_size):
        raise NotImplementedError


class SimpleNonStochProblem(Problem):
    def __init__(self, init, params, name=None):
        super(SimpleNonStochProblem, self).__init__(init, params)
        self.name = name


    def get_next_dict(self, n_bptt_steps, batch_size):
        return { }


class BatchedStochProblem(Problem):
    def __init__(self, init, params, dataset, batch_size, x, y, iteration='random', name=None):
        super(BatchedStochProblem, self).__init__(init, params)
        self.name = name

        self.dataset = dataset
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.iteration = iteration
            
        if iteration == 'random':
            self.np_random = np.random.RandomState()
            self.np_random.set_state(np.random.get_state())


    def get_next_dict(self, n_bptt_steps, n_functions=1):
        x = np.zeros((n_bptt_steps, n_functions, self.batch_size, self.dataset.num_features)) 
        y = np.zeros((n_bptt_steps, n_functions, self.batch_size)) 

        if self.iteration == 'random':
            with set_random_state(self.np_random):
                random_batches = self.dataset.random_batch_iterator(n_bptt_steps, self.batch_size)

                for i, batch in enumerate(random_batches):
                    x[i], y[i] = batch 
        elif iteration == 'full':
            batches = self.dataset.batch_iterator(n_bptt_steps, self.batch_size, shuffle=False)

            for i, batch in enumerate(batches):
                x[i], y[i] = batch

        return { 
            self.x: x,
            self.y: y,
        } 


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
