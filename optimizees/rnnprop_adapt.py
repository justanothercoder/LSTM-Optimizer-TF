import numpy as np
import tensorflow as tf
from . import optimizee


class RNNPropAdapter(optimizee.Optimizee):
    def __init__(self, rnnprop_opt, reshape_f=False):
        super(RNNPropAdapter, self).__init__()
        self.opt = rnnprop_opt
        self.reshape_f = reshape_f


    def get_x_dim(self):
        return self.opt.get_x_dim()


    def build(self):
        self.opt.build()

    
    def loss(self, x, i):
        f = self.opt.loss(i, x[0])

        if self.reshape_f:
            f = f[None]

        g = self.grad(x, f)
        return f, g


    def sample_problem(self, batch_size=1):
        init = self.opt.get_initial_x()
        return self.RNNPropProblem(init[None], self.opt.next_internal_feed_dict())


    class RNNPropProblem(optimizee.Problem):
        def get_next_dict(self, n_bptt_steps, batch_size=1):
            return self.opt.next_feed_dict(n_bptt_steps)
