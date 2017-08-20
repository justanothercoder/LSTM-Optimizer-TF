import numpy as np
import tensorflow as tf
from . import optimizee


class ClipByValue(optimizee.Optimizee):
    def __init__(self, optimizee, clip_low, clip_high):
        super(ClipByValue, self).__init__()
        self.optim = optimizee
        self.clip_low = clip_low
        self.clip_high = clip_high


    def build(self):
        self.optim.build()


    def loss(self, x, i):
        f, _ = self.optim.loss(x, i)
        #g = self.grad(x, f)

        f = tf.clip_by_value(f, self.clip_low, self.clip_high)
        g = self.grad(x, f)
        return f, g

    
    def sample_problem(self, batch_size=1):
        return self.optim.sample_problem(batch_size)


    def get_next_dict(self, n_bptt_steps, batch_size=1):
        return self.optim.get_next_dict(n_bptt_steps, batch_size)

    
    def get_x_dim(self):
        return self.optim.get_x_dim()


class UniformRandomScaling(optimizee.Optimizee):
    def __init__(self, optimizee, r=3.0):
        super(UniformRandomScaling, self).__init__()
        self.optim = optimizee
        self.r = r


    def build(self):
        self.optim.build()

        with tf.variable_scope('random_scaling'):
            self.c = tf.placeholder(tf.float32, [None, None], name='c')


    def loss(self, x, i):
        f, g = self.optim.loss(self.c * x, i)
        return f, g

    
    def sample_problem(self, batch_size=1):
        x, d = self.optim.sample_problem(batch_size)

        coef = np.exp(np.random.uniform(-self.r, self.r, size=x.shape))
        init = x / coef
        d[self.c] = coef
        return init, d


    def get_next_dict(self, n_bptt_steps, batch_size=1):
        return self.optim.get_next_dict(n_bptt_steps, batch_size)

    
    def get_x_dim(self):
        return self.optim.get_x_dim()


class ConcatAndSum(optimizee.Optimizee):
    def __init__(self, optimizee_list, weighted=False):
        super(ConcatAndSum, self).__init__()
        self.optim_list = optimizee_list
        self.weighted = weighted


    def build(self):
        for opt in self.optim_list:
            opt.build()


    def loss(self, x, i):
        batch_size = tf.shape(x)[0]

        fs = []
        s = tf.constant(0, tf.int32)

        for opt in self.optim_list:
            dim = opt.get_x_dim()

            begin = [0, s]
            size = [batch_size, dim]
            t = tf.slice(x, begin, size)

            f, _ = opt.loss(t, i)
            fs.append(f)

            s += dim

        f = tf.add_n(fs)
        g = self.grad(x, f)

        if self.weighted:
            for i in range(len(fs)):
                c = tf.stop_gradient(1 / tf.reduce_max(fs[i]))
                fs[i] *= c

        f = tf.add_n(fs)

        return f, g


    def sample_problem(self, batch_size=1):
        inits = []
        params = {}

        for opt in self.optim_list:
            x, p = opt.sample_problem(batch_size)
            inits.append(x)
            params.update(p)

        init = np.concatenate(inits, axis=-1)
        return init, params


    def get_next_dict(self, n_bptt_steps, batch_size=1):
        d = { }
        for opt in self.optim_list:
            d.update(opt.get_next_dict(n_bptt_steps, batch_size))
        return d

    
    def get_x_dim(self):
        return tf.add_n([o.get_x_dim() for o in self.optim_list])


class NormalNoisyGrad(optimizee.Optimizee):
    def __init__(self, opt, stddev=0.1):
        super(NormalNoisyGrad, self).__init__()
        self.stddev = stddev
        self.opt = opt


    def __getattr__(self, name):
        return getattr(self.opt, name)


    def build(self):
        self.opt.build()


    def loss(self, x, i):
        f, g = self.opt.loss(x, i)
        new_g = g + tf.random_normal(tf.shape(g), mean=0, stddev=self.stddev)
        return f, new_g


    def sample_problem(self, batch_size=1):
        init = self.opt.get_initial_x(batch_size)
        params = self.opt.get_new_params(batch_size)
        return init, params


    def get_next_dict(self, n_bptt_steps, batch_size=1):
        return self.opt.get_next_dict(n_bptt_steps, batch_size)

    
    def get_x_dim(self):
        return self.opt.get_x_dim()
