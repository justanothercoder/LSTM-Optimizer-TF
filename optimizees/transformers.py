import numpy as np
import tensorflow as tf
from . import optimizee


class ClipByValue(optimizee.Optimizee):
    def __init__(self, optimizee, clip_low, clip_high):
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

    
    def get_initial_x(self, batch_size=1):
        return self.optim.get_initial_x(batch_size)


    def get_new_params(self, batch_size=1):
        return self.optim.get_new_params(batch_size)


    def get_next_dict(self, n_bptt_steps, batch_size=1):
        return self.optim.get_next_dict(n_bptt_steps, batch_size)

    
    def get_x_dim(self):
        return self.optim.get_x_dim()


class UniformRandomScaling(optimizee.Optimizee):
    def __init__(self, optimizee, r=3.0):
        self.optim = optimizee
        self.r = r


    def build(self):
        self.optim.build()

        with tf.variable_scope('random_scaling'):
            self.c = tf.placeholder(tf.float32, [None, None], name='c')


    def loss(self, x, i):
        f, g = self.optim.loss(self.c * x, i)
        return f, g

    
    def get_initial_x(self, batch_size=1):
        x = self.optim.get_initial_x(batch_size)
        self.coef = np.exp(np.random.uniform(-self.r, self.r, size=x.shape))
        return x / self.coef


    def get_new_params(self, batch_size=1):
        d = self.optim.get_new_params(batch_size)
        d[self.c] = self.coef
        return d


    def get_next_dict(self, n_bptt_steps, batch_size=1):
        return self.optim.get_next_dict(n_bptt_steps, batch_size)

    
    def get_x_dim(self):
        return self.optim.get_x_dim()


class ConcatAndSum(optimizee.Optimizee):
    def __init__(self, optimizee_list, weighted=False):
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


    def get_initial_x(self, batch_size=1):
        inits = [opt.get_initial_x(batch_size) for opt in self.optim_list]
        init = np.concatenate(inits, axis=-1)
        return init


    def get_new_params(self, batch_size=1):
        params = { }
        for opt in self.optim_list:
            params.update(opt.get_new_params(batch_size))
        return params


    def get_next_dict(self, n_bptt_steps, batch_size=1):
        d = { }
        for opt in self.optim_list:
            d.update(opt.get_next_dict(n_bptt_steps, batch_size))
        return d

    
    def get_x_dim(self):
        return tf.add_n([o.get_x_dim() for o in self.optim_list])


class NormalNoisyGrad(optimizee.Optimizee):
    def __init__(self, opt, stddev=0.1):
        self.stddev = stddev
        self.opt = opt


    def build(self):
        self.opt.build()


    def loss(self, x, i):
        f, g = self.opt.loss(x, i)
        g = g + tf.random_normal(tf.shape(g), mean=0, stddev=self.stddev)
        return f, g


    def get_initial_x(self, batch_size=1):
        return self.opt.get_initial_x(batch_size)


    def get_new_params(self, batch_size=1):
        return self.opt.get_new_params(batch_size)


    def get_next_dict(self, n_bptt_steps, batch_size=1):
        return self.opt.get_next_dict(n_bptt_steps, batch_size)

    
    def get_x_dim(self):
        return self.opt.get_x_dim()
