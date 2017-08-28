import copy
import time
from collections import namedtuple
import random
import numpy as np
from contextlib import contextmanager


@contextmanager
def set_random_state(r):
    old_state = np.random.get_state()
    np.random.set_state(r.get_state())

    yield

    r.set_state(np.random.get_state())
    np.random.set_state(old_state)


class Problem:
    def __init__(self, name, optim, init, params):
        self.name = name
        self.optim = optim
        self.init = init
        self.params = params

        self.np_random = np.random.RandomState()
        self.np_random.set_state(np.random.get_state())


    def get_next_dict(self, n_bptt_steps, batch_size):
        with set_random_state(self.np_random):
            feed_dict = self.optim.get_next_dict(n_bptt_steps, batch_size)
        return feed_dict


class RandomProducer:
    def __init__(self, optimizees, seed=None):
        self.optimizees = list(optimizees.items())
        self.seed = seed or int(time.time())

        self.random = random.Random(self.seed)
        self.np_random = np.random.RandomState(self.seed)


    def reset(self):
        self.random.seed(self.seed)
        self.np_random.seed(self.seed)


    def sample(self, batch_size=1, name=None):
        if name is None:
            name, optim = self.random.choice(self.optimizees)
        else:
            optim = self.optimizees[name]

        with set_random_state(self.np_random):
            #init, params = optim.sample_problem(batch_size)
            #problem = Problem(name=name, optim=copy.copy(optim), init=init, params=params)
            problem = optim.sample_problem(batch_size)

        return problem


    def sample_sequence(self, n_batches, batch_size=1, name=None):
        for _ in range(n_batches):
            yield self.sample(batch_size, name)


class FixedProducer:
    def __init__(self, optimizees):
        self.optimizees = optimizees


    def new(self, data_size, batch_size):
        self.data_size = data_size
        self.batch_size = batch_size
        self.data = list(RandomProducer(self.optimizees).sample_sequence(data_size, batch_size))
        self.p = 0
        return self


    def sample(self, batch_size=1):
        p = self.p
        self.p = (self.p + 1) % self.data_size
        return self.data[p] 


    def sample_sequence(self):
        for p in self.data:
            yield p
