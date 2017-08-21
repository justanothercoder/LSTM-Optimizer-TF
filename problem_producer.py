import time
from collections import namedtuple
import random
import numpy as np

Problem = namedtuple('Problem', ['name', 'optim', 'init', 'params'])

class ProblemProducer:
    def __init__(self, optimizees, seed=None):
        self.optimizees = list(optimizees.items())
        self.seed = seed or int(time.time())

        random.seed(self.seed)
        np.random.seed(self.seed)


    def reset(self):
        random.seed(self.seed)
        np.random.seed(self.seed)


    def sample(self, batch_size=1, name=None):
        if name is None:
            name, optim = random.choice(self.optimizees)
        else:
            optim = self.optimizees[name]

        init, params = optim.sample_problem(batch_size)
        return Problem(name=name, optim=optim, init=init, params=params)


    def sample_sequence(self, n_batches, batch_size=1, name=None):
        for _ in range(n_batches):
            yield self.sample(batch_size, name)
