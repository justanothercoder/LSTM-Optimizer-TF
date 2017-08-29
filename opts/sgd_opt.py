import tensorflow as tf
from .basic_model import BasicModel
from .config import Config


class SgdOpt(BasicModel):
    def __init__(self, lr, **kwargs):
        super(SgdOpt, self).__init__(Config(), **kwargs)
        self.lr = lr


    def build_pre(self):
        pass
        

    def init_state(self, _):
        return tuple()


    def step(self, g, state):
        step = self.lr * g
        return step, ()

    
    def restore(self, eid):
        pass


    def save(self, eid):
        pass
