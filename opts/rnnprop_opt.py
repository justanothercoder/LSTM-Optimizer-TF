import pathlib
import tensorflow as tf

from .basic_model import BasicModel, Point
from rnnprop.nn_opt.rnn import RNNpropModel


class RNNPropOpt(BasicModel):
    def __init__(self, name='rnnprop', eid=340, **kwargs):
        super(RNNPropOpt, self).__init__(None, name=name, snapshot_path=pathlib.Path('rnnprop/snapshots'))
        self.name = name
        self.model = RNNpropModel
        self.old_build = self.model._build
        self.model._build = lambda *args: None
        self.eid = eid

        self.restored = False


    def build(self, optimizees, build_config, **kwargs):
        self.opt = RNNpropModel('rnnprop', is_training=not build_config.inference_only, **kwargs)
        super(RNNPropOpt, self).build(optimizees, build_config, **kwargs)


    def build_pre(self):
        self.opt._build_pre()


    def init_state(self, x):
        self.opt._build_input()

        self.opt.x = x[0]
        self.opt.input_state = [x[0]]

        self.opt._build_initial()
        return tuple(self.opt.initial_state)


    def restore(self, eid):
        if self.restored:
            return

        lstm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)
        self.lstm_vars = lstm_vars

        var_list = {}
        for var in lstm_vars:
            new_name = var.name.replace(self.scope.name, 'nn_opt/loop')
            if new_name.endswith(':0'):
                new_name = new_name[:-2]
            var_list[new_name] = var

        self.saver = tf.train.Saver(var_list=var_list)
        super(RNNPropOpt, self).restore(eid)
        self.restored = True


    def step_with_func(self, f, state, i):
        x = state[1][3]
        
        def opt_loss(i, x):
            opt_loss = f(x[None], i)[0]
            return opt_loss
        
        value, gradient = f(x[None], i)
        gradient = tf.stop_gradient(gradient)
        gradient_norm = tf.reduce_sum(tf.square(gradient), axis=-1)

        output = Point(value, gradient, gradient_norm)

        rnn_state, _ = self.opt._iter(opt_loss, i, state.rnn_state)
        return output, state._replace(x=rnn_state[3][None], rnn_state=rnn_state)
