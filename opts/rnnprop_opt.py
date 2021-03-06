import pathlib

import numpy as np
import tensorflow as tf

from . import basic_model
from rnnprop.nn_opt.rnn import RNNpropModel


class RNNPropOpt(basic_model.BasicModel):
    def __init__(self, name='rnnprop', eid=250, **kwargs):
        super(RNNPropOpt, self).__init__(name=name, snapshot_path=pathlib.Path('rnnprop/snapshots'), **kwargs)
        self.name = name
        self.model = RNNpropModel
        self.old_build = self.model._build
        self.model._build = lambda *args: None
        self.eid = eid

        self.restored = False


    def build(self, optimizees, n_bptt_steps=20,
              loss_type='log', optimizer='adam',
              lambd=0., lambd_l1=0., inference_only=False,
              normalize_lstm_grads=False, grad_clip=1.,
              use_moving_averages=False, stop_grad=True, **kwargs):

        self.opt = RNNpropModel('rnnprop', is_training=not inference_only, **kwargs)
        super(RNNPropOpt, self).build(optimizees, n_bptt_steps,
              loss_type, optimizer, lambd, lambd_l1, inference_only,
              normalize_lstm_grads, grad_clip, use_moving_averages, stop_grad, **kwargs)


    def build_pre(self):
        self.opt._build_pre()

    
    def build_inputs(self):
        self.opt._build_input()
        return self.opt.input_state


    def build_initial_state(self):
        self.opt._build_initial()
        return self.opt.initial_state


    def step(self, f, i, state):
        state, _ = self.opt._iter(f, i, state)

        return {'state': state}
    

    def restore(self, eid):
        if self.restored:
            return

        lstm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)
        self.lstm_vars = lstm_vars

        var_list = {}
        for var in lstm_vars:
            new_name = var.name.replace(self.scope.name + '/inference_scope', 'nn_opt/loop')
            if new_name.endswith(':0'):
                new_name = new_name[:-2]
            #print(new_name)
            var_list[new_name] = var

        #var_list = lstm_vars

        self.saver = tf.train.Saver(var_list=var_list)
        super(RNNPropOpt, self).restore(eid)
        self.restored = True
