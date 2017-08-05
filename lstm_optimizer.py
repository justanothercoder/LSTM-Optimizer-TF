from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow.python.training import optimizer
from tensorflow.python.ops import variable_scope as vs

import util

GATE_NONE = 0
GATE_OP = 1
GATE_GRAPH = 2


class LSTMOptimizer:
    def __init__(self, model_path, eid):
        self.opt = util.load_opt(model_path)
        self.opt.kwargs = {}
        self.grad_opt = tf.train.GradientDescentOptimizer(1.0)

        with tf.variable_scope('lstm_opt_scope') as scope:
            self.opt.build_pre()

        self.state = defaultdict()


    def prepare_states(self):
        self.shapes, self.sizes, vector = self.pack(self.tvars)
        n_coords = np.sum(self.sizes)

        #input_state = self.opt.build_inputs()
        self.opt.x = tf.reshape(vector, [1, -1])
        initial_state = self.opt.build_initial_state()
    
        self.state = {}

        for k, v in initial_state.items():
            if k in {'x', 'lstm_state'}:
                continue

            if k in {'m', 'v', 'm_norm', 'v_norm', 'loglr'}:
                v.set_shape([1] + vector.get_shape().as_list())
            elif k in {'b1t', 'b2t'}:
                v.set_shape([1])
                
            self.state[k] = tf.Variable(v)
            
        self.state['x'] = tf.zeros([1, n_coords])

        lstm_state = []
            
        for i, (c, h) in enumerate(initial_state['lstm_state']):
            c.set_shape([n_coords, c.get_shape()[1]])
            h.set_shape([n_coords, h.get_shape()[1]])

            c_ = tf.Variable(c) 
            h_ = tf.Variable(h)

            self.state['lstm_state_{}_c'.format(i)] = c_
            self.state['lstm_state_{}_h'.format(i)] = h_

            lstm_state.append((c_, h_))

        self.state['lstm_state'] = lstm_state


    def pack(self, v):
        shapes = [x.get_shape().as_list() for x in v]
        sizes = [np.prod(shape).astype(np.int32) for shape in shapes]
        packed = tf.concat([tf.reshape(x, [-1]) for x in v], axis=-1)

        return shapes, sizes, packed


    def unpack(self, v):
        splits = tf.split(v, self.sizes)
        return [tf.reshape(s, shape) for s, shape in zip(splits, self.shapes)]


    def prepare_grads(self):
        self.prepare_states()
        self.update_ops = []

        _, _, gradient = self.pack(self.grads)
        gradient = tf.reshape(gradient, [1, -1])

        old_x = self.state['x']
        with tf.variable_scope('step_scope') as scope:
            self.opt.scope = scope
            new_state = self.opt.step(gradient, self.state)['state']
        new_x = new_state['x']

        step = (new_x - old_x)[0]
        steps_and_vars = list(zip(self.unpack(-step), self.tvars))

        for k in new_state:
            if k not in {'x', 'lstm_state'}:
                uop = tf.assign(self.state[k], new_state[k])
                self.update_ops.append(uop)

        return steps_and_vars


    def compute_gradients(self, loss, var_list, global_step=None,
                          gate_gradients=GATE_OP, aggregation_method=None,
                          colocate_gradients_with_ops=False, name=None,
                          grad_loss=None):
        return self.grad_opt.compute_gradients(loss, var_list=var_list, gate_gradients=gate_gradients,
                                               aggregation_method=aggregation_method,
                                               colocate_gradients_with_ops=colocate_gradients_with_ops,
                                               grad_loss=grad_loss)


    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        self.grads, self.tvars = zip(*[(g, v) for g, v in grads_and_vars if g is not None])

        steps_and_vars = self.prepare_grads()
        apply_op = self.grad_opt.apply_gradients(steps_and_vars, global_step=global_step, name=name)
        
        with tf.control_dependencies([apply_op]):
            train_op = tf.group(*self.update_ops)
        return train_op


    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None):

        grads_and_vars = self.grad_opt.compute_gradients(
                loss, var_list=var_list, gate_gradients=gate_gradients,
                aggregation_method=aggregation_method,
                colocate_gradients_with_ops=colocate_gradients_with_ops,
                grad_loss=grad_loss)

        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
                    "No gradients provided for any variable, check your graph for ops"
                    " that do not support gradients, between variables %s and loss %s." %
                    ([str(v) for _, v in grads_and_vars], loss))

        for g, v in grads_and_vars:
            print("g ", g)
            print("v ", v)

        return self.apply_gradients(grads_and_vars)
