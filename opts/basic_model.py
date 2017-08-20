import pprint
import time
import os
import pathlib
import random
import numpy as np

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple

from cells import LSTMOptCell, OptFuncCell

class BasicModel:
    def __init__(self, name=None, snapshot_path=None, debug=False):
        self.bid = 0
        self.name = name
        self.snapshot_path = snapshot_path
        self.debug = debug

    
    @property
    def is_rnnprop(self):
        return self.__class__.__name__ == 'RNNPropOpt'
        

    def build_inputs(self):
        raise NotImplementedError


    def build_initial_state(self):
        raise NotImplementedError


    def build_pre(self):
        raise NotImplementedError


    def step(self, f, i, state):
        raise NotImplementedError


    def build(self, optimizees, n_bptt_steps=20,
              loss_type='log',
              lambd=0., lambd_l1=0., inference_only=False,
              normalize_lstm_grads=False, grad_clip=1.,
              stop_grad=True, dynamic=False, cell=False, **kwargs):

        self.session = tf.get_default_session()
        self.optimizees = optimizees
        self.n_bptt_steps = n_bptt_steps
        ops = {}

        self.kwargs = kwargs
        vars_opt = set()

        with tf.variable_scope('opt_scope') as scope:
            self.scope = scope
            self.build_pre()

            self.input_state = self.build_inputs()
            self.initial_state = self.build_initial_state()

            if cell:
                kwargs = util.get_kwargs(self.__init__, self.__dict__)
                self.cell = LSTMOptCell(**kwargs)

            for opt_name, optimizee in optimizees.items():
                with tf.variable_scope('inference_scope'):
                    inference = self.inference(optimizee, self.input_state, n_bptt_steps, stop_grad=stop_grad, dynamic=dynamic, cell=cell)
                    vars_opt |= set(optimizee.vars_)
                    
                losses = self.loss(inference, lambd=lambd, lambd_l1=lambd_l1, loss_type=loss_type)

                ops[opt_name] = {
                    'inference': inference,
                    'losses': losses,
                }

                if not scope.reuse:
                    scope.reuse_variables()
        
        with tf.variable_scope('opt_scope', reuse=False) as scope:
            self.all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
            self.all_vars = list(set(self.all_vars) - vars_opt)

            if not inference_only and self.all_vars:
                self.train_lr = tf.placeholder(tf.float32, shape=[], name='train_lr')
                self.momentum = tf.placeholder(tf.float32, shape=[], name='momentum')
                self.optimizer = tf.train.AdamOptimizer(self.train_lr, beta1=self.momentum)

                for opt_name in optimizees:
                    losses = ops[opt_name]['losses']

                    grads = self.grads(self.optimizer, losses, normalize_lstm_grads=normalize_lstm_grads, grad_clip=grad_clip)
                    train_op = self.train_op(self.optimizer, grads)

                    ops[opt_name].update({
                        'grads': grads,
                        'train_op': train_op
                    })

                    if not scope.reuse:
                        scope.reuse_variables()

            self.ops = ops
            self.saver = tf.train.Saver(max_to_keep=None, var_list=self.all_vars, allow_empty=True)


    def dynamic_inference(self, optimizee, input_x, input_state, n_bptt_steps, stop_grad=True):
        ks, vs = zip(*list(state.items()))
        ks = list(ks)
        vs = list(vs)

        def dict_to_tuple(d):
            print(d.keys())
            return tuple(d[k] for k in ks)


        def tuple_to_dict(t):
            return dict(zip(ks, t))


        def cond(sid, *loop_vars):
            return tf.less(sid, n_bptt_steps)


        def body(sid, vals, norms, x, *state):
            state = tuple_to_dict(state)
            value, gradient, gradient_norm = self._fg(optimizee.loss, x, sid, stop_grad)

            step_info = self.step(value, gradient, state)
            new_state = step_info['state']
            new_state = dict_to_tuple(new_state)

            new_vals = tf.concat([vals, tf.expand_dims(value, 0)], axis=0)
            new_norms = tf.concat([norms, tf.expand_dims(gradient_norm, 0)], axis=0)

            out_state = (sid + 1, new_vals, new_norms, x + step_info['step']) + new_state
            
            return out_state


        vals_init = tf.zeros([0, tf.shape(input_x)[0]])
        norms_init = tf.zeros([0, tf.shape(input_x)[0]])
        state_init = dict_to_tuple(input_state)

        i = tf.constant(0)
        in_state = (i, vals_init, norms_init, input_x) + state_init

        def get_shapes(t):
            shapes = []

            for i in t:
                if isinstance(i, tf.Tensor):
                    shapes.append(i.get_shape())
                else:
                    s = get_shapes(i)
                    if type(i) in (tuple, list):
                        s = type(i)(s)
                    else:
                        s = type(i)(c=s[0], h=s[1])
                    shapes.append(s)
            
            return tuple(shapes)

        shape_invariants = (
                    i.get_shape(),
                    tf.TensorShape([None] + vals_init.get_shape().as_list()[1:]),
                    tf.TensorShape([None] + norms_init.get_shape().as_list()[1:]),
                ) + get_shapes(state_init)

        _, vals, norms, *r = tf.while_loop(cond, body, in_state, shape_invariants=shape_invariants)

        vals = tf.unstack(vals, num=n_bptt_steps, axis=0)
        norms = tf.unstack(norms, num=n_bptt_steps, axis=0)

        steps_info = [{'value': v, 'gradient_norm': n} for v, n in zip(vals, norms)]
        final_state = tuple_to_dict(r)
        return steps_info, final_state


    def static_inference(self, optimizee, input_x, input_state, n_bptt_steps, stop_grad=True):
        steps_info = []

        state = input_state
        x = input_x
        scope = tf.get_variable_scope()
                
        def opt_loss(i, x):
            opt_loss = optimizee.loss(x[None], i)[0]
            print(opt_loss.get_shape())
            return opt_loss

        for i in range(n_bptt_steps):
            if self.is_rnnprop:
                x = state[3]
                value, gradient, gradient_norm = self._fg(optimizee.loss, x[None], i, stop_grad)
                step_info = self.step(opt_loss, i, state)
            else:
                value, gradient, gradient_norm = self._fg(optimizee.loss, x, i, stop_grad)
                step_info = self.step(value, gradient, state)
                x += step_info['step']

            state = step_info['state']

            if not scope.reuse:
                scope.reuse_variables()

            step_info['value'] = value
            step_info['gradient_norm'] = gradient_norm

            steps_info.append(step_info)

        final_state = state
        return steps_info, final_state


    def cell_inference(optimizee, input_x, input_state, n_bptt_steps, stop_grad=True):
        pass


    def inference(self, optimizee, input_x, input_state, n_bptt_steps, stop_grad=True, dynamic=False, cell=False):
        if cell:
            steps_info, final_state = self.cell_inference(optimizee, input_state, n_bptt_steps, stop_grad=stop_grad)
        else:
            if dynamic:
                steps_info, final_state = self.dynamic_inference(optimizee, input_x, input_state, n_bptt_steps, stop_grad=stop_grad)
            else:
                steps_info, final_state = self.static_inference(optimizee, input_x, input_state, n_bptt_steps, stop_grad=stop_grad)

        ret = {
            'values': [info['value'] for info in steps_info],
            'norms' : [info['gradient_norm'] for info in steps_info],
            'final_state': final_state
        }

        first_step = steps_info[0]
        if not self.is_rnnprop:
            keys = set(first_step.keys())

            if 'state' in keys:
                state_keys = set(first_step['state'].keys())
            
                if 'loglr' in state_keys:
                    ret['lrs'] = [info['state']['loglr'] for info in steps_info]

            if 'cos_step_adam' in keys:
                ret['cosines'] = [info['cos_step_adam'] for info in steps_info]

        return ret


    def loss(self, inference, lambd=0., lambd_l1=0., loss_type='log'):
        values = tf.stack(inference['values'])
        if inference.get('keys') is not None:
            lrs = tf.stack([s['loglr'] for s in inference['lrs']])
        else:
            lrs = None

        losses = []

        if loss_type == 'log':
            loss = tf.reduce_mean(tf.log(values + 1e-8) - tf.log(values[:1] + 1e-8))

            if lrs is not None:
                lr_loss = -lambd * tf.reduce_mean(lrs - lrs[:1])
                losses.append(lr_loss)

        elif loss_type == 'log_smooth':
            smooth_vals = []
            for i in range(self.n_bptt_steps):
                if i == 0:
                    smooth_val = values[i]
                else:
                    smooth_val = 0.95 * smooth_val + 0.05 * values[i]
                smooth_vals.append(smooth_val)

            smooth_vals = tf.stack(smooth_vals)
            loss = tf.reduce_mean(tf.log(smooth_vals + 1e-8) - tf.log(smooth_vals[:1] + 1e-8))

        elif loss_type == 'sum':
            loss = tf.reduce_mean(values)
        else:
            loss = values[-1]

        if self.all_vars:
            weights = [v for v in self.all_vars if 'bias' not in v.name]
            reg_loss = tf.add_n([lambd_l1 * tf.norm(v, ord=1) for v in weights])
            losses.append(reg_loss)

        return [loss] + losses


    def grads(self, optimizer, losses, normalize_lstm_grads=False, grad_clip=1.):
        loss = tf.add_n(losses)
        grads = optimizer.compute_gradients(loss, var_list=self.all_vars)

        if normalize_lstm_grads:
            print("Using normalized meta-grads")
            norm = tf.global_norm(grads)
            grads = [(grad / (norm + 1e-8), var) for grad, var in grads]

        grads, _ = tf.clip_by_global_norm([g for g, _ in grads], grad_clip)
        grads = list(zip(grads, self.all_vars))

        return grads


    def train_op(self, optimizer, grads):
        train_op = optimizer.apply_gradients(grads)

        if self.debug:
            check_op = tf.add_check_numerics_ops()
            train_op = tf.group(train_op, check_op)

        return train_op


    def log(self, message, verbosity, level=0):
        if verbosity <= self.verbose:
            message = '\t' * level + message
            print(message)


    def test(self, eid, n_batches, n_steps=20, opt_name=None, verbose=1):
        self.restore(eid)
        self.verbose = verbose

        rets = []

        sample_optimizee = (opt_name is None)
        opt_names = list(self.optimizees.keys())

        for batch in range(n_batches):
            self.log("Batch: {}".format(batch), verbosity=1, level=0)
            if sample_optimizee:
                opt_name = random.choice(opt_names)

            batch_time = time.time()
            ret = self.test_one_iteration(n_steps, opt_name)
            batch_time = time.time() - batch_time
            self.log("Time: {}".format(batch_time), verbosity=1, level=1)
            rets.append(ret)

        return rets


    def test_one_iteration(self, n_steps, opt_name):
        self.bid += 1
        session = tf.get_default_session()

        ret = {}

        optimizee = self.optimizees[opt_name]

        x = optimizee.get_initial_x()
        if self.is_rnnprop:
            state = session.run(self.initial_state, feed_dict={self.opt.x: x[0]})
        else:
            state = session.run(self.initial_state, feed_dict={self.x: x})

        optimizee_params = optimizee.get_new_params()

        inf = self.ops[opt_name]['inference']
        losses = self.ops[opt_name]['losses']

        if hasattr(self, 'devices'):
            inf = list(inf.values())[0]
            losses = list(losses.values())[0]

        run_op = {
            'loss': losses[0],
            'values': inf['values'],
            'norms': inf['norms'],
            'final_state': inf['final_state']
        }

        steps_info = {
            'values': [],
            'norms': [],
        }

        if inf.get('cosines'):
            run_op['cosines'] = inf['cosines']
            steps_info['cosines'] = []

        if inf.get('lrs'):
            run_op['lrs'] = inf['lrs']
            steps_info['lrs'] = []

        losses = []

        for _ in range(n_steps // self.n_bptt_steps):
            feed_dict = optimizee_params
            if self.is_rnnprop:
                feed_dict.update({inp: init for inp, init in zip(self.input_state, state)})
            else:
                feed_dict.update({inp: state[name] for name, inp in self.input_state.items()})
            feed_dict.update(optimizee.get_next_dict(self.n_bptt_steps))
 
            info = session.run(run_op, feed_dict=feed_dict)
            state = info['final_state']

            losses.append(info['loss'])

            for k in steps_info:
                steps_info[k].extend(info[k])

        self.log("Loss: {}".format(np.mean(losses / np.log(10))), verbosity=2, level=2)

        ret['optimizee_name'] = opt_name
        ret['loss'] = np.nanmean(losses)

        for k in steps_info:
            ret[k] = np.array(steps_info[k])

        return ret


    def train(self, n_epochs, n_batches, batch_size=100, n_steps=20, train_lr=1e-2, momentum=0.9, eid=0, test=True, verbose=1):
        self.verbose = verbose
        self.lr = train_lr
        self.mu = momentum

        if eid > 0:
            self.restore(eid)
            self.bid = eid * n_batches

        train_rets = []
        test_rets = []

        sample_steps = bool(n_steps == 0)

        try:
            for epoch in range(eid, n_epochs):
                self.log("Epoch: {}".format(epoch), verbosity=1, level=0)
                epoch_time = time.time()

                loss = None

                for batch in range(n_batches):
                    self.log("Batch: {}".format(batch), verbosity=2, level=1)
                    if sample_steps:
                        #n_steps = int(np.random.exponential(scale=200)) + 50

                        exp_scale = min(50, epoch)
                        n_steps = int(np.random.exponential(scale=exp_scale)) + 1
                        n_steps *= self.n_bptt_steps
                        self.log("n_steps: {}".format(n_steps), verbosity=2, level=2)

                    batch_time = time.time()
                    ret = self.train_one_iteration(n_steps, batch_size)

                    if np.isnan(ret['loss']):
                        print("Loss is NaN")
                        #print(ret['fxs'])

                    if loss is not None:
                        loss = 0.9 * loss + 0.1 * ret['loss']
                    else:
                        loss = ret['loss']
                    batch_time = time.time() - batch_time

                    self.log("Batch time: {}".format(batch_time), verbosity=2, level=2)
                    train_rets.append(ret)

                self.log("Epoch time: {}".format(time.time() - epoch_time), verbosity=1, level=1)
                self.log("Epoch loss: {}".format(loss / np.log(10) / self.n_bptt_steps), verbosity=1, level=1)

                if test and (epoch + 1) % 10 == 0:
                    self.save(epoch + 1)

                    opt_name = random.choice(list(self.optimizees.keys()))

                    self.log("Test epoch: {}".format(epoch), verbosity=1, level=0)
                    test_epoch_time = time.time()

                    for batch in range(n_batches):
                        self.log("Test batch: {}".format(batch), verbosity=2, level=1)

                        batch_time = time.time()
                        ret = self.test_one_iteration(n_steps, opt_name)
                        batch_time = time.time() - batch_time
                        self.log("Batch time: {}".format(batch_time), verbosity=2, level=2)

                        test_rets.append(ret)

                    test_epoch_time = time.time() - test_epoch_time
                    self.log("Epoch time: {}".format(test_epoch_time), verbosity=1, level=1)
        except KeyboardInterrupt:
            print("Stopped training early")

        return train_rets, test_rets


    def train_one_iteration(self, n_steps, batch_size=1):
        self.bid += 1
        session = tf.get_default_session()

        ret = {}

        opt_name, optimizee = random.choice(list(self.optimizees.items()))

        x = optimizee.get_initial_x(batch_size)
        state = session.run(self.initial_state, feed_dict={self.x: x})

        optimizee_params = optimizee.get_new_params(batch_size)

        losses = []
        fxs = []

        self.log("Optimizee: {}".format(opt_name), verbosity=2, level=2)

        def extract(opt_name, key, pkey='inference'):
            inf = self.ops[opt_name][pkey]

            if hasattr(self, 'devices'):
                inf = list(inf.values())[0]

            return inf[key]

        run_op = [
                self.ops[opt_name]['train_op'],
                extract(opt_name, 0, pkey='losses'),
                extract(opt_name, 'values'),
                extract(opt_name, 'norms'),
                extract(opt_name, 'cosines'),
            ]

        for i in range(n_steps // self.n_bptt_steps):
            feed_dict = optimizee_params
            feed_dict.update({inp: init for inp, init in zip(self.input_state, state)})
            feed_dict.update(optimizee.get_next_dict(self.n_bptt_steps, batch_size))
            feed_dict.update({
                self.train_lr: self.lr,
                self.momentum: self.mu,
            })

            _, state, loss, fx, g_norm, cos_step_grad = session.run(run_op, feed_dict=feed_dict)

            if i == 0:
                #self.log("fx shape: {}".format(np.array(fx).shape), verbosity=2, level=2)
                self.log("First function value: {}".format(fx[0][0]), verbosity=2, level=2)

            losses.append(loss)
            fxs.extend(fx)
            norms.extend(g_norm)
            cosines.extend(cos_step_grad)

        self.log("Last function value: {}".format(fx[-1][0]), verbosity=2, level=2)
        self.log("Loss: {}".format(np.mean(losses / np.log(10))), verbosity=2, level=2)

        ret['optimizee_name'] = opt_name
        ret['loss'] = np.mean(losses)
        ret['fxs'] = fxs

        return ret


    def _fg(self, f, x, i, stop_grad=True):
        fx, g = f(x, i)

        if stop_grad:
            g = tf.stop_gradient(g)

        g_norm = tf.reduce_sum(tf.square(g), axis=-1)
        return fx, g, g_norm


    def restore(self, eid):
        snapshot_path = self.snapshot_path / 'epoch-{}'.format(eid)
        print("Snapshot path: ", snapshot_path)

        self.saver.restore(self.session, str(snapshot_path))
        print(self.name, "restored.")


    def save(self, eid):
        folder = self.snapshot_path
        filename = folder / 'epoch'

        print("Saving to ", filename)
        self.saver.save(self.session, str(filename), global_step=eid)
        print(self.name, "saved.")
