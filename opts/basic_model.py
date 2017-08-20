import pprint
import time
import os
import pathlib
import random
import numpy as np

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
import yellowfin

class BasicModel:
    def __init__(self, name=None, model_path=None, save_tf_data=True, snapshot_path=None, debug=False):
        self.bid = 0
        self.name = name

        self.model_path = model_path or pathlib.Path('models') / name
        self.save_tf_data = save_tf_data
        self.snapshot_path = snapshot_path
        self.debug = debug

    
    @property
    def is_rnnprop(self):
        return self.__class__.__name__ == 'RNNPropOpt'
        

    def build_inputs(self):
        self.x = tf.placeholder(tf.float32, shape=[None], name='basic_model_input')
        return dict(x=self.x)


    def build_initial_state(self):
        return self.input_state


    def build_pre(self):
        self.loglr = tf.get_variable('lr', [], initializer=tf.constant_initializer(0))


    def build(self, optimizees, n_bptt_steps=20,
              loss_type='log', optimizer='adam',
              lambd=0., lambd_l1=0., inference_only=False,
              normalize_lstm_grads=False, grad_clip=1.,
              use_moving_averages=False, stop_grad=True, dynamic=False, **kwargs):

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

            for opt_name, optimizee in optimizees.items():
                with tf.variable_scope('inference_scope'):
                    inference = self.inference(optimizee, self.input_state, n_bptt_steps, stop_grad=stop_grad, dynamic=dynamic)
                    vars_opt |= set(optimizee.vars_)
                    

                losses = self.loss(inference, lambd=lambd, lambd_l1=lambd_l1, loss_type=loss_type)

                ops[opt_name] = {
                    'inference': inference,
                    'losses': losses,
                }

                if not scope.reuse:
                    scope.reuse_variables()
        
        with tf.variable_scope('opt_scope', reuse=False) as scope:
            if not self.is_rnnprop:
                ema = tf.train.ExponentialMovingAverage(decay=0.999)
            self.all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
            self.all_vars = list(set(self.all_vars) - vars_opt)

            if not self.is_rnnprop:
                averages_op = ema.apply(self.all_vars)

            if not inference_only and self.all_vars:
                self.optimizer = self.build_optimizer(optimizer)

                for opt_name in optimizees:
                    losses = ops[opt_name]['losses']

                    grads = self.grads(self.optimizer, losses, normalize_lstm_grads=normalize_lstm_grads, grad_clip=grad_clip)
                    train_op = self.train_op(self.optimizer, grads)

                    with tf.control_dependencies([train_op]):
                        train_op = tf.group(averages_op)

                    ops[opt_name].update({
                        'grads': grads,
                        'train_op': train_op
                    })

                    if not scope.reuse:
                        scope.reuse_variables()

            if self.save_tf_data:
                self.train_writer = tf.summary.FileWriter(str(self.model_path / 'tf_data/train'), self.session.graph)
                self.test_writer = tf.summary.FileWriter(str(self.model_path / 'tf_data/test'), self.session.graph)

            for opt_name in optimizees:
                ops[opt_name]['summaries'] = self.summary(ops[opt_name])

            self.ops = ops

            if use_moving_averages:
                all_vars = {ema.average_name(var): var for var in self.all_vars}
                self.saver = tf.train.Saver(max_to_keep=None, var_list=all_vars, allow_empty=True)
            else:
                self.saver = tf.train.Saver(max_to_keep=None, var_list=self.all_vars, allow_empty=True)


    def inference(self, optimizee, input_state, n_bptt_steps, stop_grad=True, dynamic=False):
        steps_info = []

        state = input_state
        scope = tf.get_variable_scope()
                
        def opt_loss(i, x):
            opt_loss = optimizee.loss(x[None], i)[0]
            print(opt_loss.get_shape())
            return opt_loss

        if dynamic:
            ks, vs = zip(*list(state.items()))
            ks = list(ks)
            vs = list(vs)

            print(ks)

            def dict_to_tuple(d):
                print(d.keys())
                return tuple(d[k] for k in ks)


            def tuple_to_dict(t):
                return dict(zip(ks, t))


            def cond(sid, *loop_vars):
                return tf.less(sid, n_bptt_steps)


            def body(sid, vals, norms, *state):
                state = tuple_to_dict(state)

                value, gradient, gradient_norm = self._fg(optimizee.loss, state['x'], sid)
                if stop_grad:
                    gradient = tf.stop_gradient(gradient)

                new_state = self.step(value, gradient, state)['state']
                new_state = dict_to_tuple(new_state)

                new_vals = tf.concat([vals, tf.expand_dims(value, 0)], axis=0)
                new_norms = tf.concat([norms, tf.expand_dims(gradient_norm, 0)], axis=0)

                out_state = (sid + 1, new_vals, new_norms) + new_state
                
                return out_state


            x = state['x']
            vals_init = tf.zeros([0, tf.shape(x)[0]])
            norms_init = tf.zeros([0, tf.shape(x)[0]])
            state_init = dict_to_tuple(state)

            i = tf.constant(0)
            in_state = (i, vals_init, norms_init) + state_init

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
        else:
            for i in range(n_bptt_steps):
                if self.is_rnnprop:
                    x = state[3]
                    
                    value, gradient, gradient_norm = self._fg(optimizee.loss, x[None], i)
                    if stop_grad:
                        gradient = tf.stop_gradient(gradient)

                    step_info = self.step(opt_loss, i, state)
                else:
                    x = state['x']
                    value, gradient, gradient_norm = self._fg(optimizee.loss, x, i)
                    if stop_grad:
                        gradient = tf.stop_gradient(gradient)

                    step_info = self.step(value, gradient, state)
                    state = step_info['state']

                if not scope.reuse:
                    scope.reuse_variables()

                step_info['value'] = value
                step_info['gradient'] = gradient
                step_info['gradient_norm'] = gradient_norm

                steps_info.append(step_info)

            final_state = state

        ret = {
            'values': [info['value'] for info in steps_info],
            'norms': [info['gradient_norm'] for info in steps_info],
            'final_state': final_state
        }

        first_step = steps_info[0]
        if not self.is_rnnprop:
            keys = set(first_step.keys())

        if not self.is_rnnprop and 'state' in keys:
            ret['states'] = [info['state'] for info in steps_info]
            state_keys = set(first_step['state'].keys())
        
            if not self.is_rnnprop and 'loglr' in state_keys:
                ret['lrs'] = [info['state']['loglr'] for info in steps_info]

        if not self.is_rnnprop and 'cos_step_adam' in keys:
            ret['cosines'] = [info['cos_step_adam'] for info in steps_info]

        return ret


    def loss(self, inference, lambd=0., lambd_l1=0., loss_type='log'):
        values = tf.stack(inference['values'])
        try:
            states = tf.stack([s['loglr'] for s in inference['states']])
        except:
            states = None

        losses = []

        if loss_type == 'log':
            loss = tf.reduce_mean(tf.log(values + 1e-8) - tf.log(values[:1] + 1e-8))

            if states is not None:
                lr_loss = -lambd * tf.reduce_mean(states - states[:1])
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

        if tf.trainable_variables():
            reg_loss = tf.add_n([
                lambd_l1 * tf.norm(v, ord=1)
                for v in tf.trainable_variables()
                if 'bias' not in v.name
            ])
            losses.append(reg_loss)

        return [loss] + losses


    def build_optimizer(self, optimizer_type):
        self.train_lr = tf.placeholder(tf.float32, shape=[], name='train_lr')
        self.momentum = tf.placeholder(tf.float32, shape=[], name='momentum')

        if optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(self.train_lr, beta1=self.momentum)
        elif optimizer_type == 'momentum':
            optimizer = tf.train.MomentumOptimizer(self.train_lr,
                                                   self.momentum,
                                                   use_nesterov=True)
        elif optimizer_type == 'yellowfin':
            optimizer = yellowfin.YFOptimizer(self.train_lr)
        else:
            raise ValueError("Unknown optimizer: {}".format(optimizer_type))

        return optimizer


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


    def summary(self, ops):
        #pylint: disable=unused-argument
        summaries = []
        return summaries

            # loglr_mean, loglr_std = tf.nn.moments(self.loglr, axes=[0])
            # lr_mean, lr_std = tf.nn.moments(tf.exp(self.loglr), axes=[0])

            #for opt_name in self.optimizees:

                #self.summaries[opt_name].append(
                #    tf.summary.scalar('train_loss', tf.reduce_mean(self.loss[opt_name])))
                #self.summaries[opt_name].append(
                #    tf.summary.scalar('function_value', tf.reduce_mean(self.fxs[opt_name][-1])))

                #loglr_mean, loglr_std = tf.nn.moments(self.states[opt_name][-1][-1], axes=[0])
                #lr_mean, lr_std = tf.nn.moments(tf.exp(self.states[opt_name][-1][-1]), axes=[0])

                #self.summaries[opt_name].append(
                #    tf.summary.scalar('log_learning_rate_mean', loglr_mean))
                #self.summaries[opt_name].append(
                #    tf.summary.scalar('log_learning_rate_std', loglr_std))
                #self.summaries[opt_name].append(
                #   tf.summary.scalar('learning_rate_mean', lr_mean))
                #self.summaries[opt_name].append(
                #   tf.summary.scalar('learning_rate_std', lr_std))


            #for grad, var in gradients:
            #    if grad is not None:
            #        self.summaries[opt_name].append(
            #          tf.summary.histogram(var.op.name + '/{}/gradients'.format(opt_name), grad))

            #        ratio_name = '/{}/grad_to_var_ratio'.format(opt_name)
            #        ratio = tf.norm(grad) / (tf.norm(var) + 1e-8)
            #        self.summaries[opt_name].append(
            #    tf.summary.histogram(var.op.name + ratio_name, ratio))

        #for var in tf.trainable_variables():
        #    self.summaries[opt_name].append(tf.summary.histogram(var.op.name, var))



    def log(self, message, verbosity, level=0):
        if verbosity <= self.verbose:
            message = '\t' * level + message
            print(message)


    def test(self, eid, n_batches, n_steps=20, opt_name=None, verbose=1, include_x=False):
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
            ret = self.test_one_iteration(n_steps, opt_name, include_x=include_x)
            batch_time = time.time() - batch_time
            self.log("Time: {}".format(batch_time), verbosity=1, level=1)
            rets.append(ret)

        return rets


    def test_one_iteration(self, n_steps, opt_name, include_x=False):
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
            'summaries': self.ops[opt_name]['summaries'],
            'values': inf['values'],
            'norms': inf['norms'],
            'final_state': inf['final_state']
        }

        if inf.get('states') is not None:
            run_op['states'] = inf['states']

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

        if include_x and not self.is_rnnprop:
            run_op['x'] = [info['x'] for info in inf['states']]
            steps_info['x'] = []

        losses = []

        for _ in range(n_steps // self.n_bptt_steps):
            feed_dict = optimizee_params
            if self.is_rnnprop:
                feed_dict.update({inp: init for inp, init in zip(self.input_state, state)})
            else:
                feed_dict.update({inp: state[name] for name, inp in self.input_state.items()})
            feed_dict.update(optimizee.get_next_dict(self.n_bptt_steps))
 
            info = session.run(run_op, feed_dict=feed_dict)
            #state = info['states'][-1]
            state = info['final_state']
            summaries_str = info['summaries']

            losses.append(info['loss'])

            for k in steps_info:
                steps_info[k].extend(info[k])

        self.log("Loss: {}".format(np.mean(losses / np.log(10))), verbosity=2, level=2)

        ret['optimizee_name'] = opt_name
        ret['loss'] = np.nanmean(losses)

        for k in steps_info:
            ret[k] = np.array(steps_info[k])

        if self.save_tf_data:
            for summary_str in summaries_str:
                self.test_writer.add_summary(summary_str, self.bid)
            self.test_writer.flush()

        return ret


    def train(self, n_epochs, n_batches,
              batch_size=100, n_steps=20,
              train_lr=1e-2, momentum=0.9,
              eid=0, test=True, verbose=1):
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
                extract(opt_name, 'states'),
                extract(opt_name, 0, pkey='losses'),
                extract(opt_name, 'values'),
                extract(opt_name, 'norms'),
                extract(opt_name, 'cosines'),
                self.ops[opt_name]['summaries']
            ]

        for i in range(n_steps // self.n_bptt_steps):
            feed_dict = optimizee_params
            feed_dict.update({inp: init for inp, init in zip(self.input_state, state)})
            feed_dict.update(optimizee.get_next_dict(self.n_bptt_steps, batch_size))
            feed_dict.update({
                self.train_lr: self.lr,
                self.momentum: self.mu,
            })

            _, state, loss, fx, g_norm, cos_step_grad, summaries_str = session.run(run_op, feed_dict=feed_dict)

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
        #ret['loss'] = np.mean(losses)
        ret['loss'] = np.nanmean(losses)
        ret['fxs'] = fxs

        if self.save_tf_data:
            for summary_str in summaries_str:
                self.train_writer.add_summary(summary_str, self.bid)
            self.train_writer.flush()

        return ret


    def _fg(self, f, x, i):
        fx, g = f(x, i)
        g_norm = tf.reduce_sum(tf.square(g), axis=-1)
        return fx, g, g_norm


    def step(self, f, i, state):
        x = state['x']

        value, gradient, gradient_norm = self._fg(f, x, i)

        x -= tf.exp(self.loglr) * gradient
        new_state = dict(x=x)
        
        return {
            'state': new_state,
            'value': value,
            'gradient': gradient,
            'gradient_norm': gradient_norm
        }


    def restore(self, eid):
        snapshot_path = self.snapshot_path / 'epoch-{}'.format(eid)
        print("Snapshot path: ", snapshot_path)

        #self.saver = tf.train.import_meta_graph(str(snapshot_path) + '.meta')
        self.saver.restore(self.session, str(snapshot_path))
        print(self.name, "restored.")


    def save(self, eid):
        folder = self.snapshot_path
        filename = folder / 'epoch'
        sfilename = folder / 'epoch-last'

        print("Saving to ", filename)

        self.saver.save(self.session, str(filename), global_step=eid)
        #os.unlink("{}-{}.meta".format(filename, eid))
        if os.path.lexists(str(sfilename)):
            os.unlink(str(sfilename))
        os.symlink("epoch-{}".format(eid), str(sfilename))
        print(self.name, "saved.")
