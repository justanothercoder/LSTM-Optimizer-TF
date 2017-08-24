from collections import namedtuple
import time
import random
import numpy as np

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple

from .cells import LSTMOptCell, OptFuncCell
import util
from problem_producer import ProblemProducer
from inference import static_inference, dynamic_inference, cell_inference


RNNOptState = namedtuple('RNNOptState', ['x', 'rnn_state'])

BuildConfig = util.namedtuple_with_defaults(
    'BuildConfig', [
        ('n_bptt_steps', 20),
        ('loss_type', 'log'),
        ('lambd', 0.),
        ('lambd_l1', 0.),
        ('inference_only', False),
        ('normalize_lstm_grads', False),
        ('grad_clip', 1.),
        ('stop_grad', True),
        ('dynamic', False),
        ('cell', False)
    ])


class BasicModel:
    def __init__(self, name=None, snapshot_path=None, debug=False):
        self.bid = 0
        self.name = name
        self.snapshot_path = snapshot_path
        self.debug = debug

    
    def build(self, optimizees, build_config=BuildConfig(), **kwargs):
        self.optimizees = optimizees
        self.config = build_config
        ops = {}

        self.kwargs = kwargs
        vars_opt = set()

        with tf.variable_scope('opt_scope') as scope:
            self.scope = scope
            self.build_pre()
            
            self.x = tf.placeholder(tf.float32, shape=[None, None], name='theta')
            
            if self.config.cell:
                self.cell = LSTMOptCell(self.init_config)
                self.cell.kwargs = kwargs
                self.input_state = RNNOptState(self.x, self.cell.zero_state(self.x))
            elif self.is_rnnprop:
                self.input_state = self.build_inputs()
                self.initial_state = self.build_initial_state(self.x)
            else:
                self.input_state = self.build_inputs()
                self.initial_state = self.build_initial_state(self.x)
                self.initial_state = RNNOptState(self.x, self.initial_state)

                self.input_state = RNNOptState(self.x, self.input_state)

            reuse = False

            for opt_name, optimizee in optimizees.items():
                with tf.variable_scope('inference_scope', reuse=reuse) as self.inf_scope:
                    inference = self.inference(optimizee, self.input_state)
                    vars_opt |= set(optimizee.vars_)
        
                ops[opt_name] = dict(inference=inference)

                if not reuse:
                    reuse = True
            
            self.all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.inf_scope.name)
            self.all_vars = list(set(self.all_vars) - vars_opt)

            for opt_name, optimizee in optimizees.items():
                inference = ops[opt_name]['inference']
                ops[opt_name]['losses'] = self.loss(inference)

        with tf.variable_scope('opt_scope', reuse=False) as scope:

            if not self.config.inference_only and self.all_vars:
                self.train_lr = tf.placeholder(tf.float32, shape=[], name='train_lr')
                self.momentum = tf.placeholder(tf.float32, shape=[], name='momentum')
                self.optimizer = tf.train.AdamOptimizer(self.train_lr, beta1=self.momentum)

                for opt_name in optimizees:
                    losses = ops[opt_name]['losses']

                    grads = self.grads(self.optimizer, losses)
                    train_op = self.train_op(self.optimizer, grads)

                    ops[opt_name].update({
                        'grads': grads,
                        'train_op': train_op
                    })

                    if not scope.reuse:
                        scope.reuse_variables()

            self.ops = ops
            self.saver = tf.train.Saver(max_to_keep=None, var_list=self.all_vars, allow_empty=True)


    def step_with_func(self, f, i, state, stop_grad=True):
        value, gradient = f(state.x, i)
        if stop_grad:
            gradient = tf.stop_gradient(gradient)

        gradient_norm = tf.reduce_sum(tf.square(gradient), axis=-1)

        if self.config.cell:
            step, rnn_state = self.cell(gradient, state.rnn_state)
        else:
            step, rnn_state = self.step(value, gradient, state.rnn_state)
        state = state._replace(x=state.x + step, rnn_state=rnn_state)

        ret = dict(value=value, gradient_norm=gradient_norm, state=state)

        if 'loglr' in rnn_state._fields:
            ret['loglr'] = tf.reshape(rnn_state.loglr, tf.shape(state.x))

        return ret


    def inference(self, optimizee, input_state):
        Model = namedtuple('Model', ['input_state', 'inference_scope', 'step_with_func', 'config'])
        model = Model(input_state, self.inf_scope, self.step_with_func, self.config)

        if self.config.cell:
            Model = namedtuple('Model', ['input_state', 'inference_scope', 'step_with_func', 'config', 'cell'])
            model = Model(input_state, self.inf_scope, self.step_with_func, self.config, self.cell)
            ret = cell_inference(model, optimizee)
        else:
            if self.config.dynamic:
                ret = dynamic_inference(model, optimizee)
            else:
                ret = static_inference(model, optimizee)

        # first_step = steps_info[0]
        # if not self.is_rnnprop:
        #     keys = set(first_step.keys())

        #     if 'state' in keys:
        #         state_keys = set(first_step['state'].keys())
        #     
        #         if 'loglr' in state_keys:
        #             ret['lrs'] = [info['state']['loglr'] for info in steps_info]

        #     if 'cos_step_adam' in keys:
        #         ret['cosines'] = [info['cos_step_adam'] for info in steps_info]

        return ret


    def loss(self, inference):
        values = tf.stack(inference['values'])
        if 'lrs' in inference:
            lrs = tf.stack(inference['lrs'])
        else:
            lrs = None

        losses = []

        if self.config.loss_type == 'log':
            loss = tf.reduce_mean(tf.log(values + 1e-8) - tf.log(values[:1] + 1e-8))

            if lrs is not None:
                lr_loss = -self.config.lambd * tf.reduce_mean(lrs - lrs[:1])
                losses.append(lr_loss)

        elif self.config.loss_type == 'log_smooth':
            smooth_vals = []
            for i in range(self.config.n_bptt_steps):
                if i == 0:
                    smooth_val = values[i]
                else:
                    smooth_val = 0.95 * smooth_val + 0.05 * values[i]
                smooth_vals.append(smooth_val)

            smooth_vals = tf.stack(smooth_vals)
            loss = tf.reduce_mean(tf.log(smooth_vals + 1e-8) - tf.log(smooth_vals[:1] + 1e-8))

        elif self.config.loss_type == 'sum':
            loss = tf.reduce_mean(values)
        else:
            loss = values[-1]

        if self.all_vars:
            weights = [v for v in self.all_vars if 'bias' not in v.name]
            reg_loss = tf.add_n([self.config.lambd_l1 * tf.norm(v, ord=1) for v in weights])
            losses.append(reg_loss)

        return [loss] + losses


    def grads(self, optimizer, losses):
        loss = tf.add_n(losses)
        grads = optimizer.compute_gradients(loss, var_list=self.all_vars)

        if self.config.normalize_lstm_grads:
            print("Using normalized meta-grads")
            norm = tf.global_norm(grads)
            grads = [(grad / (norm + 1e-8), var) for grad, var in grads]

        grads, _ = tf.clip_by_global_norm([g for g, _ in grads], self.config.grad_clip)
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


    def test(self, eid, n_batches, n_steps=20, opt_name=None, verbose=1, session=None):
        self.session = session or tf.get_default_session()
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

        optimizee = self.optimizees[opt_name]
        x, optimizee_params = optimizee.sample_problem()

        inf = self.ops[opt_name]['inference']
        losses = self.ops[opt_name]['losses']

        if hasattr(self, 'devices'):
            inf = list(inf.values())[0]
            losses = list(losses.values())[0]

        run_op = {
            'loss': losses[0],
            'values': inf['values'],
            'norms': inf['norms'],
            'final_state': inf['final_state'],
        }

        steps_info = dict(values=[], norms=[])

        #if inf.get('cosines'):
        #    run_op['cosines'] = inf['cosines']
        #    steps_info['cosines'] = []

        if inf.get('lrs'):
            run_op['lrs'] = inf['lrs']
            steps_info['lrs'] = []

        losses = []

        input_state = self.input_state

        state = self.get_init_state(x)
        for _ in range(n_steps // self.config.n_bptt_steps):
            feed_dict = self.get_feed_dict(input_state, state, optimizee_params, optimizee)
            info = self.session.run(run_op, feed_dict=feed_dict)

            state = info['final_state']
            losses.append(info['loss'])

            for k in steps_info:
                steps_info[k].extend(info[k])

        self.log("Loss: {}".format(np.mean(losses / np.log(10))), verbosity=2, level=2)

        ret = dict(optimizee_name=opt_name, loss=np.mean(losses))
        for k in steps_info:
            ret[k] = np.array(steps_info[k])

        return ret

    
    def get_init_state(self, x):
        if self.config.cell:
            state = self.session.run(self.input_state, {self.x: x})
        elif self.is_rnnprop:
            state = self.session.run(self.initial_state, feed_dict={self.opt.x: x[0]})
        else:
            state = self.session.run(self.initial_state, feed_dict={self.x: x})

        return state


    def get_feed_dict(self, input_state, state, params, opt, batch_size=1):
        feed_dict = dict(zip(input_state, state))
        feed_dict.update(params)
        feed_dict.update(opt.get_next_dict(self.config.n_bptt_steps, batch_size))
        #if self.is_rnnprop:
        #    feed_dict = {self.opt.x: state.x[0]}
        return feed_dict


    #def train(self, n_epochs, n_batches, batch_size=100, n_steps=20, train_lr=1e-2, momentum=0.9, eid=0, test=True, verbose=1):
    #    self.verbose = verbose
    #    self.lr = train_lr
    #    self.mu = momentum

    #    if eid > 0:
    #        self.restore(eid)
    #        self.bid = eid * n_batches

    #    train_rets = []
    #    test_rets = []

    #    sample_steps = bool(n_steps == 0)

    #    try:
    #        for epoch in range(eid, n_epochs):
    #            self.log("Epoch: {}".format(epoch), verbosity=1, level=0)
    #            epoch_time = time.time()

    #            loss = None

    #            for batch in range(n_batches):
    #                self.log("Batch: {}".format(batch), verbosity=2, level=1)
    #                if sample_steps:
    #                    #n_steps = int(np.random.exponential(scale=200)) + 50

    #                    exp_scale = min(50, epoch)
    #                    n_steps = int(np.random.exponential(scale=exp_scale)) + 1
    #                    n_steps *= self.n_bptt_steps
    #                    self.log("n_steps: {}".format(n_steps), verbosity=2, level=2)

    #                batch_time = time.time()
    #                ret = self.train_one_iteration(n_steps, batch_size)

    #                if np.isnan(ret['loss']):
    #                    print("Loss is NaN")
    #                    #print(ret['fxs'])

    #                if loss is not None:
    #                    loss = 0.9 * loss + 0.1 * ret['loss']
    #                else:
    #                    loss = ret['loss']
    #                batch_time = time.time() - batch_time

    #                self.log("Batch time: {}".format(batch_time), verbosity=2, level=2)
    #                train_rets.append(ret)

    #            self.log("Epoch time: {}".format(time.time() - epoch_time), verbosity=1, level=1)
    #            self.log("Epoch loss: {}".format(loss / np.log(10) / self.n_bptt_steps), verbosity=1, level=1)

    #            if test and (epoch + 1) % 10 == 0:
    #                self.save(epoch + 1)

    #                opt_name = random.choice(list(self.optimizees.keys()))

    #                self.log("Test epoch: {}".format(epoch), verbosity=1, level=0)
    #                test_epoch_time = time.time()

    #                for batch in range(n_batches):
    #                    self.log("Test batch: {}".format(batch), verbosity=2, level=1)

    #                    batch_time = time.time()
    #                    ret = self.test_one_iteration(n_steps, opt_name)
    #                    batch_time = time.time() - batch_time
    #                    self.log("Batch time: {}".format(batch_time), verbosity=2, level=2)

    #                    test_rets.append(ret)

    #                test_epoch_time = time.time() - test_epoch_time
    #                self.log("Epoch time: {}".format(test_epoch_time), verbosity=1, level=1)
    #    except KeyboardInterrupt:
    #        print("Stopped training early")

    #    return train_rets, test_rets


    #def train_one_iteration(self, n_steps, batch_size=1):
    #    self.bid += 1
    #    session = tf.get_default_session()

    #    ret = {}

    #    opt_name, optimizee = random.choice(list(self.optimizees.items()))

    #    x = optimizee.get_initial_x(batch_size)
    #    state = session.run(self.initial_state, feed_dict={self.x: x})

    #    optimizee_params = optimizee.get_new_params(batch_size)

    #    losses = []
    #    fxs = []

    #    self.log("Optimizee: {}".format(opt_name), verbosity=2, level=2)

    #    def extract(opt_name, key, pkey='inference'):
    #        inf = self.ops[opt_name][pkey]

    #        if hasattr(self, 'devices'):
    #            inf = list(inf.values())[0]

    #        return inf[key]

    #    run_op = [
    #            self.ops[opt_name]['train_op'],
    #            extract(opt_name, 0, pkey='losses'),
    #            extract(opt_name, 'values'),
    #            extract(opt_name, 'norms'),
    #            extract(opt_name, 'cosines'),
    #        ]

    #    for i in range(n_steps // self.n_bptt_steps):
    #        feed_dict = optimizee_params
    #        feed_dict.update({inp: init for inp, init in zip(self.input_state, state)})
    #        feed_dict.update(optimizee.get_next_dict(self.n_bptt_steps, batch_size))
    #        feed_dict.update({
    #            self.train_lr: self.lr,
    #            self.momentum: self.mu,
    #        })
    #        _, state, loss, fx, g_norm, cos_step_grad = session.run(run_op, feed_dict=feed_dict)

    #        if i == 0:
    #            #self.log("fx shape: {}".format(np.array(fx).shape), verbosity=2, level=2)
    #            self.log("First function value: {}".format(fx[0][0]), verbosity=2, level=2)

    #        losses.append(loss)
    #        fxs.extend(fx)
    #        norms.extend(g_norm)
    #        cosines.extend(cos_step_grad)

    #    self.log("Last function value: {}".format(fx[-1][0]), verbosity=2, level=2)
    #    self.log("Loss: {}".format(np.mean(losses / np.log(10))), verbosity=2, level=2)

    #    ret['optimizee_name'] = opt_name
    #    ret['loss'] = np.mean(losses)
    #    ret['fxs'] = fxs

    #    return ret
    
    
    def train(self, n_epochs, n_batches, batch_size=100, n_steps=100, eid=0, test=True, seed=None, session=None):
        self.session = session or tf.get_default_session() 
        if eid > 0:
            self.restore(eid)

        epoch_losses = []
        problem_producer = ProblemProducer(self.optimizees, seed=seed)

        for epoch in range(eid, n_epochs):
            epoch_start = time.time()
            batch_losses = []
            for problem in problem_producer.sample_sequence(n_batches, batch_size):
                batch_start = time.time()

                ops = self.ops[problem.name]
                inf = ops['inference']

                losses = ops['losses']

                if hasattr(self, 'devices'):
                    inf = list(inf.values())[0]
                    losses = list(losses.values())[0]


                #input_state = inf['cell'].zero_state(tf.size(self.x))
                #input_state = inf['istate']
                state = self.session.run(self.input_state, {self.x: problem.init})

                losses_ = []
                n_unrolls = n_steps // self.config.n_bptt_steps
                for _ in range(n_unrolls):
                    feed_dict = dict(zip(input_state, state))
                    feed_dict.update(problem.params)
                    feed_dict.update(problem.optim.get_next_dict(self.config.n_bptt_steps, batch_size))
                    feed_dict.update({self.x: problem.init, self.train_lr: 1e-4, self.momentum: 0.9})
                    
                    loss, state, _ = self.session.run([losses, inf['final_state'], ops['train_op']], feed_dict=feed_dict)
                    losses_.append(loss)

                batch_loss = np.mean(losses_)
                batch_time = time.time() - batch_start
                batch_losses.append(batch_loss)

            epoch_loss = np.mean(batch_losses)
            epoch_losses.extend(batch_losses)

            train_loss = np.mean(epoch_losses)
            epoch_time = time.time() - epoch_start
            print("Epoch: {}, Train loss: {}, Time: {}".format(epoch, train_loss, epoch_time))

            if (epoch + 1) % 5 == 0:
                self.save(epoch + 1)

        return epoch_losses


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
    
    
    @property
    def is_rnnprop(self):
        return self.__class__.__name__ == 'RNNPropOpt'
        

    def build_inputs(self):
        raise NotImplementedError


    def build_initial_state(self):
        raise NotImplementedError


    def build_pre(self):
        raise NotImplementedError


    def step(self, g, state):
        if self.config.cell:
            return self.cell(g, state)
        else:
            raise NotImplementedError
