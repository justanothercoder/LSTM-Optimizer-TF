"""
    This module defines basic model which defines structure of optimizer.
"""

import time
import os
import pathlib
import random
import numpy as np

import tensorflow as tf
import yellowfin


class BasicModel:
    """This class defines basic model."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, name=None, model_path=None, save_tf_data=True, snapshot_path=None):
        self.bid = 0
        self.name = name

        self.model_path = model_path or pathlib.Path('models') / name
        self.save_tf_data = save_tf_data
        self.snapshot_path = snapshot_path


    def log(self, message, verbosity, level=0):
        """logs message"""
        if verbosity <= self.verbose:
            message = '\t' * level + message
            print(message)


    def test(self, eid, n_batches, n_steps=20, opt_name=None, verbose=1):
        #pylint: disable=too-many-arguments
        """Runs testing"""
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
        #pylint: disable=too-many-locals
        """Runs test on one batch"""
        self.bid += 1
        session = tf.get_default_session()

        ret = {}

        optimizee = self.optimizees[opt_name]

        x = optimizee.get_initial_x()
        state = session.run(self.initial_state, feed_dict={self.x: x})

        optimizee_params = optimizee.get_new_params()

        losses = []
        fxs = []
        lrs = []
        norms = []

        dev = self.devices[0]

        for _ in range(n_steps // self.n_bptt_steps):
            feed_dict = optimizee_params
            feed_dict.update({inp: init for inp, init in zip(self.input_state, state)})
            feed_dict.update(optimizee.get_next_dict(self.n_bptt_steps))

            states, loss, fx, g_norm, summaries_str = session.run([
                self.states[opt_name][dev],
                self.loss[opt_name][dev],
                self.fxs[opt_name][dev],
                self.norms[opt_name][dev],
                self.summaries[opt_name]
            ], feed_dict=feed_dict)

            state = states[-1]

            losses.append(loss)
            fxs.extend(fx)
            lrs.extend([s[-1] for s in states])
            norms.extend(g_norm)

        self.log("Loss: {}".format(np.mean(losses / np.log(10))), verbosity=2, level=2)

        ret['optimizee_name'] = opt_name
        ret['loss'] = np.mean(losses)
        ret['fxs'] = np.array(fxs)
        ret['lrs'] = np.array(lrs).mean(axis=1)
        ret['norms'] = np.array(norms)

        if self.save_tf_data:
            for summary_str in summaries_str:
                self.test_writer.add_summary(summary_str, self.bid)
            self.test_writer.flush()

        return ret


    def train(self, n_epochs, n_batches,
              batch_size=100, n_steps=20,
              train_lr=1e-2, momentum=0.9,
              eid=0, test=True, verbose=1):
        #pylint: disable=too-many-arguments
        #pylint: disable=too-many-locals
        """Runs training."""
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
                        print(ret['fxs'])

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
        #pylint: disable=too-many-locals
        """Runs training on one batch"""
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

        dev = self.devices[0]

        for i in range(n_steps // self.n_bptt_steps):
            feed_dict = optimizee_params
            feed_dict.update({inp: init for inp, init in zip(self.input_state, state)})
            feed_dict.update(optimizee.get_next_dict(self.n_bptt_steps, batch_size))
            feed_dict.update({
                self.train_lr: self.lr,
                self.momentum: self.mu,
            })

            #_, state, loss, fx, summaries_str = session.run([
            #    self.apply_gradients[opt_name],
            #    self.states[opt_name][-1],
            #    self.loss[opt_name], self.fxs[opt_name], self.summaries[opt_name]
            #], feed_dict=feed_dict)
            _, state, loss, fx, summaries_str = session.run([
                self.apply_gradients[opt_name],
                self.states[opt_name][dev][-1],
                self.loss[opt_name][dev],
                self.fxs[opt_name][dev],
                self.summaries[opt_name]
            ], feed_dict=feed_dict)

            if i == 0:
                #self.log("fx shape: {}".format(np.array(fx).shape), verbosity=2, level=2)
                self.log("First function value: {}".format(fx[0][0]), verbosity=2, level=2)

            losses.append(loss)
            fxs.extend(fx)

        self.log("Last function value: {}".format(fx[-1][0]), verbosity=2, level=2)
        self.log("Loss: {}".format(np.mean(losses / np.log(10))), verbosity=2, level=2)

        ret['optimizee_name'] = opt_name
        ret['loss'] = np.mean(losses)
        ret['fxs'] = fxs

        if self.save_tf_data:
            for summary_str in summaries_str:
                self.train_writer.add_summary(summary_str, self.bid)
            self.train_writer.flush()

        return ret


    def build(self, optimizees, n_bptt_steps=20,
              loss_type='log', optimizer='adam',
              lambd=0, inference_only=False, devices=None):
        """Builds model"""
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-arguments
        self.devices = devices or ['/cpu:0']

        self.optimizees = optimizees
        self.n_bptt_steps = n_bptt_steps
        self.loss_type = loss_type
        self.optimizer_type = optimizer
        self.lambd = lambd

        self.session = tf.get_default_session()

        with tf.variable_scope('opt_global_vscope') as scope:
            if self.save_tf_data:
                tf_data_path = self.model_path / 'tf_data'
                train_path = str(tf_data_path / 'train')
                test_path = str(tf_data_path / 'test')

                self.train_writer = tf.summary.FileWriter(train_path, self.session.graph)
                self.test_writer = tf.summary.FileWriter(test_path, self.session.graph)
            self.summaries = {name: [] for name in self.optimizees}

            self._build_pre()
            self._build_input()
            self._build_initial_state()

            self._build_loop()
            self._build_loss()

            self.all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

            if not inference_only and self.all_vars:
                self._build_optimizer()

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

            self.saver = tf.train.Saver(max_to_keep=None, var_list=self.all_vars, allow_empty=True)


    def _build_pre(self):
        self.loglr = tf.get_variable('lr', [], initializer=tf.constant_initializer(0))


    def _build_loop(self):
        self.fxs = {}
        self.states = {}
        self.norms = {}

        with tf.variable_scope('loop_scope') as self.loop_scope:
            reused = False

            for opt_name, optimizee in self.optimizees.items():
                self.fxs[opt_name] = {}
                self.states[opt_name] = {}
                self.norms[opt_name] = {}
                for dev in self.devices:
                    with tf.device(dev):
                        states = []
                        fxs = []
                        norms = []

                        state = self.input_state

                        for i in range(self.n_bptt_steps):
                            state, f, g_norm = self._iter(optimizee.loss, i, state)

                            fxs.append(f)
                            states.append(state)
                            norms.append(g_norm)

                            if not reused:
                                self.loop_scope.reuse_variables()
                                reused = True

                        self.fxs[opt_name][dev] = fxs
                        self.states[opt_name][dev] = states
                        self.norms[opt_name][dev] = norms


    def _build_loss(self):
        self.loss = {}

        for opt_name in self.optimizees:
            self.loss[opt_name] = {}
            for dev in self.devices:
                with tf.device(dev):
                    fxs = self.fxs[opt_name][dev]
                    states = self.states[opt_name][dev]

                    fxs = tf.stack(fxs)
                    states = tf.stack([s[-1] for s in states])

                    if self.loss_type == 'log':
                        loss = tf.reduce_mean(tf.reduce_sum(tf.log(fxs) - tf.log(fxs[:1]), axis=0))
                        lr_loss = -self.lambd * tf.reduce_mean(states - states[:1])

                        loss += lr_loss
                    elif self.loss_type == 'sum':
                        loss = tf.reduce_mean(fxs)
                    else:
                        loss = fxs[-1]

                    self.loss[opt_name][dev] = loss


    def _fg(self, f, x, i):
        fx, g = f(x, i)
        #g = gradients.gradients(fx, x)[0]
        #g = gradients.gradients(tf.reduce_sum(fx), x)[0]
        g_norm = tf.reduce_sum(g**2, axis=-1)
        return fx, g, g_norm


    def _iter(self, f, i, state):
        x, = state

        fx, g, g_norm = self._fg(f, x, i)

        x -= tf.exp(self.loglr) * g
        return [x], fx, g_norm


    def _build_input(self):
        self.x = tf.placeholder(tf.float32, shape=[None], name='basic_model_input')
        self.input_state = [self.x]


    def _build_initial_state(self):
        self.initial_state = [self.x]


    def _build_optimizer(self):
        self.apply_gradients = {}

        self.train_lr = tf.placeholder(tf.float32, shape=[], name='train_lr')
        self.momentum = tf.placeholder(tf.float32, shape=[], name='momentum')

        if self.optimizer_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.train_lr, beta1=self.momentum)
        elif self.optimizer_type == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(self.train_lr,
                                                        self.momentum,
                                                        use_nesterov=True)
        elif self.optimizer_type == 'yellowfin':
            self.optimizer = yellowfin.YFOptimizer(self.train_lr)
        else:
            raise ValueError("Unknown optimizer: {}".format(self.optimizer_type))

        for opt_name in self.optimizees:
            tower_grads = []
            for dev in self.devices:
                with tf.device(dev):
                    gradients = self.optimizer.compute_gradients(
                        self.loss[opt_name][dev],
                        var_list=self.all_vars)
                    tower_grads.append(gradients)

            average_grads = []
            for grad_and_vars in zip(*tower_grads):
                grad = tf.stack([g for g, _ in grad_and_vars], axis=0)
                grad = tf.reduce_mean(grad, axis=0)

                v = grad_and_vars[0][1]
                average_grads.append((grad, v))

            #self.apply_gradients[opt_name] = self.optimizer.apply_gradients(gradients)
            self.apply_gradients[opt_name] = self.optimizer.apply_gradients(average_grads)

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


    def restore(self, eid):
        snapshot_path = self.snapshot_path / 'epoch-{}'.format(eid)
        print("Snapshot path: ", snapshot_path)
        self.saver.restore(self.session, str(snapshot_path))
        print(self.name, "restored.")


    def save(self, eid):
        folder = self.snapshot_path
        filename = folder / 'epoch'
        sfilename = folder / 'epoch-last'

        print("Saving to ", filename)

        self.saver.save(self.session, str(filename), global_step=eid)
        os.unlink("{}-{}.meta".format(filename, eid))
        if os.path.lexists(str(sfilename)):
            os.unlink(str(sfilename))
        os.symlink("epoch-{}".format(eid), str(sfilename))
        print(self.name, "saved.")
