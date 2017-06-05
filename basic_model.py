import os
import random
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import gradients

class BasicModel:
    def __init__(self, optimizees, train_lr=1e-2, n_bptt_steps=20, loss_type='log', name=None):
        self.optimizees = optimizees
        self.train_lr = train_lr
        self.n_bptt_steps = n_bptt_steps
        self.loss_type = loss_type

        self.session = tf.get_default_session()
        self.bid = 0
        self.name = name


    def test(self, eid, n_batches, n_steps=20):
        self.restore(eid)

        rets = []

        for batch in range(n_batches):
            ret = self.test_one_iteration(n_steps)
            rets.append(ret)

            print("\tBatch: {}".format(batch))

        return rets


    def test_one_iteration(self, n_steps):
        self.bid += 1
        session = tf.get_default_session()

        ret = {}

        opt_name, optimizee = random.choice(list(self.optimizees.items()))

        x = optimizee.get_initial_x()
        state = session.run(self.initial_state, feed_dict={self.x: x})

        optimizee_params = optimizee.get_new_params()

        losses = []
        fxs    = []
        lrs    = []
        norms  = []

        for i in range(n_steps // self.n_bptt_steps):
            feed_dict = optimizee_params
            feed_dict.update({inp: init for inp, init in zip(self.input_state, state)})
            feed_dict.update(optimizee.get_next_dict(self.n_bptt_steps))

            states, loss, fx, g_norm, summaries_str = session.run([
                self.states[opt_name], self.loss[opt_name], self.losses[opt_name], self.norms[opt_name], self.summaries[opt_name]
            ], feed_dict=feed_dict)

            state = states[-1]
            
            if i == 0:
                print("\t\tFirst function value: {}".format(fx[0]))

            losses.append(loss)
            fxs.extend(fx)
            lrs.extend([s[-1] for s in states])
            norms.extend(g_norm)

        print("\t\tLoss: {}".format(np.mean(losses)))
        print("\t\tLast function value: {}".format(fx[-1]))

        ret['optimizee_name'] = opt_name
        ret['loss']  = np.sum(losses)
        ret['fxs']   = np.array(fxs)
        ret['lrs']   = np.array(lrs).mean(axis=1)
        ret['norms'] = np.array(norms)

        for summary_str in summaries_str:
            self.test_writer.add_summary(summary_str, self.bid)
        self.test_writer.flush()

        return ret


    def train(self, n_epochs, n_batches, batch_size=1, n_steps=20, eid=0):
        if eid > 0:
            self.restore(eid)
            self.bid = eid * n_batches 

        for epoch in range(eid, n_epochs):
            print("Epoch: {}".format(epoch))
            for batch in range(n_batches):
                ret = self.train_one_iteration(n_steps)
                print("\tBatch: {}".format(batch))

            if (epoch + 1) % 10 == 0:
                self.save(epoch + 1)

                print("Test epoch: {}".format(epoch))
                for batch in range(n_batches):
                    self.test_one_iteration(n_steps)
                    print("\tTest batch: {}".format(batch))


    def train_one_iteration(self, n_steps):
        self.bid += 1
        session = tf.get_default_session()

        ret = {}

        opt_name, optimizee = random.choice(list(self.optimizees.items()))

        x = optimizee.get_initial_x()
        state = session.run(self.initial_state, feed_dict={self.x: x})

        optimizee_params = optimizee.get_new_params()

        losses = []
        fxs = []
                
        print("Optimizee: {}".format(opt_name))

        for i in range(n_steps // self.n_bptt_steps):
            feed_dict = optimizee_params
            feed_dict.update({inp: init for inp, init in zip(self.input_state, state)})
            feed_dict.update(optimizee.get_next_dict(self.n_bptt_steps))

            _, state, loss, fx, summaries_str = session.run([
                self.apply_gradients[opt_name], self.states[opt_name][-1], self.loss[opt_name], self.losses[opt_name], self.summaries[opt_name]
            ], feed_dict=feed_dict)

            if i == 0:
                print("\t\tFirst function value: {}".format(fx[0]))

            losses.append(loss)
            fxs.extend(fx)

        print("\t\tLoss: {}".format(np.mean(losses)))
        print("\t\tLast function value: {}".format(fx[-1]))

        ret['optimizee_name'] = opt_name
        ret['loss'] = np.mean(losses)
        ret['fxs'] = fxs

        for summary_str in summaries_str:
            self.train_writer.add_summary(summary_str, self.bid)
        self.train_writer.flush()

        return ret


    def build(self):
        with tf.variable_scope('{}_opt'.format(self.name)) as scope:
            self.train_writer = tf.summary.FileWriter('{}_data/train'.format(self.name), self.session.graph)
            self.test_writer = tf.summary.FileWriter('{}_data/test'.format(self.name), self.session.graph)
            self.summaries = {name: [] for name in self.optimizees}

            self._build_pre()
            self._build_input()
            self._build_initial_state()

            self._build_loop()
            self._build_loss()

            self.all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
            self._build_optimizer()

            # loglr_mean, loglr_std = tf.nn.moments(self.loglr, axes=[0])
            # lr_mean, lr_std = tf.nn.moments(tf.exp(self.loglr), axes=[0])

            for opt_name in self.optimizees:

                loglr_mean, loglr_std = tf.nn.moments(self.states[opt_name][-1][-1], axes=[0])
                lr_mean, lr_std = tf.nn.moments(tf.exp(self.states[opt_name][-1][-1]), axes=[0])
                 
                self.summaries[opt_name].append(tf.summary.scalar('train_loss', self.loss[opt_name]))
                self.summaries[opt_name].append(tf.summary.scalar('log_learning_rate_mean', loglr_mean))
                self.summaries[opt_name].append(tf.summary.scalar('log_learning_rate_std', loglr_std))
                self.summaries[opt_name].append(tf.summary.scalar('learning_rate_mean', lr_mean))
                self.summaries[opt_name].append(tf.summary.scalar('learning_rate_std', lr_std))
                self.summaries[opt_name].append(tf.summary.scalar('function_value', self.losses[opt_name][-1]))

            self.saver = tf.train.Saver(max_to_keep=None, var_list=self.all_vars, allow_empty=True)


    def _build_pre(self):
        self.loglr = tf.get_variable('lr', [], initializer=tf.constant_initializer(0))


    def _build_loop(self):
        self.losses = {}
        self.states = {}
        self.norms  = {}

        with tf.variable_scope('loop_scope') as self.loop_scope:
            
            reused = False

            for opt_name, optimizee in self.optimizees.items():
                states = []
                losses = []
                norms = []

                state = self.input_state

                for i in range(self.n_bptt_steps):
                    state, f, g_norm = self._iter(optimizee.loss, i, state)

                    losses.append(f)
                    states.append(state)
                    norms.append(g_norm)

                    if not reused:
                       self.loop_scope.reuse_variables()
                       reused = True

                self.losses[opt_name] = losses
                self.states[opt_name] = states
                self.norms [opt_name] = norms


    def _build_loss(self):
        self.loss = {}

        for opt_name in self.optimizees:
            losses = self.losses[opt_name]

            if self.loss_type == 'log':
                loss = tf.reduce_mean(tf.log(losses) - tf.log(losses[0]))
            elif self.loss_type == 'sum':
                loss = tf.reduce_mean(losses)
            else:
                loss = losses[-1]

            self.loss[opt_name] = loss


    def _fg(self, f, x, i):
        fx = f(x, i)
        g = gradients.gradients(fx, x)[0]
        return fx, g


    def _iter(self, f, i, state):
        x, = state

        fx, g = self._fg(f, x, i)
        g_norm = tf.reduce_sum(g**2)

        x -= tf.exp(self.loglr) * g
        return [x], fx, g_norm


    def _build_input(self):
        self.x = tf.placeholder(tf.float32, shape=[None])
        self.input_state = [self.x]


    def _build_initial_state(self):
        self.initial_state = [self.x]


    def _build_optimizer(self):
        try:
            self.apply_gradients = {}

            self.optimizer = tf.train.AdamOptimizer(self.train_lr)

            for opt_name in self.optimizees:
                gradients = self.optimizer.compute_gradients(self.loss[opt_name], var_list=self.all_vars)
                self.apply_gradients[opt_name] = self.optimizer.apply_gradients(gradients)
        except:
            self.apply_gradients = None


    def restore(self, eid):
        self.saver.restore(self.session, '{}_data/epoch-{}'.format(self.name, eid))
        print(self.name, "restored.")


    def save(self, eid):
        folder = '{}_data'.format(self.name)
        filename = '{}/epoch'.format(folder)
        sfilename = '{}/epoch-last'.format(folder)

        self.saver.save(self.session, filename, global_step=eid)
        os.unlink("{}-{}.meta".format(filename, eid))
        if os.path.lexists(sfilename):
            os.unlink(sfilename)
        os.symlink("epoch-{}".format(eid), sfilename)
        print(self.name, "saved.")
