import logging
from collections import namedtuple

import numpy as np
import tensorflow as tf

from util import log_execution_time
from problem_producer import RandomProducer, FixedProducer


def clip_grads(grads, clip=3.0):
    grads, tvars = list(zip(*grads))

    grads, _ = tf.clip_by_global_norm(grads, clip)
    grads = list(zip(grads, tvars))

    return grads


RNNOptState = namedtuple('RNNOptState', ['x', 'rnn_state'])
Point = namedtuple('Point', ['value', 'gradient', 'gradient_norm'])


class BasicModel:
    def __init__(self, init_config, name=None, snapshot_path=None):
        self.name = name
        self.init_config = init_config
        self.snapshot_path = snapshot_path

        #FORMAT = '%(asctime)-15s epoch=%(epoch)d;batch=%(batch)d; %(message)s'
        FORMAT = '%(asctime)-15s %(message)s'
        logging.basicConfig(format=FORMAT)
        logging.getLogger().setLevel(logging.INFO)

    
    def build_pre(self):
        raise NotImplementedError


    def step(self, f, state):
        raise NotImplementedError


    @staticmethod
    def get_inputs(f, theta, i):
        value, gradient = f(theta, i)
        gradient = tf.stop_gradient(gradient)
        gradient_norm = tf.reduce_sum(tf.square(gradient), axis=-1)

        return Point(value, gradient, gradient_norm)


    def step_with_func(self, f, state, step):
        output = self.get_inputs(f, state.x, step)
        step, rnn_state = self.step(output.gradient, state.rnn_state)
        state = state._replace(x=state.x + step, rnn_state=rnn_state)
        return output, state


    def build(self, optimizees, config):
        self.theta = tf.placeholder(tf.float32, shape=[None, None], name='theta')

        self.build_pre()
        self.initial_state = RNNOptState(self.theta, self.init_state(self.theta))

        self.n_bptt_steps = config.n_bptt_steps
        self.optimizees = optimizees

        ops = {}
        
        if not config.inference_only:
            self.train_lr = tf.placeholder(tf.float32, shape=[], name='train_lr')
            self.momentum = tf.placeholder(tf.float32, shape=[], name='momentum')

            optimizer = tf.train.AdamOptimizer(self.train_lr, beta1=self.momentum)

        reuse = False
        reg_loss = None

        for opt_name, optimizee in optimizees.items():
            with tf.variable_scope('inference', reuse=reuse) as self.scope:
                with tf.name_scope(opt_name):
                    state = self.initial_state
                    values = []
                    norms = []
                    lrs = []

                    for step in range(config.n_bptt_steps):
                        output, state = self.step_with_func(optimizee.loss, state, step)

                        values.append(output.value)
                        norms.append(output.gradient_norm)

                        if 'loglr' in state.rnn_state._fields:
                            lrs.append(state.rnn_state.loglr)

                        if step == 0:
                            self.scope.reuse_variables()

                    values = tf.stack(values, axis=0)
                    norms = tf.stack(norms, axis=0)

                    loss = tf.reduce_mean(tf.log(values + 1e-8) - tf.log(values[:1] + 1e-8))
                    ops[opt_name] = dict(values=values, norms=norms, losses=[loss], final_state=state)

            reuse = True

            if not config.inference_only:
                with tf.variable_scope('train'):
                    if reg_loss is None:
                        reg_loss = tf.add_n([tf.norm(v, 1) for v in tf.trainable_variables()]) * config.lambd_l1

                    grads = optimizer.compute_gradients(loss + reg_loss, var_list=tf.trainable_variables())

                    if config.normalize_lstm_grads:
                        norm = tf.global_norm(grads)
                        grads = [(grad / (norm + 1e-8), var) for grad, var in grads]

                    grads = clip_grads(grads, config.grad_clip)
                    train_op = optimizer.apply_gradients(grads)

                    ops[opt_name].update(losses=[loss, reg_loss], train_op=train_op)
                    if 'loglr' in state._fields:
                        ops[opt_name]['lrs'] = lrs

            self.ops = ops
            self.saver = tf.train.Saver(max_to_keep=None, var_list=tf.trainable_variables(), allow_empty=True)


    def test(self, config, producer=None):
        self.session = tf.get_default_session()
        if config.restore:
            self.restore(config.eid)

        producer = producer or RandomProducer(self.optimizees)
        rets = []

        for batch in range(config.n_batches):
            with log_execution_time('batch {}'.format(batch), logging.info):
                problem = producer.sample(config.batch_size)
                logging.info("Optimizee: {}".format(problem.name))

                ops = self.ops[problem.name].copy()

                if 'train_op' in ops:
                    ops.pop('train_op')

                ret = self.run_iteration(problem, config, ops)
                
            #logging.info("Loss: {}".format(ret['loss'] / np.log(10)))
            rets.append(ret)

        return rets


    def run_iteration(self, problem, config, run_op):
        n_unrolls = config.n_steps // self.n_bptt_steps
        train = 'train_op' in run_op

        if train:
            if config.masked_train == 'random':
                mask = np.random.binomial(n=1, p=config.masked_train_p, size=n_unrolls)
            elif config.masked_train == 'first-last':
                mask = np.ones(n_unrolls)
                left = int(config.masked_train_p / 2 * n_unrolls)
                right = n_unrolls - int(config.masked_train_p / 2 * n_unrolls)
                mask[left:right] = 0 
            else:
                mask = np.ones(n_unrolls)

        state = self.session.run(self.initial_state, feed_dict={self.theta: problem.init})
        losses = []
        results = {field: [] for field in run_op if field not in {'losses', 'final_state', 'train_op'}}

        for i in range(n_unrolls):
            feed_dict = self.get_feed_dict(state, problem, self.n_bptt_steps, config.batch_size)

            if train:
                feed_dict.update({
                    self.train_lr: config.train_lr,
                    self.momentum: config.momentum,
                })

            if train and mask[i] == 0:
                run_op = run_op.copy()
                del run_op['train_op']

            info = self.session.run(run_op, feed_dict=feed_dict)
            state = info['final_state']

            losses.append(info['losses'][0])

            for field in results:
                results[field].extend(info[field])
                
        d = dict(optimizee_name=problem.name, loss=np.mean(losses))
        d.update({
            field: np.array(vals)
            for field, vals in results.items()
        })
        return d
    
    
    def run_train_epoch(self, config, producer):
        train_rets = []

        for batch in range(config.n_batches):
            with log_execution_time('train batch', logging.info):
                problem = producer.sample(config.batch_size)
                ret = self.run_iteration(problem, config, self.ops[problem.name])

            logging.info("First function value: {}".format(ret['values'][0][0]))
            logging.info("Last function value: {}".format(ret['values'][-1][0]))
            logging.info("Loss: {}".format(ret['loss'] / np.log(10)))

            if np.isnan(ret['loss']):
                logging.error("Loss is NaN")
                raise "Loss is NaN"
            elif np.isinf(ret['loss']):
                logging.error("Loss is +-INF")
                raise "Loss is +-INF"

            train_rets.append(ret)

        return train_rets


    def train(self, config, train_producer=None, test_producer=None):
        self.session = tf.get_default_session()
        train_producer = train_producer or RandomProducer(self.optimizees)
        test_producer = test_producer or FixedProducer(self.optimizees).new(config.n_batches, config.batch_size)

        logging.info("Training model: {}".format(self.name))

        if config.eid > 0:
            self.restore(config.eid)
            self.bid = config.eid * config.n_batches

        train_rets = []
        test_rets = []

        test_config = config.to_test_config()

        try:
            for epoch in range(config.eid, config.n_epochs):
                self.epoch = epoch
                with log_execution_time('train epoch', logging.info):
                    train_rets.extend(self.run_train_epoch(config, producer=train_producer))

                loss = np.mean([r['loss'] for r in train_rets])
                logging.info("Epoch loss: {}".format(loss / np.log(10)))

                if (epoch + 1) % config.save_every == 0:
                    self.save(epoch + 1)

                if config.test and (epoch + 1) % config.test_every == 0:
                    with log_execution_time('test epoch', logging.info):
                        test_rets.extend(self.test(test_config, producer=test_producer))

        except KeyboardInterrupt:
            print("Stopped training early")

        return train_rets, test_rets


    def get_feed_dict(self, state, problem, n_bptt_steps, batch_size=1):
        feed_dict = dict(zip(self.initial_state, state))
        feed_dict.update(problem.params)
        feed_dict.update(problem.get_next_dict(n_bptt_steps, batch_size))
        return feed_dict


    def restore(self, eid):
        snapshot_path = self.snapshot_path / 'epoch-{}'.format(eid)
        print("Snapshot path: ", snapshot_path)

        self.saver.restore(self.session, str(snapshot_path))
        print(self.name, "restored.")


    def save(self, eid):
        filename = self.snapshot_path / 'epoch'
        print("Saving to ", filename)

        self.saver.save(self.session, str(filename), global_step=eid)
        print(self.name, "saved.")
