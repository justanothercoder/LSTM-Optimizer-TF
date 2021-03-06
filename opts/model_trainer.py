import time
import random
import logging
import functools

import numpy as np
import tensorflow as tf
from opts import distributed


def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        logging.info("Time: {}".format(time.time() - start_time))
        return ret
    return wrapper


class Trainer:
    def __init__(self):
        self.model = None
        self.mode = None
        self.bid = None
        self.session = None
        self.run_op = None

        FORMAT = '%(asctime)-15s epoch=%(epoch)d;batch=%(batch)d; %(message)s'
        logging.basicConfig(format=FORMAT)

        self.logger = logging.getLogger('trainer')
        self.logger.setLevel(logging.INFO)


    def setup_and_run(self, model, mode, session=None, **kwargs):
        self.model = model
        self.mode = mode
        self.bid = 0
        self.session = session or tf.get_default_session()

        if kwargs['verbose'] == 2:
            self.logger.setLevel(15)
        elif kwargs['verbose'] == 0:
            self.logger.setLevel(30)


        self.run_op = {}

        for opt_name in model.optimizees:
            inf = model.ops[opt_name]['inference']
            losses = model.ops[opt_name]['losses']

            if hasattr(model, 'devices'):
                inf = list(inf.values())[0]
                losses = list(losses.values())[0]
        
            self.run_op[opt_name] = {
                'loss': losses[0],
                'summaries': model.ops[opt_name]['summaries'],
                'values': inf['values'],
                'norms': inf['norms'],
                'final_state': inf['final_state']
            }

            if inf.get('states') is not None:
                self.run_op[opt_name]['states'] = inf['states']

            if mode == 'train':
                self.run_op[opt_name]['train_op'] = model.ops[opt_name]['train_op']


        return getattr(self, mode)(**kwargs)


    def test(self, eid, n_batches, n_steps=20, opt_name=None):
        self.model.restore(eid)

        sample_optimizee = (opt_name is None)
        opt_names = list(self.model.optimizees.keys())

        rets = []
        for batch in range(n_batches):
            #self.logger.info("Batch: {}".format(batch))
            if sample_optimizee:
                opt_name = random.choice(opt_names)

            ret = self.run_iter(opt_name, n_steps, batch_size=1, train=False)
            rets.append(ret)

        return rets


    def train(self, n_epochs, n_batches,
              batch_size=100, n_steps=20,
              train_lr=1e-4, momentum=0.9,
              eid=0, test=True, verbose=1, masked_train='none', masked_train_p=0.2):

        self.masked_train = masked_train
        self.masked_train_p = masked_train_p

        self.logger.info("Training model: {}".format(self.model.name), extra={'epoch': 0, 'batch': 0})

        self.lr = train_lr
        self.mu = momentum

        if eid > 0:
            self.model.restore(eid)
            self.bid = eid * n_batches

        train_rets = []
        test_rets = []

        sample_steps = bool(n_steps == 0)

        try:
            for epoch in range(eid, n_epochs):
                self.epoch = epoch
                #self.logger.info("Epoch: {}".format(epoch))
                epoch_time = time.time()

                loss = None

                for batch in range(n_batches):
                    self.batch = batch
                    #self.logger.info("Batch: {}".format(batch))
                    if sample_steps:
                        #n_steps = int(np.random.exponential(scale=200)) + 50

                        exp_scale = min(50, epoch)
                        n_steps = int(np.random.exponential(scale=exp_scale)) + 1
                        n_steps *= self.n_bptt_steps
                        self.log("n_steps: {}".format(n_steps), level=15)

                    opt_name = random.choice(list(self.model.optimizees.keys()))
                    ret = self.run_iter(opt_name, n_steps, batch_size, train=True)

                    if np.isnan(ret['loss']):
                        self.log("Loss is NaN", level=40)
                        raise "Loss is NaN"
                    elif np.isinf(ret['loss']):
                        self.log("Loss is +-INF", level=40)
                        raise "Loss is +-INF"
                        #print(ret['fxs'])

                    if loss is not None:
                        loss = 0.9 * loss + 0.1 * ret['loss']
                    else:
                        loss = ret['loss']

                    train_rets.append(ret)

                self.log("Epoch time: {}".format(time.time() - epoch_time))
                self.log("Epoch loss: {}".format(loss / np.log(10) / self.model.n_bptt_steps))

                if test and (epoch + 1) % 10 == 0:
                    self.model.save(epoch + 1)

                    self.log("Test epoch: {}".format(epoch))
                    test_epoch_time = time.time()

                    for batch in range(n_batches):
                        self.log("Test batch: {}".format(batch), level=15)
                    
                        opt_name = random.choice(list(self.model.optimizees.keys()))
                        ret = self.run_iter(opt_name, n_steps, batch_size=1, train=False)

                        test_rets.append(ret)

                    test_epoch_time = time.time() - test_epoch_time
                    self.log("Epoch time: {}".format(test_epoch_time))
        except KeyboardInterrupt:
            print("Stopped training early")

        return train_rets, test_rets


    @log_execution_time
    def run_iter(self, opt_name, n_steps, batch_size, train=True):
        self.bid += 1

        optimizee = self.model.optimizees[opt_name]
        x = optimizee.get_initial_x(batch_size)
        state = self.session.run(self.model.initial_state, feed_dict={self.model.x: x})

        optimizee_params = optimizee.get_new_params(batch_size)
        
        steps_info = {
            'values': [],
            'norms': [],
        }

        losses = []
        fxs = []

        self.log("Optimizee: {}".format(opt_name), level=15)

        n_unrolls = n_steps // self.model.n_bptt_steps
        if self.masked_train == 'random':
            mask = np.random.binomial(n=1, p=self.masked_train_p, size=n_unrolls)
        elif self.masked_train == 'first-last':
            mask = np.ones(n_unrolls)
            left = int(self.masked_train_p / 2 * n_unrolls)
            right = n_unrolls - int(self.masked_train_p / 2 * n_unrolls)
            mask[left:right] = 0 
        else:
            mask = np.ones(n_unrolls)


        for i in range(n_steps // self.model.n_bptt_steps):
            feed_dict = optimizee_params
            feed_dict.update({inp: state[name] for name, inp in self.model.input_state.items()})
            feed_dict.update(optimizee.get_next_dict(self.model.n_bptt_steps, batch_size))
            feed_dict.update({
                self.model.train_lr: self.lr,
                self.model.momentum: self.mu,
            })

            run_op = self.run_op[opt_name].copy()
            if mask[i] == 0:
                del run_op['train_op']

            info = self.session.run(run_op, feed_dict=feed_dict)
            #state = info['states'][-1]
            state = info['final_state']
            summaries_str = info['summaries']

            losses.append(info['loss'])
            fxs.extend(info['values'])

        self.log("First function value: {}".format(fxs[0][0]), level=15)
        self.log("Last function value: {}".format(fxs[-1][0]), level=15)
        self.log("Loss: {}".format(np.mean(losses / np.log(10))), level=15)
        
        if self.model.save_tf_data:
            if train:
                for summary_str in summaries_str:
                    self.model.train_writer.add_summary(summary_str, self.bid)
                self.model.train_writer.flush()
            else:
                for summary_str in summaries_str:
                    self.model.test_writer.add_summary(summary_str, self.bid)
                self.model.test_writer.flush()

        return {
            'optimizee_name': opt_name,
            'loss': np.nanmean(losses),
            'fxs': np.array(fxs),
        }


    def log(self, message, level=logging.INFO):
        extra = {
            'batch': self.batch,
            'epoch': self.epoch,
        }
        self.logger.log(level, message, extra=extra)
