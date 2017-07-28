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

        
        def extract(opt_name, key, pkey='inference'):
            inf = model.ops[opt_name][pkey]

            if hasattr(model, 'devices'):
                inf = list(inf.values())[0]

            return inf[key]


        self.run_op = {opt_name: [
            #model.ops[opt_name]['inference']['states'],
            #model.ops[opt_name]['losses'][0],
            #model.ops[opt_name]['inference']['values'],
            #model.ops[opt_name]['inference']['norms'],
            extract(opt_name, 'states'),
            extract(opt_name, 0, pkey='losses'),
            extract(opt_name, 'values'),
            extract(opt_name, 'norms'),
            model.ops[opt_name]['summaries']
        ] for opt_name in model.optimizees}
            
        if mode == 'train':
            for opt_name in model.optimizees:
                self.run_op[opt_name].append(model.ops[opt_name]['train_op'])


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
              eid=0, test=True, verbose=1):

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

        losses, fxs, norms, lrs = [], [], [], []

        self.log("Optimizee: {}".format(opt_name), level=15)

        for i in range(n_steps // self.model.n_bptt_steps):
            feed_dict = optimizee_params
            feed_dict.update({inp: init for inp, init in zip(self.model.input_state, state)})
            feed_dict.update(optimizee.get_next_dict(self.model.n_bptt_steps, batch_size))
            feed_dict.update({
                self.model.train_lr: self.lr,
                self.model.momentum: self.mu,
            })

            states, loss, fx, g_norm, summaries_str = self.session.run(self.run_op[opt_name], feed_dict=feed_dict)[:5]
            state = states[-1]

            losses.append(loss)
            fxs.extend(fx)
            lrs.extend([s[-1] for s in states])
            norms.extend(g_norm)

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
            'lrs': np.array(lrs).mean(axis=1),
            'norms': np.array(norms)
        }


    def log(self, message, level=logging.INFO):
        extra = {
            'batch': self.batch,
            'epoch': self.epoch,
        }
        self.logger.log(level, message, extra=extra)
