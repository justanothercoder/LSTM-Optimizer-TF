import shutil
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from sgd_opt import SgdOpt
from momentum_opt import MomentumOpt

import util
from util import lstm_opt, get_optimizees


def get_tests(optimizee):
    tests = {
        'rosenbrock': {
            'sgd': [SgdOpt(optimizee, lr=2**(-i-5), name='sgd_lr_{}'.format(-i-9)) for i in range(1, 6)],
            'momentum': [MomentumOpt(optimizee, lr=2**(-i-9), name='momentum_lr_{}'.format(-i-9)) for i in range(1, 3)],
        },
        'quadratic': {
            'sgd': [SgdOpt(optimizee, lr=16 * 2**(-i), name='sgd_lr_{}'.format(4-i)) for i in range(0, 6)],
            'momentum': [MomentumOpt(optimizee, lr=16 * 2**(-i), name='momentum_lr_{}'.format(4-i)) for i in range(0, 6)],
        },
        'logreg': {
            'sgd': [SgdOpt(optimizee, lr=2**(-i-5), name='sgd_lr_{}'.format(-i-9)) for i in range(1, 6)],
            'momentum': [MomentumOpt(optimizee, lr=2**(-i-1), name='momentum_lr_{}'.format(-i-1)) for i in range(1, 3)],
        }
    }

    return tests


def run_test(flags):
    graph = tf.Graph()
            
    optimizees = get_optimizees(clip_by_value=True, random_scale=flags.enable_random_scaling)
    optimizee = {flags.problem: optimizees[flags.problem]}

    with graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4

        with tf.Session(config=config, graph=graph) as session:
            opt = lstm_opt(optimizee, flags)
            s_opts = get_tests(optimizee)[flags.problem][flags.compare_with]
            
            optimizees[flags.problem].build()
            opt.build()

            for s_opt in s_opts:
                s_opt.build()

            session.run(tf.global_variables_initializer())

            if flags.eid == 0:
                raise ValueError("eid must be > 0 if mode is testing")

            st = np.random.get_state()

            results = OrderedDict()

            if flags.mode == 'many':
                for o in [opt] + s_opts:
                    np.random.set_state(st)
                    rets = o.test(eid=flags.eid, n_batches=flags.n_batches, n_steps=flags.n_steps)
                    results[o.name] = rets
            else:
                for eid in range(flags.start_eid, flags.eid + 1, flags.step):
                    np.random.set_state(st)
                    rets = opt.test(eid=eid, n_batches=flags.n_batches, n_steps=flags.n_steps)

                    name = '{name}_{eid}'.format(name=flags.name, eid=eid)
                    results[name] = rets

            util.dump_results(flags.model_path, results, phase='test', problem=flags.problem, mode=flags.mode)

            for o in s_opts:
                try:
                    shutil.rmtree('models/' + o.name)
                except:
                    pass
