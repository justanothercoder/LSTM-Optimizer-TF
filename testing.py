from collections import OrderedDict

import numpy as np
import tensorflow as tf

from opts.sgd_opt import SgdOpt
from opts.momentum_opt import MomentumOpt

import util
from util import lstm_opt, get_optimizees


def get_tests():
    tests = {
        'rosenbrock': {
            'sgd': [SgdOpt(lr=2**(-i-5), name='sgd_lr_{}'.format(-i-5)) for i in range(1, 6)],
            'momentum': [MomentumOpt(lr=2**(-i-9), name='momentum_lr_{}'.format(-i-9)) for i in range(1, 3)],
        },
        'quadratic': {
            'sgd': [SgdOpt(lr=16 * 2**(-i), name='sgd_lr_{}'.format(4-i)) for i in range(0, 6)],
            'momentum': [MomentumOpt(lr=16 * 2**(-i), name='momentum_lr_{}'.format(4-i)) for i in range(0, 6)],
        },
        'logreg': {
            'sgd': [SgdOpt(lr=2**(-i-5), name='sgd_lr_{}'.format(-i-5)) for i in range(1, 6)],
            'momentum': [MomentumOpt(lr=2**(-i-1), name='momentum_lr_{}'.format(-i-1)) for i in range(1, 3)] +
                        [MomentumOpt(lr=2**(-i+2), name='momentum_lr_{}'.format(-i+2)) for i in range(1, 3)],
        },
        'stoch_logreg': {
            'sgd': [SgdOpt(lr=2**(-i-5), name='sgd_lr_{}'.format(-i-5)) for i in range(1, 6)],
            'momentum': [MomentumOpt(lr=2**(-i-1), name='momentum_lr_{}'.format(-i-1)) for i in range(1, 3)] +
                        [MomentumOpt(lr=2**(-i+2), name='momentum_lr_{}'.format(-i+2)) for i in range(1, 3)],
        },
        'stoch_linear': {
            'sgd': [SgdOpt(lr=2**(-i-5), name='sgd_lr_{}'.format(-i-5)) for i in range(1, 6)],
            'momentum': [MomentumOpt(lr=2**(-i), name='momentum_lr_{}'.format(2**(-i))) for i in range(8, 10)],
        }
    }

    return tests


def run_cv_testing(opt, flags):
    results = OrderedDict()
    st = np.random.get_state()

    for eid in range(flags.start_eid, flags.eid + 1, flags.step):
        np.random.set_state(st)
        rets = opt.test(eid=eid, n_batches=flags.n_batches, n_steps=flags.n_steps, verbose=flags.verbose)

        name = '{name}_{eid}'.format(name=flags.name, eid=eid)
        results[name] = rets

    return results


def run_many_testing(opt, s_opts, flags):
    results = OrderedDict()
    st = np.random.get_state()

    for o in [opt] + s_opts:
        np.random.set_state(st)
        rets = o.test(eid=flags.eid, n_batches=flags.n_batches, n_steps=flags.n_steps, verbose=flags.verbose)
        results[o.name] = rets

    return results


def run_test(flags):
    if flags.eid == 0:
        raise ValueError("eid must be > 0 if mode is testing")

    if flags.gpu is not None and flags.gpu:
        flags.gpu = flags.gpu[0]

    optimizees = get_optimizees(clip_by_value=True, random_scale=flags.enable_random_scaling, noisy_grad=flags.noisy_grad)
    optimizee = {flags.problem: optimizees[flags.problem]}

    opt = util.load_opt(flags.name)
    s_opts = get_tests()[flags.problem][flags.compare_with]

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(config=util.get_tf_config(), graph=graph) as session:

            optimizees[flags.problem].build()
            opt.build(optimizee, inference_only=True)
            for s_opt in s_opts:
                s_opt.build(optimizee, inference_only=True)

            #session.run(tf.global_variables_initializer())
            session.run(tf.global_variables_initializer())

            if flags.mode == 'many':
                results = run_many_testing(opt, s_opts, flags)
            else:
                results = run_cv_testing(opt, flags)

            model_path = util.get_model_path(flags.name)
            util.dump_results(model_path, results, phase='test', problem=flags.problem, mode=flags.mode, tag=flags.tag)
