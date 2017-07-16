"""
    This module defines various testing functions.
"""

from collections import OrderedDict
import numpy as np
import tensorflow as tf

from opts.sgd_opt import SgdOpt
from opts.momentum_opt import MomentumOpt
from opts.adam_opt import AdamOpt

import util
import optimizees as optim


def get_tests():
    """
        This functions returns set of non-trainable optimizees
        to compare with on different experiments.
    """
    def opt(name, lr):
        return {
            'sgd': SgdOpt,
            'momentum': MomentumOpt,
            'adam': MomentumOpt
        }[name](lr=lr, name='{}_lr_{}'.format(name, lr))


    problems = {
        'rosenbrock',
        'quadratic',
        'beale',
        'booth',
        'matyas',
        'logreg',
        'stoch_logreg',
        'stoch_linear',
        'digits_classifier'
    }

    opts = {
        'sgd',
        'momentum',
        'adam'
    }


    tests = {}

    for p in problems:
        tests[p] = {
            o: [opt(o, lr) for lr in np.logspace(start=-1, stop=-5, num=5)]
            for o in opts
        }


    return tests


def run_cv_testing(opt, flags):
    """
        Runs testing of different snapshots of LSTM optimizer.
    """
    results = OrderedDict()
    random_state = np.random.get_state()

    for eid in range(flags.start_eid, flags.eid + 1, flags.step):
        np.random.set_state(random_state)
        rets = opt.test(eid=eid,
                        n_batches=flags.n_batches,
                        n_steps=flags.n_steps,
                        verbose=flags.verbose)

        name = '{name}_{eid}'.format(name=flags.name, eid=eid)
        results[name] = rets

    return results


def run_many_testing(opt, s_opts, flags):
    """
        Runs testing of LSTM with non-trainable optimizers.
    """
    results = OrderedDict()
    random_state = np.random.get_state()

    for optimizer in [opt] + s_opts:
        np.random.set_state(random_state)
        rets = optimizer.test(eid=flags.eid,
                              n_batches=flags.n_batches,
                              n_steps=flags.n_steps,
                              verbose=flags.verbose)
        results[optimizer.name] = rets

    return results


def run_test(flags):
    """
        This functions runs testing according to flags.
    """
    if flags.eid == 0:
        raise ValueError("eid must be > 0 if mode is testing")

    if flags.gpu is not None and flags.gpu:
        flags.gpu = flags.gpu[0]

    optimizees = optim.get_optimizees(clip_by_value=False,
                                      random_scale=flags.enable_random_scaling,
                                      noisy_grad=flags.noisy_grad)
    optimizee = {flags.problem: optimizees[flags.problem]}

    save_path = 'snapshots'
    if flags.tag:
        save_path += '_' + flags.tag

    opt = util.load_opt(flags.name, save_path=save_path, snapshot_path=flags.snapshot_path)
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
            util.dump_results(model_path, 
                              results,
                              phase='test',
                              problem=flags.problem,
                              mode=flags.mode,
                              tag=flags.tag,
                              compare_with=flags.compare_with)
