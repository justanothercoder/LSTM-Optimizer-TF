"""This module defines various testing functions."""

from collections import OrderedDict
import itertools
import numpy as np
import tensorflow as tf

from opts.sgd_opt import SgdOpt
from opts.momentum_opt import MomentumOpt
from opts.adam_opt import AdamOpt

import util
import paths
import optimizees as optim


def get_tests(problem, compare_with):
    """
        This function returns set of non-trainable optimizees
        to compare with on different experiments.
    """
    def opt(name, lr):
        return {
            'sgd': SgdOpt,
            'momentum': MomentumOpt,
            'adam': AdamOpt
        }[name](lr=lr, name='{}_lr_{}'.format(name, lr))

    problems = {
        'rosenbrock', 'quadratic',
        'beale', 'booth', 'matyas',
        'logreg',
        'stoch_logreg', 'stoch_linear',
        'digits_classifier', 'mnist_classifier',
        'digits_classifier_2'
    }

    opts = {'sgd', 'momentum', 'adam'}

    lrs = np.logspace(start=-1, stop=-5, num=5)
    tests = {}
    for p in problems:
        tests[p] = {}
        for o in opts:
            tests[p][o] = [opt(o, lr) for lr in lrs]

    return tests[problem][compare_with]


def run_cv_testing(opt, flags):
    """Runs testing of different snapshots of LSTM optimizer."""
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
    """Runs testing of LSTM with non-trainable optimizers."""
    results = OrderedDict()
    random_state = np.random.get_state()

    for optimizer in [opt] + s_opts:
        np.random.set_state(random_state)
        results[optimizer.name] = optimizer.test(eid=flags.eid,
                                                 n_batches=flags.n_batches,
                                                 n_steps=flags.n_steps,
                                                 verbose=flags.verbose)

    return results


def run_test(flags):
    """This function runs testing according to flags."""
    if flags.eid == 0:
        raise ValueError("eid must be > 0 if mode is testing")

    optimizees = optim.get_optimizees([flags.problem],
                                      clip_by_value=False,
                                      random_scale=flags.enable_random_scaling,
                                      noisy_grad=flags.noisy_grad)

    model_path = paths.model_path(flags.name)
    train_experiment_path = paths.experiment_path(flags.name, flags.train_experiment_name, 'train')
    experiment_path = paths.experiment_path(flags.name, flags.experiment_name, 'test')

    paths.make_dirs(experiment_path)

    s_opts = get_tests(flags.problem, flags.compare_with)
    
    graph = tf.Graph()
    session = tf.Session(config=util.get_tf_config(), graph=graph)
    with graph.as_default(), session:
        optimizees[flags.problem].build()
        
        opt = util.load_opt(model_path, train_experiment_path)
        opt.build(optimizees, inference_only=True, devices=util.get_devices(flags))
        
        for s_opt in s_opts:
            s_opt.build(optimizees, inference_only=True)

        session.run(tf.global_variables_initializer())

        if flags.mode == 'many':
            results = run_many_testing(opt, s_opts, flags)
        else:
            results = run_cv_testing(opt, flags)

        data = {
            'results': results,
            'problem': flags.problem,
            'mode': flags.mode,
        }
        
        util.dump_results(experiment_path, data)
