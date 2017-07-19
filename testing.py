"""This module defines various testing functions."""

from collections import OrderedDict
import numpy as np
import tensorflow as tf

from opts.sgd_opt import SgdOpt
from opts.momentum_opt import MomentumOpt
from opts.adam_opt import AdamOpt

import util
import util.tf_utils as tf_utils
import util.paths as paths
import optimizees as optim


def get_tests(test_problem, compare_with):
    """
        This function returns set of non-trainable optimizees
        to compare with on different experiments.
    """
    def make_opt(name, learning_rate):
        #pylint: disable=missing-docstring
        return {
            'sgd': SgdOpt,
            'momentum': MomentumOpt,
            'adam': AdamOpt
        }[name](lr=learning_rate, name='{}_lr_{}'.format(name, learning_rate))

    problems = {
        'rosenbrock', 'quadratic', 'beale', 'booth', 'matyas', 'logreg',
        'stoch_logreg', 'stoch_linear',
        'digits_classifier', 'mnist_classifier', 'digits_classifier_2'
    }

    opts = {'sgd', 'momentum', 'adam'}

    lrs = np.logspace(start=-1, stop=-5, num=5)
    tests = {}
    for problem in problems:
        tests[problem] = {}
        for opt in opts:
            tests[problem][opt] = [make_opt(opt, lr) for lr in lrs]

    return tests[test_problem][compare_with]


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


def save_test_config(flags, experiment_path):
    """This function dumps testing config to directory where model lies."""
    testing_options = {'eid', 'n_batches', 'n_steps', 'verbose'}
    util.dump_config(experiment_path / 'config', flags, testing_options)


def setup_experiment(flags):
    """Setups directories and loads optimizer"""
    if flags.eid == 0:
        raise ValueError("eid must be > 0 if mode is testing")

    model_path = paths.model_path(flags.name)
    train_experiment_path = paths.experiment_path(flags.name, flags.train_experiment_name, 'train')
    experiment_path = paths.experiment_path(flags.name, flags.experiment_name, 'test')

    print("Model path: ", model_path)
    print("Train experiment path: ", train_experiment_path)
    print("Test experiment path: ", experiment_path)

    paths.make_dirs(experiment_path)
    save_test_config(flags, experiment_path)

    opt = util.load_opt(model_path, train_experiment_path)

    optimizees = optim.get_optimizees([flags.problem],
                                      clip_by_value=False,
                                      random_scale=flags.enable_random_scaling,
                                      noisy_grad=flags.noisy_grad)

    s_opts = get_tests(flags.problem, flags.compare_with)
    return experiment_path, opt, s_opts, optimizees


@tf_utils.with_tf_graph
def testing(flags, opt, s_opts, optimizees):
    """Runs testing"""
    for optimizee in optimizees.values():
        optimizee.build()

    opt.build(optimizees, inference_only=True, devices=tf_utils.get_devices(flags))

    for s_opt in s_opts:
        s_opt.build(optimizees, inference_only=True, devices=tf_utils.get_devices(flags))

    session = tf.get_default_session()
    session.run(tf.global_variables_initializer())

    if flags.mode == 'many':
        results = run_many_testing(opt, s_opts, flags)
    else:
        results = run_cv_testing(opt, flags)

    return results


def run_test(flags):
    """This function runs testing according to flags."""
    experiment_path, opt, s_opts, optimizees = setup_experiment(flags)

    if not flags.force and (experiment_path / 'results.pkl').exists():
        print("You will overwrite existing results. Add -f/--force to force it.")
        return

    results = testing(flags, opt, s_opts, optimizees)

    data = {
        'results': results,
        'problem': flags.problem,
        'mode': flags.mode,
    }

    util.dump_results(experiment_path, data)
