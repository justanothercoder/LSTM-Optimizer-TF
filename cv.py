"""
    This module defines several functions which perform parameter tuning
    using validation set.
"""

import time
import json
import itertools
import random
import pathlib
import subprocess
import shlex
from collections import OrderedDict

import numpy as np
import tensorflow as tf

import util
import optimizees as optim


def random_product(*args, repeat=1):
    "Random selection from itertools.product(*args, **kwds)"
    pools = [tuple(pool) for pool in args]
    result = []
    for _ in range(repeat):
        result.append(tuple(random.choice(pool) for pool in pools))
    return result


def get_score(rets):
    """
        This function computes score given the results of testing.
    """
    by_opt = lambda ret: ret['optimizee_name']
    splits, opt_names = util.split_list(rets, by_opt)

    scores = {}

    for opt_name in opt_names:
        losses = [ret['loss'] for ret in splits[opt_name]]
        scores[opt_name] = -np.mean(losses)

    return scores


def train_opt(opt, flags):
    """This function runs training of optimizer given flags."""
    if not isinstance(flags, dict):
        flags = vars(flags)

    print("Running training with parameters: {}".format(flags))

    train_options = {
        'n_epochs', 'n_batches', 'batch_size',
        'n_steps', 'train_lr', 'momentum'
    }

    train_options = {k: v for k, v in flags.items() if k in train_options}
    print('train_options: {}'.format(train_options))

    train_start_time = time.time()
    session = tf.get_default_session()
    session.run(tf.global_variables_initializer(), feed_dict={
        opt.train_lr: train_options['train_lr'],
        opt.momentum: train_options['momentum']
    })
    train_rets, _ = opt.train(test=False, **train_options)
    train_time = time.time() - train_start_time

    return train_rets, train_time


def build_opt(opt, optimizees, flags):
    """This function builds optimizer."""
    if not isinstance(flags, dict):
        flags = vars(flags)

    opt.build(optimizees,
              n_bptt_steps=flags['n_bptt_steps'],
              loss_type=flags['loss_type'],
              optimizer=flags['optimizer'],
              lambd=flags['lambd'])


def test_configuration(opt, optimizees, flags):
    """
        This function runs testing of optimizer with given flags.
        Returns results and time of execution.
    """
    test_start_time = time.time()

    test_rets = []

    for opt_name in optimizees.keys():
        rets = opt.test(eid=flags.n_epochs,
                        n_batches=flags.n_batches,
                        n_steps=flags.n_steps,
                        opt_name=opt_name)
        test_rets.extend(rets)

    test_time = time.time() - test_start_time
    return test_rets, test_time


def make_opt(flags, optimizees, keys, val):
    """Initializes optimizer with given configuration."""
    configuration_hash = hash(val)

    kwargs = dict(zip(keys, val))
    configuration = vars(flags).copy()
    configuration.update(kwargs)
    del configuration['name']

    local_path = pathlib.Path('cv') / 'snapshots' / '{}.snapshot'.format(configuration_hash)

    opt = util.load_opt(flags.name, **configuration)
    opt.model_path = util.get_model_path(flags.name)
    opt.save_path = str(local_path)

    print(opt.save_path)

    subprocess.call(shlex.split('mkdir -p {}'.format(opt.model_path / local_path)))

    with tf.variable_scope('cv_scope_{}'.format(configuration_hash)):
        build_opt(opt, optimizees, configuration)

    return opt, configuration, configuration_hash


def process_configuration(flags, optimizees, keys, val, results):
    """
        This function makes optimizer, trains it, tests and returns various
        characteristics.
    """
    opt, configuration, val_hash = make_opt(flags, optimizees, keys, val)

    _, train_time = train_opt(opt, configuration)
    opt.save(flags.n_epochs)

    test_rets, test_time = test_configuration(opt, optimizees, flags)
    print(train_time, test_time)

    scores = get_score(test_rets)

    for key, value in zip(keys, val):
        results[key].append(value)

    for key, score in scores.items():
        results['score_{}'.format(key)].append(score)

    results['train_time'].append(train_time)
    results['test_time'].append(test_time)
    results['hash'].append(val_hash)
    results['params'].append(val)
    results['score'].append(np.nanmean(list(scores.values())))


def exhaustive_sampler(values):
    """Samples all combinations of values."""
    yield from itertools.product(*values)


def random_sampler(values, repeat):
    """Samples random combinations of values."""
    yield from random_product(*values, repeat=repeat)


def abstract_cv(params, flags, sampler):
    """Performs sampling of configurations and computes scores."""

    keys = params.keys()
    values = params.values()

    results = {}

    for k in itertools.chain(keys, ['train_time', 'test_time', 'hash', 'params', 'score']):
        results[k] = []

    for k in flags.optimizee:
        results['score_{}'.format(k)] = []

    rand_state = np.random.get_state()

    for val in sampler(values):
        graph = tf.Graph()
        session = tf.Session(config=util.get_tf_config(), graph=graph)
        with graph.as_default(), session:
            optimizees = optim.get_optimizees(flags.optimizee,
                                              clip_by_value=False,
                                              random_scale=flags.enable_random_scaling,
                                              noisy_grad=flags.noisy_grad)

            for optimizee in optimizees.values():
                optimizee.build()

            print(val)

            np.random.set_state(rand_state)
            process_configuration(flags, optimizees, keys, val, results)

    #best_index = np.argmax(results['score'])
    #best_score = results['score'][best_index]
    #best_params = results['params'][best_index]

    results['keys'] = list(keys)
    #results['best_index'] = best_index
    #results['best_params'] = best_params
    #results['best_score'] = best_score

    print(results)
    return results


def grid_cv(params, flags):
    """Performs exhaustive grid search over parameters"""
    return abstract_cv(params, flags, exhaustive_sampler)


def random_cv(params, flags, num_tries=5):
    """Performs random grid search over parameters"""
    sampler = lambda a: random_sampler(a, num_tries)
    return abstract_cv(params, flags, sampler)


def bayesian_cv(*args):
    """Performs bayesian cv"""
    raise NotImplementedError


def run_cv(flags):
    """Performs parameter tuning"""
    with open(flags.config, 'r') as conf:
        params = json.load(conf)
        params = OrderedDict(params)

    model_path = util.get_model_path(flags.name)
    with (model_path / 'cv' / 'run_config.json').open('w') as conf:
        run_config = vars(flags).copy()
        del run_config['command_name']
        json.dump(run_config, conf)

    if flags.method == 'grid':
        results = grid_cv(params, flags)
    elif flags.method == 'random':
        results = random_cv(params, flags, num_tries=flags.num_tries)
    else:
        results = bayesian_cv(params, flags)

    util.dump_results(model_path, results, phase='cv', tag=flags.tag)
