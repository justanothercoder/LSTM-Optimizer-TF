import time
import json
import itertools
from collections import OrderedDict

from toolz.itertoolz import toolz
import numpy as np
import tensorflow as tf

import optimizees as optim
import util
import util.paths as paths
import util.tf_utils as tf_utils


def get_score(rets):
    splits = toolz.groupby('optimizee_name', rets)

    scores = {}

    for opt_name in splits:
        losses = [ret['loss'] for ret in splits[opt_name]]
        scores[opt_name] = -np.mean(losses)

    return scores


def train_opt(opt, flags):
    session = tf.get_default_session()
    session.run(tf.global_variables_initializer(), feed_dict={
        opt.train_lr: flags.train_lr,
        opt.momentum: flags.momentum
    })

    build_config = BuildConfig.from_namespace(flags)
    build_config.test = False

    with log_execution_time('optimizer training'):
        train_rets, _ = opt.train(build_config)

    return train_rets


def test_configuration(opt, optimizees, flags):
    test_rets = []
    test_config = TestConfig(eid=flags.n_epochs, n_batches=flags.n_batches, n_steps=flags.n_steps)

    with log_execution_time('configuration testing'):
        for opt_name in optimizees.keys():
            test_config.opt_name = opt_name
            test_rets.extend(opt.test(test_config))

    return test_rets


def make_opt(flags, optimizees, val_hash):
    model_path = paths.model_path(flags.name)
    opt = util.load_opt(model_path)
    opt.snapshot_path = model_path / 'cv/snapshots/{}.snapshot'.format(val_hash)

    paths.make_dirs(opt.snapshot_path)
    print("Snapshot path: ", opt.snapshot_path)

    with tf.variable_scope('cv_scope_{}'.format(val_hash)):
        build_config = BuildConfig.from_namespace(flags)
        opt.build(optimizees, build_config)

    return opt


def process_configuration(flags, optimizees, keys, val, results):
    mapping = dict(zip(keys, val))

    configuration = vars(flags)
    configuration.update(mapping)
    del configuration['name']

    opt = make_opt(configuration, optimizees, hash(val))

    train_opt(opt, configuration)
    opt.save(flags.n_epochs)

    test_rets = test_configuration(opt, optimizees, flags)
    scores = get_score(test_rets)

    for key, value in mapping.items():
        results[key].append(value)

    for key, score in scores.items():
        results['score_{}'.format(key)].append(score)

    results['hash'].append(hash(val))
    results['params'].append(val)
    results['score'].append(np.mean(list(scores.values())))


def exhaustive_sampler(values):
    yield from itertools.product(*values)


def random_sampler(values, repeat):
    yield from util.random_product(*values, repeat=repeat)


@tf_utils.with_tf_graph
def cv_iteration(flags, keys, val, results):
    optimizees = optim.get_optimizees(flags.optimizee,
                                      clip_by_value=False,
                                      random_scale=flags.enable_random_scaling,
                                      noisy_grad=flags.noisy_grad)

    for optimizee in optimizees.values():
        optimizee.build()

    process_configuration(flags, optimizees, keys, val, results)


def abstract_cv(params, flags, sampler):

    keys = params.keys()
    values = params.values()

    results = {}

    for k in itertools.chain(keys, ['train_time', 'test_time', 'hash', 'params', 'score']):
        results[k] = []

    for k in flags.optimizee:
        results['score_{}'.format(k)] = []

    rand_state = np.random.get_state()

    for val in sampler(values):
        np.random.set_state(rand_state)
        print(val)
        cv_iteration(flags, keys, val, results)

    results['keys'] = list(keys)

    print(results)
    return results


def grid_cv(params, flags):
    return abstract_cv(params, flags, exhaustive_sampler)


def random_cv(params, flags, num_tries=5):
    sampler = lambda a: random_sampler(a, num_tries)
    return abstract_cv(params, flags, sampler)


def bayesian_cv(*_):
    raise NotImplementedError


def run_cv(flags):
    with open(flags.config, 'r') as conf:
        params = json.load(conf)
        params = OrderedDict(params)

    experiment_path = paths.experiment_path(flags.name, flags.experiment_name, 'cv')
    run_config_path = experiment_path / 'run_config.json'

    with run_config_path.open('w') as conf:
        run_config = vars(flags).copy()
        del run_config['command_name']
        json.dump(run_config, conf)

    if flags.method == 'grid':
        results = grid_cv(params, flags)
    elif flags.method == 'random':
        results = random_cv(params, flags, num_tries=flags.num_tries)
    else:
        results = bayesian_cv(params, flags)

    data = {
        'results': results,
        'keys': list(params.keys())
    }

    util.dump_results(experiment_path, data)
