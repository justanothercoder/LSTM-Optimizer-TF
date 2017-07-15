import time
import json, pickle
import itertools
import random
import pathlib, subprocess, shlex
from collections import OrderedDict

import numpy as np
import tensorflow as tf

import util


def random_product(*args, repeat=1):
    "Random selection from itertools.product(*args, **kwds)"
    pools = [tuple(pool) for pool in args]
    result = []
    for _ in range(repeat):
        result.append(tuple(random.choice(pool) for pool in pools))
    return result


def get_score(rets):
    by_opt = lambda ret: ret['optimizee_name']
    splits, opt_names = util.split_list(rets, by_opt)

    scores = {}
    
    for i, opt_name in enumerate(opt_names):
        losses = [ret['loss'] for ret in splits[opt_name]]
        scores[opt_name] = -np.mean(losses)

    return scores


def train_opt(session, opt, flags):
    if type(flags) is not dict:
        flags = vars(flags)
    
    print("Running training with parameters: {}".format(flags))

    train_options = {
        'n_epochs': flags['n_epochs'],
        'n_batches': flags['n_batches'],
        'batch_size': flags['batch_size'],
        'n_steps': flags['n_steps'],
        'test': False,
        'train_lr': flags['train_lr'],
        'momentum': flags['momentum']
    }

    print('train_options: {}'.format(train_options))
        
    train_start_time = time.time()
    session.run(tf.global_variables_initializer(), feed_dict={
        opt.train_lr: train_options['train_lr'], 
        opt.momentum: train_options['momentum']
    })
    train_rets, _ = opt.train(**train_options)
    train_time = time.time() - train_start_time

    return train_rets, train_time


def build_opt(opt, optimizees, flags):
    if type(flags) is not dict:
        flags = vars(flags)

    options = {
        'n_bptt_steps': flags['n_bptt_steps'],
        'loss_type': flags['loss_type'],
        'optimizer': flags['optimizer'],
        'lambd': flags['lambd']
    }

    opt.build(optimizees, **options)


def test_configuration(opt, optimizees, flags):
    test_start_time = time.time()

    test_rets = []

    for opt_name in optimizees.keys():
        rets = opt.test(eid=flags.n_epochs, n_batches=flags.n_batches, n_steps=flags.n_steps, opt_name=opt_name)
        test_rets.extend(rets)
        
    test_time = time.time() - test_start_time
    return test_rets, test_time


def make_opt(flags, optimizees, keys, val):
    h = hash(val)

    kwargs = dict(zip(keys, val))
    d = vars(flags).copy()
    d.update(kwargs)

    local_path = pathlib.Path('cv') / 'snapshots' / '{}.snapshot'.format(h)

    opt = util.load_opt(flags.name, **d)
    opt.model_path = util.get_model_path(flags.name)
    opt.save_path = str(local_path)

    print(opt.save_path)

    subprocess.call(shlex.split('mkdir -p {}'.format(opt.model_path / local_path)))
        
    with tf.variable_scope('cv_scope_{}'.format(h)):
        build_opt(opt, optimizees, d)

    return opt, d, h


def process_configuration(session, flags, optimizees, keys, val, results):
    opt, d, val_hash = make_opt(flags, optimizees, keys, val)

    train_rets, train_time = train_opt(session, opt, d)
    opt.save(flags.n_epochs)

    test_rets, test_time = test_configuration(opt, optimizees, flags)
    print(train_time, test_time)

    scores = get_score(test_rets)

    for k, v in zip(keys, val):
        results[k].append(v) 

    for k, s in scores.items():
        results['score_{}'.format(k)].append(s)

    results['train_time'].append(train_time)
    results['test_time'].append(test_time)
    results['hash'].append(val_hash)
    results['params'].append(val)
    results['score'].append(np.nanmean(list(scores.values())))

    return opt


def exhaustive_sampler(values):
    yield from itertools.product(*values)


def random_sampler(values, n):
    yield from random_product(*values, repeat=n)


def cv(params, flags, sampler):
    keys = params.keys()
    values = params.values()

    results = { }

    for k in itertools.chain(keys, ['train_time', 'test_time', 'hash', 'params', 'score']):
        results[k] = []
                
    optimizees = util.get_optimizees(clip_by_value=True, random_scale=flags.enable_random_scaling)

    for k in optimizees.keys():
        results['score_{}'.format(k)] = []

    rand_state = np.random.get_state()

    #for i, val in enumerate(itertools.product(*values)):
    opt = None
    for val in sampler(values):
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(config=util.get_tf_config(), graph=graph) as session:
                optimizees = util.get_optimizees(clip_by_value=True, random_scale=flags.enable_random_scaling)

                for optimizee in optimizees.values():
                    optimizee.build()

                print(val)

                np.random.set_state(rand_state)
                opt = process_configuration(session, flags, optimizees, keys, val, results)

    best_index = np.argmax(results['score'])
    best_score = results['score'][best_index]
    best_params = results['params'][best_index]

    results['keys'] = list(keys)
    results['best_index' ] = best_index
    results['best_params'] = best_params
    results['best_score' ] = best_score

    print(results)
    return results


def grid_cv(params, flags):
    return cv(params, flags, exhaustive_sampler)


def random_cv(params, flags, num_tries=5):
    sampler = lambda a: random_sampler(a, num_tries)
    return cv(params, flags, sampler)


def bayesian_cv(*args):
    raise NotImplementedError


def run_cv(flags):
    with open(flags.config, 'r') as conf:
        params = json.load(conf)
        params = OrderedDict(params)

    model_path = util.get_model_path(flags.name) 

    with (model_path / 'cv' / 'run_config.json').open('w') as f:
        d = vars(flags).copy()
        del d['func']
        json.dump(d, f)

    if flags.method == 'grid':
        results = grid_cv(params, flags)
    elif flags.method == 'random':
        random_cv(params, flags, num_tries=flags.num_tries),
    else:
        bayesian_cv(params, flags)

    util.dump_results(model_path, results, phase='cv', tag=flags.tag)
