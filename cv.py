import time
import json, pickle
import itertools
import pathlib, subprocess, shlex
from collections import OrderedDict

import numpy as np
import tensorflow as tf

import util
from util import get_optimizees, lstm_opt


def get_score(rets):
    by_opt = lambda ret: ret['optimizee_name']
    splits, opt_names = util.split_list(rets, by_opt)

    scores = {}
    
    for i, opt_name in enumerate(opt_names):
        losses = [ret['loss'] for ret in splits[opt_name]]
        scores[opt_name] = -np.mean(losses)

    return scores


def grid_cv(session, optimizees, params, flags):
    keys = params.keys()
    values = params.values()

    results = { }

    for k in itertools.chain(keys, ['train_time', 'test_time', 'hash', 'params', 'score']):
        results[k] = []

    for k in optimizees.keys():
        results['score_{}'.format(k)] = []

    rand_state = np.random.get_state()

    for i, val in enumerate(itertools.product(*values)):
        np.random.set_state(rand_state)
        h = hash(val)

        kwargs = dict(zip(keys, val))
        d = vars(flags).copy()
        d.update(kwargs)

        local_path = pathlib.Path('cv') / 'snapshots' / '{}.snapshot'.format(h)

        d['model_path'] = util.get_model_path(flags.name) / local_path
        d['save_path'] = str(local_path)

        subprocess.call(shlex.split('mkdir -p {}'.format(d['model_path'])))

        opt = lstm_opt(optimizees, d)
        with tf.variable_scope('cv_scope_{}'.format(h)):
            opt.build()

        print("Running training with parameters: {}".format(d))

        train_start_time = time.time()
        session.run(tf.global_variables_initializer())
        train_rets, _ = opt.train(n_epochs=flags.n_epochs, n_batches=flags.n_batches, batch_size=flags.batch_size, n_steps=flags.n_steps, test=False)
        train_time = time.time() - train_start_time

        opt.save(flags.n_epochs)

        test_start_time = time.time()

        test_rets = []

        for opt_name in optimizees.keys():
            rets = opt.test(eid=flags.n_epochs, n_batches=flags.n_batches, n_steps=flags.n_steps, opt_name=opt_name)
            test_rets.extend(rets)
            
        test_time = time.time() - test_start_time

        print(train_time, test_time)

        scores = get_score(test_rets)

        for k, v in zip(keys, val):
            results[k].append(v) 

        for k, s in scores.items():
            results['score_{}'.format(k)].append(s)

        results['train_time'].append(train_time)
        results['test_time'].append(test_time)
        results['hash'].append(h)
        results['params'].append(val)
        results['score'].append(np.nanmean(list(scores.values())))

    best_index = np.argmax(results['score'])
    best_score = results['score'][best_index]
    best_params = results['params'][best_index]

    results['keys'] = list(keys)
    results['best_index' ] = best_index
    results['best_params'] = best_params
    results['best_score' ] = best_score

    print(results)
        
    return results


def random_cv(*args):
    raise NotImplementedError


def bayesian_cv(*args):
    raise NotImplementedError


def run_cv(flags):
    with open(flags.config, 'r') as conf:
        params = json.load(conf)
        params = OrderedDict(params)

    model_path = util.get_model_path(flags.name) 

    with (model_path / 'cv' / 'run_config.json').open('w') as f:
        d = vars(flags).copy()
        del d['model_path'], d['func']
        json.dump(d, f)

    graph = tf.Graph()

    cv_func = {
        'grid': grid_cv,
        'random': random_cv,
        'bayesian': bayesian_cv,
    }[flags.method]

    with graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4

        with tf.Session(config=config, graph=graph) as session:
            optimizees = get_optimizees(clip_by_value=True, random_scale=flags.enable_random_scaling)

            for optimizee in optimizees.values():
                optimizee.build()

            results = cv_func(session, optimizees, params, flags)

    util.dump_results(model_path, results, phase='cv')
