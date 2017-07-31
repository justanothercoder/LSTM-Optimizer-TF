from collections import OrderedDict
import sys
import json
import numpy as np
import tensorflow as tf

from opts import distributed
from opts.sgd_opt import SgdOpt
from opts.momentum_opt import MomentumOpt
from opts.adam_opt import AdamOpt

import util
import util.tf_utils as tf_utils
import util.paths as paths
import optimizees as optim


def get_tests(test_problem, compare_with):
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
        'digits_classifier', 'mnist_classifier', 'digits_classifier_2',
        'digits_classifier_relu', 'digits_classifier_relu_2',
        'conv_digits_classifier', 'conv_digits_classifier_2',
        'digits_classifier_3', 'digits_classifier_relu_3',
    }

    opts = {'sgd', 'momentum', 'adam'}

    lrs = np.logspace(start=-1, stop=-5, num=5)
    tests = {}
    for problem in problems:
        tests[problem] = {}
        for opt in opts:
            tests[problem][opt] = [make_opt(opt, lr) for lr in lrs]

        #if problem == 'matyas':
        #    tests['matyas'][opt] = [make_opt(opt, lr) for lr in np.logspace(start=-5, stop=-9, num=5)]

    return tests[test_problem][compare_with]


def run_cv_testing(opt, flags):
    results = OrderedDict()
    random_state = np.random.get_state()

    for eid in range(flags.start_eid, flags.eid + 1, flags.step):
        np.random.set_state(random_state)

        kwargs = util.get_kwargs(opt.test, flags)
        kwargs['eid'] = eid
        rets = opt.test(**kwargs)

        name = '{name}_{eid}'.format(name=flags.name, eid=eid)
        results[name] = rets

    return results


def run_many_testing(opt, s_opts, flags):
    results = OrderedDict()
    random_state = np.random.get_state()

    for optimizer in [opt] + s_opts:
        np.random.set_state(random_state)
        kwargs = util.get_kwargs(optimizer.test, flags)
        results[optimizer.name] = optimizer.test(**kwargs)

    return results


def setup_experiment(flags):
    if flags.eid == 0:
        raise ValueError("eid must be > 0")

    model_path = paths.model_path(flags.name)
    experiment_path = paths.experiment_path(flags.experiment_name)

    print("Model path: ", model_path)
    print("Experiment path: ", experiment_path)

    paths.make_dirs(experiment_path)
    
    for opt_name in flags.problems:
        prefix = opt_name + "_" + flags.mode

        if flags.mode == 'many':
            prefix += "_" + flags.compare_with

        if not flags.force and (experiment_path / (prefix + '_results.pkl')).exists():
            print("You will overwrite existing results. Add -f/--force to force it.")
            sys.exit(1)


    with (experiment_path / 'config').open('w') as conf:
        testing_options = {'eid', 'n_batches', 'n_steps', 'problems'}
        config = {k: v for k, v in vars(flags).items() if k in testing_options}
        config['model_path'] = str(model_path)
        json.dump(config, conf, sort_keys=True, indent=4)

    opt = util.load_opt(model_path)
    return experiment_path, opt


@tf_utils.with_tf_graph
def testing(flags, opt, s_opts, optimizees):
    for optimizee in optimizees.values():
        optimizee.build()

    opt.build(optimizees, inference_only=True)

    for i, s_opt in enumerate(s_opts):
        #s_opt.build(optimizees, inference_only=True, devices=tf_utils.get_devices(flags))
        with tf.variable_scope('s_opt_{}'.format(i)):
            s_opt.build(optimizees, inference_only=True)

    session = tf.get_default_session()
    session.run(tf.global_variables_initializer())

    if flags.mode == 'many':
        results = run_many_testing(opt, s_opts, flags)
    else:
        results = run_cv_testing(opt, flags)

    return results


def run_test(flags):
    if flags.problems is None or flags.problems == 'all':
        flags.problems = [
            'rosenbrock', 'quadratic',
            'beale', 'booth', 'matyas',
            'logreg',
            'stoch_logreg', 'stoch_linear'
        ]

    
    for problem in flags.problems:
        assert problem in optim.problems

    experiment_path, opt = setup_experiment(flags)

    opt = distributed.distribute(opt, tf_utils.get_devices(flags))
    optimizees = optim.get_optimizees(flags.problems,
                                      clip_by_value=False,
                                      random_scale=flags.enable_random_scaling,
                                      noisy_grad=flags.noisy_grad)

    for opt_name, optimizee in optimizees.items():
        s_opts = get_tests(opt_name, flags.compare_with)
        results = testing(flags, opt, s_opts, {opt_name: optimizee})

        data = {
            'problem': opt_name,
            'mode': flags.mode,
            'results': results
        }

        prefix = opt_name + "_" + flags.mode
        if flags.mode == 'many':
            data['compare_with'] = flags.compare_with
            prefix += "_" + flags.compare_with

        util.dump_results(experiment_path, data, prefix=prefix)
