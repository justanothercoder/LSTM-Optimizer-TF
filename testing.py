from collections import OrderedDict
import json

import numpy as np
import tensorflow as tf

from opts import SgdOpt, MomentumOpt, AdamOpt, InitConfig, RNNPropOpt
from opts import BuildConfig, TestConfig

import util
import util.tf_utils as tf_utils
import util.paths as paths
import optimizees as optim
    

def get_tests(test_problem, compare_with, with_rnnprop=False):
    def make_opt(name, learning_rate):
        opt_name = '{}_lr_{}'.format(name, learning_rate)

        reduce_config = InitConfig(lr=learning_rate, enable_reduce=True, patience_max=10, epsilon=1e-4, factor=0.5)
        init_config = InitConfig(lr=learning_rate)

        return {
            'sgd': SgdOpt,
            'momentum': MomentumOpt,
            'adam': AdamOpt,
        }[name](init_config, name=opt_name)

    lrs = np.logspace(start=-1, stop=-4, num=4)
    tests = [make_opt(compare_with, lr) for lr in lrs]

    if with_rnnprop:
        tests.append(RNNPropOpt(eid=340))

    return tests


def run_cv_testing(opt, flags):
    results = OrderedDict()
    random_state = np.random.get_state()

    for eid in range(flags.start_eid, flags.eid + 1, flags.step):
        np.random.set_state(random_state)

        test_config = TestConfig.from_namespace(flags)
        test_config.eid = eid
        rets = opt.test(test_config)

        name = '{name}_{eid}'.format(name=flags.name, eid=eid)
        results[name] = rets

    return results


def run_many_testing(opt, s_opts, flags):
    results = OrderedDict()
    random_state = np.random.get_state()

    for optimizer in [opt] + s_opts:
        np.random.set_state(random_state)

        test_config = TestConfig.from_namespace(flags)
        results[optimizer.name] = optimizer.test(test_config)

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
            raise RuntimeError("You will overwrite existing results. Add -f/--force to force it.")


    with (experiment_path / 'config').open('w') as conf:
        test_config = TestConfig.from_namespace(flags)
        test_config.model_path = str(model_path)
        json.dump(vars(test_config), conf, sort_keys=True, indent=4)

    opt = util.load_opt(model_path)
    return opt, experiment_path


def testing(flags, opt, s_opts, optimizees):
    graph = tf.Graph()
    with graph.as_default():
        if hasattr(flags, 'seed') and flags.seed is not None:
            tf.set_random_seed(flags.seed)
            
        for optimizee in optimizees.values():
            optimizee.build()
            
        build_config = BuildConfig(inference_only=True, n_bptt_steps=1)
        opt.build(optimizees, build_config)

        for i, s_opt in enumerate(s_opts):
            with tf.variable_scope('s_opt_{}'.format(i)):
                s_opt.build(optimizees, build_config)

        session = tf.Session(graph=graph, config=tf_utils.get_tf_config())
        with session.as_default():
            session.run(tf.global_variables_initializer())

            for i, s_opt in enumerate(s_opts):
                if hasattr(s_opt, 'eid'):
                    s_opt.session = tf.get_default_session()
                    s_opt.restore(s_opt.eid)

            if flags.mode == 'many':
                results = run_many_testing(opt, s_opts, flags)
            else:
                results = run_cv_testing(opt, flags)

    return results


def run_test(flags):
    opt, experiment_path = setup_experiment(flags)
    optimizees = optim.get_optimizees(flags.problems,
                                      clip_by_value=False,
                                      random_scale=flags.enable_random_scaling,
                                      noisy_grad=flags.noisy_grad)

    for opt_name in flags.problems:
        optimizee = optimizees[opt_name]
        print("Running testing on: ", opt_name)

        s_opts = get_tests(opt_name, flags.compare_with, with_rnnprop=flags.with_rnnprop)
        results = testing(flags, opt, s_opts, {opt_name: optimizee})

        data = dict(problem=opt_name, mode=flags.mode, results=results)

        prefix = opt_name + "_" + flags.mode
        if flags.mode == 'many':
            data['compare_with'] = flags.compare_with
            prefix += "_" + flags.compare_with

        util.dump_results(experiment_path, data, prefix=prefix)
