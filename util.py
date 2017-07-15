import pathlib
import subprocess, shlex
import json, pickle


def get_tf_config():
    import tensorflow as tf
    
    config = tf.ConfigProto(allow_soft_placement=True)
#    config.device_count.CPU = 8
#    config.inter_op_parallelism_threads = 1
#    config.intra_op_parallelism_threads = 1

    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True
    return config


def get_moving(values, mu=0.9):
    v = [values[0]]

    for i in values[1:]:
        v.append(mu * v[-1] + (1. - mu) * i)

    return v


def split_list(lst, descr):
    splits = { }

    keys = set()

    for item in lst:
        key = descr(item)
        keys.add(key)
        splits.setdefault(key, []).append(item)

    return splits, keys


def dump_results(model_path, results, **kwargs):
    assert kwargs.get('phase') in ['train', 'test', 'cv']
    results_path = model_path / kwargs['phase']
    
    tags = [kwargs[k] for k in sorted(kwargs) if k != 'phase' and kwargs[k] is not None]
    filename = 'results' + '_{}' * len(tags)
    filename = (filename + '.pkl').format(*tags)

    d = {
        'model_name': model_path.parts[-1],
        'results': results,
    }
    d.update(**kwargs)


    with (results_path / filename).open('wb') as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_results(model_path, **kwargs):
    assert kwargs.get('phase') in ['train', 'test', 'cv']
    results_path = model_path / kwargs['phase']

    tags = [kwargs[k] for k in sorted(kwargs) if k != 'phase' and kwargs[k] is not None]

    filename = 'results' + '_{}' * len(tags)
    filename = (filename + '.pkl').format(*tags)

    with (results_path / filename).open('rb') as f:
        d = pickle.load(f)

    return d


def load_opt(name, **kwargs):
    conf_path = get_model_path(name) / 'model_config.json'
    with conf_path.open('r') as conf:
        flags = json.load(conf)

    for name in flags:
        if kwargs.get(name) is not None:
            flags[name] = kwargs[name]

    from opts.lstm_opt import LSTMOpt
    opt = LSTMOpt(**flags)
    return opt


def get_optimizees(clip_by_value=False, random_scale=False, noisy_grad=False):
    from optimizees.quadratic import Quadratic
    from optimizees.rosenbrock import Rosenbrock
    from optimizees.logistic_regression import LogisticRegression
    from optimizees.stochastic_logistic_regression import StochasticLogisticRegression
    from optimizees.stochastic_linear_regression import StochasticLinearRegression

    import optimizees.transformers as transformers

    optimizees = {
        'quadratic'   : Quadratic(low=50, high=100),
        'rosenbrock'  : Rosenbrock(low=2, high=10),
        'logreg'      : LogisticRegression(max_data_size=1000, max_features=100),
        'stoch_logreg': StochasticLogisticRegression(max_data_size=1000, max_features=100),
        'stoch_linear': StochasticLinearRegression(max_data_size=1000, max_features=100)
    }

    optimizees['mixed'] = transformers.ConcatAndSum([
        optimizees['quadratic'],
        optimizees['rosenbrock'],
        #optimizees['logreg'],
        #optimizees['stoch_logreg'],
    ])
    
    optimizees['mixed_stoch'] = transformers.ConcatAndSum([
        optimizees['quadratic'],
        optimizees['rosenbrock'],
        optimizees['stoch_logreg'],
        optimizees['stoch_linear'],
    ])

    for name in optimizees:
        opt = optimizees[name]

        if random_scale:
            opt = transformers.UniformRandomScaling(opt, r=3.0)

        if clip_by_value:
            opt = transformers.ClipByValue(opt, clip_low=0, clip_high=10**10)

        if noisy_grad and not name.startswith('stoch'):
            opt = transformers.NormalNoisyGrad(opt, stddev=0.1)

        optimizees[name] = opt

    return optimizees


def get_model_path(name):
    path = pathlib.Path('models') / name
    return path


def run_new(flags):
    path = get_model_path(flags.name)

    if not flags.force and path.exists():
        print('Model already exists')
        return

    paths = ['train', 'test', 'cv/snapshots', 'tf_data', 'snapshots']
    command = "mkdir -p " + ' '.join(str(path / p) for p in paths)
    subprocess.call(shlex.split(command))

    model_parameters = {'num_layers', 'num_units', 'layer_norm', 'name', 'stop_grad', 'rnn_type'}

    with (path / 'model_config.json').open('w') as conf:
        d = {k: v for k, v in vars(flags).items() if k in model_parameters}
        json.dump(d, conf, sort_keys=True, indent=4)
