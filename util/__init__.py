import functools
import time
import random
import json
import pickle
from . import paths


def random_product(*args, repeat=1):
    "Random selection from itertools.product(*args, **kwds)"
    pools = [tuple(pool) for pool in args]
    result = []
    for _ in range(repeat):
        result.append(tuple(random.choice(pool) for pool in pools))
    return result


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


def dump_results(path, results, prefix=None):
    filename = 'results.pkl'
    if prefix:
        filename = prefix + filename

    results_path = path / filename

    with results_path.open('wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_results(path, prefix=None):
    filename = 'results.pkl'
    if prefix:
        filename = prefix + filename

    with (path / filename).open('rb') as f:
        results = pickle.load(f)

    return results


def load_opt(model_path, experiment_path):
    config_path = paths.config_path(model_path)
    with config_path.open('r') as conf:
        flags = json.load(conf)

    flags['model_path'] = experiment_path
    flags['snapshot_path'] = paths.snapshots_path(experiment_path)

    from opts.lstm_opt import LSTMOpt
    opt = LSTMOpt(**flags)
    return opt


def dump_config(path, flags, options):
    if not isinstance(flags, dict):
        flags = vars(flags)

    config = {k: v for k, v in flags.items() if k in options}
    print('Config: ', config)

    with path.open('w') as conf:
        json.dump(config, conf, sort_keys=True, indent=4)


def catch_keyboard_interrupt(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            pass
    return wrapper


def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t = time.time()
        ret = func(*args, **kwargs)
        t = time.time() - t
        return ret, t
    return wrapper


def run_new(flags):
    model_path = paths.model_path(flags.name)
    config_path = paths.config_path(model_path)

    if not flags.force and model_path.exists():
        print('Model already exists')
        return

    dirs = ['train', 'test', 'cv/snapshots']
    dirs = [str(model_path / d) for d in dirs]
    paths.make_dirs(dirs)

    model_parameters = {
        'num_layers', 'num_units',
        'layer_norm', 'name', 'stop_grad',
        'rnn_type', 'residual', 'normalize_gradients', 'rmsprop_gradients'
    }

    with config_path.open('w') as conf:
        d = {k: v for k, v in vars(flags).items() if k in model_parameters}
        json.dump(d, conf, sort_keys=True, indent=4)
