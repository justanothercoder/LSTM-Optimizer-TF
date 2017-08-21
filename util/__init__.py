import inspect
import random
import json
import pickle
import numpy as np
from . import paths

seed = None

def set_seed(seed_):
    global seed
    seed = seed_

    random.seed(seed)
    np.random.seed(seed)

    return seed


def get_seed():
    global seed
    return seed


def get_kwargs(func, flags):
    if not isinstance(flags, dict):
        flags = vars(flags)

    accepted_kwargs = set(inspect.signature(func).parameters.keys())
    kwargs = {k: v for k, v in flags.items() if k in accepted_kwargs}
    return kwargs


def random_product(*args, repeat=1):
    "Random selection from itertools.product(*args, **kwds)"
    pools = [tuple(pool) for pool in args]
    result = []
    for _ in range(repeat):
        result.append(tuple(random.choice(pool) for pool in pools))
    return result


def get_moving(values, mu=0.9):
    if len(values) == 0:
        raise ValueError("empty array")

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

    try:
        from pympler import asizeof
        print("Results size: {}MB".format(asizeof.asizeof(results) / 2**20))
    except:
        pass

    with results_path.open('wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_results(path, prefix=None):
    filename = 'results.pkl'
    if prefix:
        filename = prefix + filename

    with (path / filename).open('rb') as f:
        results = pickle.load(f)

    return results


def load_opt(model_path):
    with (model_path / 'config.json').open('r') as conf:
        flags = json.load(conf)

    flags['snapshot_path'] = model_path / 'snapshots'

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


def run_new(flags):
    model_path = paths.model_path(flags.name)
    config_path = model_path / 'config.json'

    if not flags.force and model_path.exists():
        print('Model already exists')
        return
    
    paths.make_dirs(str(model_path), str(model_path / 'snapshots'))

    model_parameters = {k for k in vars(flags) if k not in {'force', 'command_name'}}

    with config_path.open('w') as conf:
        d = {k: v for k, v in vars(flags).items() if k in model_parameters}
        json.dump(d, conf, sort_keys=True, indent=4)
