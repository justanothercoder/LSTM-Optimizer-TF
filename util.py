import pickle
from lstm_opt import LSTMOpt

import quadratic_optimizee, rosenbrock_optimizee, logistic_regression_optimizee
import optimizee_transformers


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


def dump_results(model_path, results, phase='train', **kwargs):
    if phase == 'train':
        results_path = model_path / 'train' / 'results.pkl'
    elif phase == 'test':
        results_path = model_path / 'test' / '{problem}_{mode}.pkl'.format(**kwargs)
    elif phase == 'cv':
        results_path = model_path / 'cv' / 'results.pkl'
    else:
        raise ValueError("Unknown phase: {}".format(phase))

    d = {
        'model_name': model_path.parts[-1],
        'phase': phase,
        'results': results,
    }
    d.update(**kwargs)

    with results_path.open('wb') as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


def lstm_opt(optimizees, flags):
    if type(flags) is not dict:
        flags = vars(flags)

    used_kwargs = {'train_lr', 'n_bptt_steps', 'loss_type', 'stop_grad', 'add_skip', 'num_units', 'num_layers', 'name', 'layer_norm', 'verbose'}

    flags = {k: v for k, v in flags.items() if k in used_kwargs}

    #opt = LSTMOpt(optimizees, train_lr=flags.train_lr, 
    #                       n_bptt_steps=flags.n_bptt_steps, loss_type=flags.loss_type, stop_grad=flags.stop_grad, add_skip=flags.add_skip,
    #                       num_units=flags.num_units, num_layers=flags.num_layers, name=flags.name)

    opt = LSTMOpt(optimizees, **flags)
    return opt


def get_optimizees(clip_by_value=True, random_scale=False):
    optimizees = {
        'quadratic': quadratic_optimizee.Quadratic(low=50, high=100),
        'rosenbrock': rosenbrock_optimizee.Rosenbrock(low=2, high=10),
        'logreg': logistic_regression_optimizee.LogisticRegression(max_data_size=1000, max_features=100),
    }

    optimizees['mixed'] = optimizee_transformers.ConcatAndSum([
        optimizees['quadratic'],
        optimizees['rosenbrock'],
        #optimizees['logreg']
    ])

    for name in optimizees:
        opt = optimizees[name]

        if random_scale:
            opt = optimizee_transformers.UniformRandomScaling(opt, r=3.0)

        if clip_by_value:
            opt = optimizee_transformers.ClipByValue(opt, clip_low=0, clip_high=10**10)

        optimizees[name] = opt

    return optimizees
