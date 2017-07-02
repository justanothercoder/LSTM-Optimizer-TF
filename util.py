import pickle
from lstm_opt import LSTMOpt


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


def dump_results(model_name, results, phase='train', **kwargs):
    filename = 'models/{model_name}/'.format(model_name=model_name)

    if phase == 'train':
        filename += 'train/results.pkl'
    elif phase == 'test':
        filename += 'test/{problem}_{mode}.pkl'.format(**kwargs)
    else:
        raise ValueError("Unknown phase: {}".format(phase))

    d = {
        'model_name': model_name,
        'phase': phase,
        'results': results,
    }
    d.update(**kwargs)

    with open(filename, 'wb') as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


def lstm_opt(optimizees, flags):
    if type(flags) is not dict:
        flags = vars(flags)

    used_kwargs = {'train_lr', 'n_bptt_steps', 'loss_type', 'stop_grad', 'add_skip', 'num_units', 'num_layers', 'name', 'layer_norm'}

    flags = {k: v for k, v in flags.items() if k in used_kwargs}

    #opt = LSTMOpt(optimizees, train_lr=flags.train_lr, 
    #                       n_bptt_steps=flags.n_bptt_steps, loss_type=flags.loss_type, stop_grad=flags.stop_grad, add_skip=flags.add_skip,
    #                       num_units=flags.num_units, num_layers=flags.num_layers, name=flags.name)

    opt = LSTMOpt(optimizees, **flags)
    return opt


