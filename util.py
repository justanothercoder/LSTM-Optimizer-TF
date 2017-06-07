import pickle


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
