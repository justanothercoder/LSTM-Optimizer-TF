import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import toolz.itertoolz as toolz
import util
import util.paths as paths


def save_figure(fig, filename):
    filename = str(filename)
    fig.savefig('{filename}.svg'.format(filename=filename), format='svg')
    os.system('convert {filename}.svg {filename}.png'.format(filename=filename))
    print("Plotted to {filename}.svg and {filename}.png".format(filename=filename))


def extract_test_run_info(rets, flags, key, normalize, concat=True):
    vals = []
    for ret in rets:
        if key in ret:
            value = ret[key]
        else:
            return []

        if len(value) == 0:
            return []

        if normalize:
            value = value / ret[key][:1]
        vals.append(value)

    if concat:
        vals = np.concatenate(vals, axis=1)

    l_test = int((1. - flags.frac) * len(vals))
    vals = vals[l_test:]

    return vals


def setup_test_plot(flags):
    nrows = 1 + (1 - int(flags.stochastic)) + int(flags.plot_lr)
    fig, axes = plt.subplots(nrows=nrows, figsize=(15, 12), sharex=True)

    if nrows == 1:
        axes = (axes,)

    ax_f = axes[0]
    if not flags.stochastic:
        ax_g = axes[1]

    if flags.plot_lr:
        ax_lr = axes[2 - int(flags.stochastic)]


    if flags.stochastic:
        ax_f.set_ylabel(r'function value: $f(\theta_t)$')
        #ax_g.set_ylabel(r'mean $\|\nabla f(\theta_t)\|^2$')
    else:
        ax_f.set_ylabel(r'function value: $\frac{f(\theta_t)}{f(\theta_0)}$')
        ax_g.set_ylabel(r'mean $\frac{\|\nabla f(\theta_t)\|^2}{\|\nabla f(\theta_0)\|^2}$')

    if flags.plot_lr:
        ax_lr.set_ylabel('mean learning rate')

    axes[-1].set_xlabel('iteration number')

    fig.tight_layout()
    return fig, axes


def plot(ax, vals, name, logscale=True, with_moving=False):
    alpha = 1.0
    if with_moving:
        alpha = 0.3

    plot_func = ax.semilogy if logscale else ax.plot

    if with_moving:
        moving_vals = util.get_moving(vals, mu=0.95)
        plot_func(moving_vals, label=name)
    else:
        plot_func(vals, label=name, alpha=alpha)


def plot_test_results(flags, experiment_path, data):
    fig, axes = setup_test_plot(flags)

    for name, rets in data['results'].items():
        fxs = extract_test_run_info(rets, flags, 'values', not flags.stochastic)
        fxs_mean = fxs.mean(axis=1)

        norms_mean = extract_test_run_info(rets, flags, 'norms', not flags.stochastic).mean(axis=1)

        trainable_opt = not (name.startswith('adam') or name.startswith('sgd') or name.startswith('momentum'))
        if trainable_opt:
            lrs = extract_test_run_info(rets, flags, 'lrs', False, concat=False)
            if lrs:
                lrs_mean = [e.mean(axis=2) for e in lrs]
                lrs_max = [e.max(axis=2) for e in lrs]

                lrs_mean = np.concatenate(lrs_mean, axis=1)
                lrs_max = np.concatenate(lrs_max, axis=1)

                lrs_mean_mean = lrs_mean.mean(axis=1)
                lrs_mean_max = lrs_max.mean(axis=1)
                lrs_max_max = lrs_max.max(axis=1)

            cos_mean = extract_test_run_info(rets, flags, 'cosines', False)
            if cos_mean:
                cos_mean = cos_mean.mean(axis=1)

        cur_ax = 0

        plot(axes[cur_ax], fxs_mean, name, with_moving=flags.stochastic and flags.plot_moving)
        cur_ax += 1
        
        if not flags.stochastic:
            plot(axes[cur_ax], norms_mean, name, with_moving=flags.stochastic)
            cur_ax += 1

        if lrs and flags.plot_lr:
            axes[cur_ax].semilogy(np.exp(lrs_max_max), label=name)
            axes[cur_ax].semilogy(np.exp(lrs_mean_max), label=name)
            axes[cur_ax].semilogy(np.exp(lrs_mean_mean), label=name)

            cur_ax += 1
        
        #if trainable_opt:
        #    p = axes[cur_ax].plot(cos_mean, label=name)
        #    cur_ax += 1

    title = r"""{problem}: mean $f(\theta_t), \|\nabla f(\theta_t)\|^2$ over {} functions for {} steps"""
    title = title.format(fxs.shape[0], fxs.shape[1], problem=data['problem'])
    axes[0].set_title(title)
    axes[0].legend(loc='best')

    filename = '{problem}_{mode}'.format(**data)
    path = experiment_path / filename
    save_figure(fig, path)


def plot_training_results(flags, experiment_path, results):
    train_results, test_results = results

    train_results = toolz.groupby('optimizee_name', train_results)
    test_results = toolz.groupby('optimizee_name', test_results)

    opts = list(train_results.keys())

    for opt_name, rets in train_results.items():
        print("{}: {} iterations".format(opt_name, len(rets)))

    fig, axes = plt.subplots(nrows=len(opts), figsize=(15, 12), squeeze=False)

    alpha = 1.0
    if flags.plot_moving:
        alpha = 0.5

    for i, opt_name in enumerate(opts):
        ax = axes[i][0]

        losses_train = [ret['loss'] for ret in train_results[opt_name]]
        losses_test  = [ret['loss'] for ret in test_results[opt_name]]

        l_train = int(len(losses_train) * (1. - flags.frac))
        l_test = int(len(losses_test) * (1. - flags.frac))

        if flags.plot_moving:
            moving_train = util.get_moving(losses_train, mu=0.95)[l_train:]
            moving_test = util.get_moving(losses_test, mu=0.95)[l_test:]

        losses_train = losses_train[l_train:]
        losses_test  = losses_test[l_test:]

        s = len(losses_train) // len(losses_test)
        lt = list(range(0, len(losses_train), s))
        lt = lt[:len(losses_test)]

        p_train = ax.plot(losses_train, label='train', alpha=alpha)
        p_test = ax.plot(lt, losses_test, label='test', alpha=alpha)

        if flags.plot_moving:
            ax.plot(moving_train, label='moving train', color=p_train[-1].get_color())
            ax.plot(lt, moving_test, label='moving test', color=p_test[-1].get_color())

        ax.set_title(opt_name)
        ax.set_ylabel('loss')
        ax.set_xlabel('iteration number')
        ax.legend(loc='best')

    fig.tight_layout()

    save_figure(fig, filename=experiment_path / 'training')


def plot_cv_results(flags, experiment_path, data):
    data = data['results']
    keys = data['keys']

    aux_keys = {
        'keys', 'best_index', 'best_params',
        'best_score', 'train_time', 'hash',
        'opts', 'params'
    }

    data = {k: v for k, v in data.items() if k not in aux_keys}
    data = pd.DataFrame(data)

    n_keys = len(keys)
    n_keys = math.ceil(np.sqrt(n_keys))

    fig, axes = plt.subplots(nrows=n_keys, ncols=n_keys, figsize=(15, 12))

    if n_keys == 1:
        axes = ((axes,),)

    for i, key in enumerate(keys):
        ax = axes[i // n_keys][i % n_keys]
        sns.barplot(x=key, y='score', data=data, ax=ax)

    save_figure(fig, experiment_path / 'summary')


def run_plot_test(flags):
    experiment_path = paths.experiment_path(flags.name)

    for problem in flags.problems:
        prefix = problem + '_' + flags.mode
        if flags.mode == 'many':
            prefix += '_' + flags.compare_with

        data = util.load_results(experiment_path, prefix=prefix)
        plot_test_results(flags, experiment_path, data)


def run_plot(flags):
    if flags.phase == 'test':
        run_plot_test(flags)
        return

    model_path = paths.model_path(flags.name)
    data = util.load_results(model_path)

    plot_func = {
        'train': plot_training_results,
        'cv': plot_cv_results
    }[flags.phase]

    plot_func(flags, model_path, data)
