"""
    This module contains various plotting functions.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import util


def save_figure(fig, filename):
    """
        This function saves figure to files.
        First it saves to .svg and then converts to .png.
    """
    fig.savefig('{filename}.svg'.format(**locals()), format='svg')
    os.system('convert {filename}.svg {filename}.png'.format(filename=filename))
    print("Plotted to {filename}.svg and {filename}.png".format(filename=filename))


def extract_test_run_info(rets, flags, key, normalize):
    """
        This function extracts function values, gradient norms and learning rates
    """
    vals = []
    for ret in rets:
        value = ret[key]
        if key == 'lrs':
            value = np.mean(value, axis=1)
        if normalize:
            value = value / ret[key][:1]
        vals.append(value.reshape(-1, 1))
    vals = np.concatenate(vals, axis=-1).T

    l_test = int((1. - flags.frac) * vals.shape[1])
    vals = vals[:, l_test:]

    mean = np.mean(vals, axis=0)
    std = np.std(vals, axis=0)

    return vals, mean, std


def setup_test_plot(flags):
    """This function setups plot."""

    nrows = 1 + (1 - int(flags.stochastic)) + int(flags.plot_lr)
    fig, axes = plt.subplots(nrows=nrows, figsize=(15, 12), sharex=True)

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
    """
        This function plots values on ax.
    """
    alpha = 1.0
    if with_moving:
        alpha = 0.3

    plot_func = ax.semilogy if logscale else ax.plot

    p = plot_func(vals, label=name, alpha=alpha)

    if with_moving:
        moving_vals = util.get_moving(vals, mu=0.95)
        plot_func(moving_vals, label='moving {}'.format(name), color=p[-1].get_color())


def plot_test_results(flags, data):
    """This function plots tests results."""
    fig, axes = setup_test_plot(flags)

    for name, rets in data['results'].items():
        fxs, fxs_mean, _ = extract_test_run_info(rets, flags, 'fxs', not flags.stochastic)
        _, norms_mean, _ = extract_test_run_info(rets, flags, 'norms', not flags.stochastic)

        trainable_opt = not (name.startswith('adam') or name.startswith('sgd') or name.startswith('momentum'))
        if trainable_opt:
            _, lrs_mean, lrs_std = extract_test_run_info(rets, flags, 'lrs', False)

        plot(axes[0], fxs_mean, name, with_moving=flags.stochastic)
        if not flags.stochastic:
            plot(axes[1], norms_mean, name, with_moving=flags.stochastic)

        if trainable_opt and flags.plot_lr:
            p = axes[2 - int(flags.stochastic)].plot(lrs_mean, label=name)
            axes[2 - int(flags.stochastic)].fill_between(np.arange(lrs_mean.shape[0]),
                                                        lrs_mean + lrs_std,
                                                        lrs_mean  - lrs_std,
                                                        alpha=0.3,
                                                        facecolor=p[-1].get_color())

    print(fxs_mean.shape)

    title = r"""{problem}: mean $f(\theta_t), \|\nabla f(\theta_t)\|^2$ over {} functions for {} steps"""
    title = title.format(fxs.shape[0], fxs.shape[1], **data)
    axes[0].set_title(title)
    axes[0].legend(loc='best')

    model_path = util.get_model_path(flags.name)

    filename = '{problem}_{mode}'
    if flags.tag:
        filename += '_{tag}'
    filename = filename.format(**data)

    save_figure(fig, filename=str(model_path / 'test' / filename))


def plot_training_results(flags, data):
    """
        This function plots training results.
    """
    by_opt = lambda ret: ret['optimizee_name']

    train_results, test_results = data['results']

    train_results_splits, opts = util.split_list(train_results, by_opt)
    test_results_splits, _ = util.split_list(test_results, by_opt)

    for opt_name, rets in train_results_splits.items():
        print("{}: {} iterations".format(opt_name, len(rets)))

    fig, axes = plt.subplots(nrows=len(opts), figsize=(15, 12))

    if len(opts) == 1:
        axes = (axes,)

    alpha = 1.0
    if flags.plot_moving:
        alpha = 0.5

    for i, opt_name in enumerate(opts):
        ax = axes[i]

        losses_train = [ret['loss'] for ret in train_results_splits[opt_name]]
        losses_test = [ret['loss'] for ret in test_results_splits[opt_name]]

        l_train = int(len(losses_train) * (1. - flags.frac))
        l_test = int(len(losses_test) * (1. - flags.frac))

        if flags.plot_moving:
            moving_train = util.get_moving(losses_train, mu=0.95)[l_train:]
            moving_test = util.get_moving(losses_test, mu=0.95)[l_test:]

        losses_train = losses_train[l_train:]
        losses_test = losses_test[l_test:]

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
    model_path = util.get_model_path(flags.name)
    save_figure(fig, filename=str(model_path / 'train' / 'training'))


def plot_cv_results(flags, data):
    """
        This function plots validation results.
    """
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

    model_path = util.get_model_path(flags.name)
    save_figure(fig, filename=str(model_path / 'cv' / 'summary'))


def run_plot(flags):
    """
        This function handles command-line arguments and runs plotting.
    """
    accepted_keys = {'phase', 'problem', 'mode', 'tag', 'compare_with'}
    kwargs = {k: v for k, v in vars(flags).items() if k in accepted_keys and v is not None}

    data = util.load_results(util.get_model_path(flags.name), **kwargs)

    plot_func = {
        'train': plot_training_results,
        'test': plot_test_results,
        'cv': plot_cv_results
    }[flags.phase]

    plot_func(flags, data)
