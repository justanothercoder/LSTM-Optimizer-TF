import os
import math
import numpy as np
import pandas as pd

import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import util
from util import split_list


def save_figure(fig, filename):
    fig.savefig('{filename}.svg'.format(**locals()), format='svg')
    os.system('convert {filename}.svg {filename}.png'.format(**locals()))
    print("Plotted to {filename}.svg and {filename}.png".format(**locals()))


def extract_test_run_info(rets, flags, key, normalize):
    vals = []
    for ret in rets:
        v = ret[key]
        if key == 'lrs':
            v = np.mean(v, axis=1)
        if normalize:
            v = v / ret[key][:1]
        vals.append(v.reshape(-1, 1))
    vals = np.concatenate(vals, axis=-1).T

    l_test = int((1. - flags.frac) * vals.shape[1])
    vals = vals[:, l_test:]

    mean = np.mean(vals, axis=0)
    std  = np.std(vals, axis=0)

    return vals, mean, std


def setup_test_plot(flags, d):
    if flags.plot_lr:
        fig, axes = plt.subplots(nrows=3, figsize=(15, 12), sharex=True)
        (ax_f, ax_g, ax_lr) = axes
    else:
        fig, axes = plt.subplots(nrows=2, figsize=(15, 12), sharex=True)
        (ax_f, ax_g) = axes
    

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
        alpha = 0.5

    plot_func = ax.semilogy if logscale else ax.plot

    p = plot_func(vals, label=name, alpha=alpha)

    if with_moving:
        moving_vals = util.get_moving(vals, mu=0.95)
        plot_func(moving_vals, label='moving {}'.format(name), color=p[-1].get_color())


def plot_test_results(flags, d):
    fig, axes = setup_test_plot(flags, d)
    ax_f, ax_g, ax_lr = axes

    for name, rets in d['results'].items():
        fxs, fxs_mean, fxs_std       = extract_test_run_info(rets, flags, 'fxs', not flags.stochastic)
        norms, norms_mean, norms_std = extract_test_run_info(rets, flags, 'norms', not flags.stochastic)
        lrs, lrs_mean, lrs_std       = extract_test_run_info(rets, flags, 'lrs', False)

        plot(ax_f, fxs_mean, name, with_moving=flags.stochastic)
        plot(ax_g, norms_mean, name, with_moving=flags.stochastic)

        if flags.plot_lr:
            p = ax_lr.plot(lrs_mean, label=name)
            ax_lr.fill_between(np.arange(lrs_mean.shape[0]), lrs_mean + lrs_std, lrs_mean  - lrs_std, alpha=0.3, facecolor=p[-1].get_color())
    
    axes[0].set_title(r'{problem}: mean $f(\theta_t), \|\nabla f(\theta_t)\|^2$ over {n_functions} functions for {n_steps} steps'.format(n_steps=fxs.shape[1], n_functions=fxs.shape[0], **d))
    axes[0].legend(loc='best')

    model_path = util.get_model_path(flags.name)

    filename = '{problem}_{mode}'
    if flags.tag:
        filename += '_{tag}'
    filename = filename.format(**d)

    save_figure(fig, filename=str(model_path / 'test' / filename))


def plot_training_results(flags, d):
    by_opt = lambda ret: ret['optimizee_name']

    train_results, test_results = d['results']

    train_results_splits, opts = split_list(train_results, by_opt)
    test_results_splits , _    = split_list(test_results, by_opt)

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
        losses_test  = [ret['loss'] for ret in test_results_splits [opt_name]]

        l_train = int(len(losses_train) * (1. - flags.frac))
        l_test = int(len(losses_test) * (1. - flags.frac))

        s = len(losses_train[l_train:]) // len(losses_test[l_test:])

        lt = list(range(0, len(losses_train[l_train:]), s))

        p_train = ax.plot(losses_train[l_train:], label='train', alpha=alpha)
        p_test  = ax.plot(lt[:len(losses_test[l_test:])], losses_test[l_test:], label='test', alpha=alpha)

        if flags.plot_moving:
            moving_train = util.get_moving(losses_train, mu=0.95)
            moving_test  = util.get_moving(losses_test,  mu=0.95)

            ax.plot(moving_train[l_train:], label='moving train', color=p_train[-1].get_color())
            #ax.plot(range(0, len(moving_train[l_train:]), s), moving_test[l_test:], label='moving test', color=p_test[-1].get_color())
            ax.plot(lt[:len(moving_test[l_test:])], moving_test[l_test:], label='moving test', color=p_test[-1].get_color())
            
        ax.set_title(opt_name)
        ax.set_ylabel('loss')
        ax.set_xlabel('iteration number')
        ax.legend(loc='best')

    fig.tight_layout()
    #save_figure(fig, filename='models/{model_name}/train/training'.format(**d))
    model_path = util.get_model_path(flags.name)
    save_figure(fig, filename=str(model_path / 'train' / 'training'))


def plot_cv_results(flags, d):
    d = d['results']
    keys = d['keys']

    aux_keys = {'keys', 'best_index', 'best_params', 'best_score', 'train_time', 'hash', 'opts', 'params'}

    df = {k: v for k, v in d.items() if k not in aux_keys}
    df = pd.DataFrame(df)

    n = len(keys) 
    n = math.ceil(np.sqrt(n))

    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(15, 12)) 

    for i, key in enumerate(keys):
        ax = axes[i // n][i % n]
        sns.barplot(x=key, y='score', data=df, ax=ax)

    model_path = util.get_model_path(flags.name)
    save_figure(fig, filename=str(model_path / 'cv' / 'summary'))


def run_plot(flags):
    model_path = util.get_model_path(flags.name)
    path = model_path / flags.phase

    accepted_keys = {'phase', 'problem', 'mode', 'tag'}
    kwargs = {k: v for k, v in vars(flags).items() if k in accepted_keys and v is not None}

    d = util.load_results(model_path, **kwargs)

    plot_func = {
        'train': plot_training_results,
        'test': plot_test_results,
        'cv': plot_cv_results
    }[flags.phase]
    
    plot_func(flags, d)
