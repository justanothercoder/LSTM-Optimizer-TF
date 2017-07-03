import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import util
from util import split_list


def save_figure(fig, filename):
    fig.savefig('{filename}.svg'.format(**locals()), format='svg')
    os.system('convert {filename}.svg {filename}.png'.format(**locals()))
    print("Plotted to {filename}.svg and {filename}.png".format(**locals()))


def plot_test_results(flags, d):
    if flags.plot_lr:
        fig, axes = plt.subplots(nrows=3, figsize=(15, 12), sharex=True)
        (ax_f, ax_g, ax_lr) = axes
    else:
        fig, axes = plt.subplots(nrows=2, figsize=(15, 12), sharex=True)
        (ax_f, ax_g) = axes


    for name, rets in d['results'].items():
        #fxs = np.array([ret['fxs'] / ret['fxs'][0] for ret in rets])
        #norms = np.array([ret['norms'] / ret['norms'][0] for ret in rets])
        #lrs = np.array([ret['lrs'] for ret in rets])

        fxs = np.concatenate([
            np.reshape(ret['fxs'] / ret['fxs'][:1], (-1, 1))
            for ret in rets
        ], axis=-1).T

        norms = np.concatenate([
            np.reshape(ret['norms'] / ret['norms'][:1], (-1, 1))
            for ret in rets
        ], axis=-1).T

        lrs = np.concatenate([ret['lrs'] for ret in rets], axis=-1).T

        #if np.mean(lrs, axis=0)[-1] < -500:
        #    print("Skipping {}".format(name))
        #    continue

        #if fxs.max() > 1e5 or norms.max() > 1e5 or lrs.max() > 1e3:
        #    print("Skipped {}".format(name))
        #    continue

        l_test = int((1. - flags.frac) * fxs.shape[1])

        fxs   = fxs[:, l_test:]
        norms = norms[:, l_test:]
        lrs   = lrs[:, l_test:]

        mean_trajectory = np.mean(fxs, axis=0)
        std = np.std(fxs, axis=0)

        mean_norm = np.mean(norms, axis=0)
        std_norm = np.std(norms, axis=0)

        ax_f.semilogy(mean_trajectory, label=name)
        ax_g.semilogy(mean_norm, label=name)

        if flags.plot_lr:
            mean_lr = np.mean(lrs, axis=0)
            std_lr = np.std(lrs, axis=0)

            #p = ax_lr.plot(mean_lr, label=name)
            #ax_lr.fill_between(np.arange(mean_lr.shape[0]), mean_lr + std_lr, mean_lr - std_lr, alpha=0.3, facecolor=p[-1].get_color())

            p = ax_lr.semilogy(np.exp(mean_lr), label=name)
            ax_lr.fill_between(np.arange(mean_lr.shape[0]), np.exp(mean_lr + std_lr), np.exp(mean_lr - std_lr), alpha=0.3, facecolor=p[-1].get_color())

    axes[0].set_title(r'{problem}: mean $f(\theta_t), \|\nabla f(\theta_t)\|^2$ over {n_functions} functions for {n_steps} steps'.format(n_steps=fxs.shape[1], n_functions=fxs.shape[0], **d))

    ax_f.set_ylabel(r'function value: $\frac{f(\theta_t)}{f(\theta_0)}$')
    ax_g.set_ylabel(r'mean $\frac{\|\nabla f(\theta_t)\|^2}{\|\nabla f(\theta_0)\|^2}$')

    if flags.plot_lr:
        ax_lr.set_ylabel('mean learning rate')

    axes[-1].set_xlabel('iteration number')
    axes[0].legend(loc='best')

    fig.tight_layout()
    save_figure(fig, filename='models/{model_name}/test/{problem}_{mode}'.format(**d))


def plot_training_results(flags, d):
    by_opt = lambda ret: ret['optimizee_name']

    train_results, test_results = d['results']

    train_results_splits, opts = split_list(train_results, by_opt)
    test_results_splits , _    = split_list(test_results, by_opt)

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
            ax.plot(range(0, len(moving_train[l_train:]), s), moving_test[l_test:], label='moving test', color=p_test[-1].get_color())
            
        ax.set_title(opt_name)
        ax.set_ylabel('loss')
        ax.set_xlabel('iteration number')
        ax.legend(loc='best')

    fig.tight_layout()
    save_figure(fig, filename='models/{model_name}/train/training'.format(**d))


def run_plot(flags):
    if flags.phase == 'train':
        filename = 'models/{name}/train/results.pkl'
    elif flags.phase == 'test':
        filename = 'models/{name}/test/{problem}_{mode}.pkl'
    else:
        raise ValueError("Unknown phase: {}".format(flags.phase))

    with open(filename.format(**vars(flags)), 'rb') as f:
        d = pickle.load(f)

    if flags.phase == 'train':
        plotting.plot_training_results(flags, d)
    elif flags.phase == 'test':
        plotting.plot_test_results(flags, d)