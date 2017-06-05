#!/usr/bin/env python

import os
import argparse
import pprint
import shutil
import pickle
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import basic_model

from lstm_opt import LSTMOpt
from sgd_opt import SgdOpt
from momentum_opt import MomentumOpt

import quadratic_optimizee
import rosenbrock_optimizee


optimizees = {
    'quadratic': quadratic_optimizee.Quadratic(low=50, high=100),
    'rosenbrock': rosenbrock_optimizee.Rosenbrock(low=2, high=10)
}


def run_train(flags):
    graph = tf.Graph()

    with graph.as_default():
        with tf.Session(graph=graph) as session:

            opt = LSTMOpt(optimizees, train_lr=flags.train_lr, 
                                   n_bptt_steps=flags.n_bptt_steps, loss_type=flags.loss_type, 
                                   num_units=flags.num_units, num_layers=flags.num_layers, name=flags.name)

            for optimizee in optimizees.values():
                optimizee.build()

            opt.build()

            session.run(tf.global_variables_initializer())
            opt.train(n_epochs=flags.n_epochs, n_batches=flags.n_batches, n_steps=flags.n_steps, eid=flags.eid)


def run_test(flags):
    graph = tf.Graph()

    with graph.as_default():
        with tf.Session(graph=graph) as session:
            optimizee = {flags.problem: optimizees[flags.problem]}
            opt = LSTMOpt(optimizee, num_units=flags.num_units, num_layers=flags.num_layers, name=flags.name)
            
            if flags.problem == 'quadratic':
                s_opts = [MomentumOpt(optimizee, lr=16 * 2**(-i), name='momentum_opt_lr_{}'.format(4-i)) for i in range(0, 6)]
            elif flags.problem == 'rosenbrock':
                s_opts = [MomentumOpt(optimizee, lr=2**(-i-9), name='momentum_opt_lr_{}'.format(-i-9)) for i in range(1, 3)]

            optimizees[flags.problem].build()
            opt.build()

            for s_opt in s_opts:
                s_opt.build()

            session.run(tf.global_variables_initializer())

            if flags.eid == 0:
                print("eid must be > 0 if mode is testing")
                return

            st = np.random.get_state()

            results = OrderedDict()

            if flags.mode == 'many':
                for o in [opt] + s_opts:
                    np.random.set_state(st)
                    rets = o.test(eid=flags.eid, n_batches=flags.n_batches, n_steps=flags.n_steps)
                    results[o.name] = rets
            else:
                for eid in range(100, flags.eid + 1, 100):
                    np.random.set_state(st)
                    rets = opt.test(eid=eid, n_batches=flags.n_batches, n_steps=flags.n_steps)

                    name = '{name}_{eid}'.format(name=opt.name, eid=eid)
                    results[name] = rets

            filename = '{}_{}_results.pkl'.format(flags.name, flags.problem)
            with open(filename, 'wb') as res_file:
                d = {
                    'problem': flags.problem,
                    'n_steps': flags.n_steps,
                    'results': results,
                    'n_functions': flags.n_batches,
                }
                pickle.dump(d, res_file, protocol=pickle.HIGHEST_PROTOCOL)

            
def run_plot(flags):
    with open(flags.filename, 'rb') as f:
        d = pickle.load(f)

    if flags.plot_lr:
        fig, axes = plt.subplots(nrows=3, figsize=(15, 12), sharex=True)
        (ax_f, ax_g, ax_lr) = axes
    else:
        fig, axes = plt.subplots(nrows=2, figsize=(15, 12), sharex=True)
        (ax_f, ax_g) = axes

    #fig.subplots_adjust(hspace=0)

    #ax_t = ax.twinx()

    for name, rets in d['results'].items():
        fxs = np.array([ret['fxs'] / ret['fxs'][0] for ret in rets])
        norms = np.array([ret['norms'] / ret['norms'][0] for ret in rets])
        #lrs = np.array([ret['lrs'] / ret['lrs'][0] for ret in rets])
        lrs = np.array([ret['lrs'] for ret in rets])

        if fxs.max() > 1e5 or norms.max() > 1e5 or lrs.max() > 1e3:
            continue

        mean_trajectory = np.mean(fxs, axis=0)
        std = np.std(fxs, axis=0)

        mean_norm = np.mean(norms, axis=0)
        std_norm = np.std(norms, axis=0)

        ax_f.semilogy(mean_trajectory, label=name)
        ax_g.semilogy(mean_norm, label=name)

        if flags.plot_lr:
            mean_lr = np.mean(lrs, axis=0)
            std_lr = np.std(lrs, axis=0)

            p = ax_lr.plot(mean_lr, label=name)
            ax_lr.fill_between(np.arange(mean_lr.shape[0]), mean_lr + std_lr, mean_lr - std_lr, alpha=0.3, facecolor=p[-1].get_color())

        #plt.plot(mean_trajectory, label="{name}_{epoch}".format(name=opt.name, epoch=eid))
        #plt.fill_between(np.arange(rets.shape[1]), mean_trajectory + std, mean_trajectory - std, alpha=0.5)

    axes[0].set_title(r'{problem}: mean $f(\theta_t), \|\nabla f(\theta_t)\|^2$ over {n_functions} functions for {n_steps} steps'.format(**d))

    ax_f.set_ylabel(r'function value: $\frac{f(\theta_t)}{f(\theta_0)}$')
    ax_g.set_ylabel(r'mean $\frac{\|\nabla f(\theta_t)\|^2}{\|\nabla f(\theta_0)\|^2}$')

    if flags.plot_lr:
        ax_lr.set_ylabel('mean learning rate')

    axes[-1].set_xlabel('iteration number')
    axes[0].legend(loc='best')

    fig.savefig('{}_plot_test.svg'.format(d['problem']), format='svg')
    fig.tight_layout()
    os.system('convert {problem}_plot_test.svg {problem}_plot_test.png'.format(**d))

    print("Plotted to {problem}_plot_test.svg and {problem}_plot_test.png".format(**d))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2, help='gpu id')
    parser.add_argument('--eid', type=int, default=0, help='epoch id from which train/test optimizer')
    parser.add_argument('--num_units', type=int, default=20, help='number of units in LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='number of lstm layers')
    parser.add_argument('--name', type=str, default='lstm_opt', help='name of model')

    subparsers = parser.add_subparsers(help='mode: train or test')

    parser_train = subparsers.add_parser('train', help='train optimizer on a set of functions')
    parser_train.add_argument('--n_steps', type=int, default=100, help='number of steps')
    parser_train.add_argument('--n_bptt_steps', type=int, default=20, help='number of bptt steps')
    parser_train.add_argument('--n_batches', type=int, default=100, help='number of batches per epoch')
    parser_train.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    parser_train.add_argument('--train_lr', type=float, default=1e-2, help='learning rate')
    parser_train.add_argument('--loss_type', type=str, choices=['log', 'sum', 'last'], default='log', help='loss function to use')

    parser_train.set_defaults(func=run_train)

    parser_test = subparsers.add_parser('test', help='run trained optimizer on some problem')
    parser_test.add_argument('problem', choices=['quadratic', 'rosenbrock'], help='problem to run test on')
    parser_test.add_argument('--n_steps', type=int, default=100, help='number of steps')
    parser_test.add_argument('--n_batches', type=int, default=100, help='number of batches per epoch')
    parser_test.add_argument('--filename', type=str, default='plot_test')
    parser_test.add_argument('--mode', type=str, default='many', choices=['many', 'cv'], help='which mode to run')

    parser_test.set_defaults(func=run_test)

    parser_plot = subparsers.add_parser('plot', help='plot dumped results')
    parser_plot.add_argument('filename', type=str, help='results to print')
    parser_plot.add_argument('--plot_lr', action='store_true')

    parser_plot.set_defaults(func=run_plot)

    flags = parser.parse_args()
    pprint.pprint(flags)
    
    #shutil.rmtree('./{}_data/'.format(flags.name), ignore_errors=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(flags.gpu)
    flags.func(flags)
