#!/usr/bin/env python

import os
import argparse
import pprint
import shutil

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
                                   num_units=flags.num_units, num_layers=flags.num_layers, name='lstm_opt')

            for optimizee in optimizees:
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

            #s_opts = [SgdOpt(optimizee, lr=16 * 2**(-i), name='sgd_opt_lr_{}'.format(4-i)) for i in range(0, 6)]
            
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

            fig, ax = plt.subplots()
            ax_t = ax.twinx()

            for o in [opt] + s_opts:
                np.random.set_state(st)
                rets = o.test(eid=flags.eid, n_batches=flags.n_batches, n_steps=flags.n_steps)
                #rets = opt.test(eid=eid, n_batches=flags.n_batches, n_steps=flags.n_steps)

                fxs = np.array([ret['fxs'] / ret['fxs'][0] for ret in rets])
                lrs = np.array([ret['lrs'] / ret['lrs'][0] for ret in rets])
                norms = np.array([ret['norms'] / ret['norms'][0] for ret in rets])

                mean_trajectory = np.mean(fxs, axis=0)
                std = np.std(fxs, axis=0)

                mean_lr = np.mean(lrs, axis=0)
                std_lr = np.std(lrs, axis=0)

                mean_norm = np.mean(norms, axis=0)
                std_norm = np.std(norms, axis=0)

                ax.semilogy(mean_trajectory, label=o.name)
                #ax_t.plot(mean_lr, label=o.name, linestyle='--')
                ax_t.semilogy(mean_norm, label=o.name, linestyle='--')

                #plt.plot(mean_trajectory, label="{name}_{epoch}".format(name=opt.name, epoch=eid))
                #plt.fill_between(np.arange(rets.shape[1]), mean_trajectory + std, mean_trajectory - std, alpha=0.5)
            
            ax.set_title('{optimizee_name}: mean trajectory over {n_functions} functions for {n_steps} steps'.format(optimizee_name=flags.problem, n_functions=flags.n_batches, n_steps=flags.n_steps))
            ax.set_xlabel('iteration number')
            ax.set_ylabel(r'function value: $\frac{f(\theta_t)}{f(\theta_0)}$')
            ax.legend(loc='best')

            #ax_t.set_ylabel('mean learning rate')
            ax_t.set_ylabel(r'mean $\frac{\|\nabla f(\theta_t)\|^2}{\|\nabla f(\theta_0)\|^2}$')

            fig.savefig('{}_plot_test.svg'.format(flags.problem), format='svg')
            os.system('convert {name}_plot_test.svg {name}_plot_test.png'.format(name=flags.problem))


if __name__ == '__main__':
    #shutil.rmtree('./lstm_opt_data/', ignore_errors=True)

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

    parser_test.set_defaults(func=run_test)

    flags = parser.parse_args()
    pprint.pprint(flags)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(flags.gpu)
    flags.func(flags)
