#!/usr/bin/env python

import os
import argparse
import pprint
import shutil
import pickle
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import basic_model

from lstm_opt import LSTMOpt
from sgd_opt import SgdOpt
from momentum_opt import MomentumOpt

import quadratic_optimizee
import rosenbrock_optimizee

import plotting


optimizees = {
    'quadratic': quadratic_optimizee.Quadratic(low=50, high=100),
    'rosenbrock': rosenbrock_optimizee.Rosenbrock(low=2, high=10)
}


def run_train(flags):
    graph = tf.Graph()

    with graph.as_default():
        with tf.Session(graph=graph) as session:

            opt = LSTMOpt(optimizees, train_lr=flags.train_lr, 
                                   n_bptt_steps=flags.n_bptt_steps, loss_type=flags.loss_type, stop_grad=flags.stop_grad,
                                   num_units=flags.num_units, num_layers=flags.num_layers, name=flags.name)

            for optimizee in optimizees.values():
                optimizee.build()

            opt.build()

            session.run(tf.global_variables_initializer())
            train_rets, test_rets = opt.train(n_epochs=flags.n_epochs, n_batches=flags.n_batches, n_steps=flags.n_steps, eid=flags.eid)
            
            filename = '{}_train_results.pkl'.format(flags.name)
            with open(filename, 'wb') as res_file:
                d = {
                    'name': flags.name,
                    'mode': 'training',
                    'train_results': train_rets,
                    'test_results': test_rets,
                }
                pickle.dump(d, res_file, protocol=pickle.HIGHEST_PROTOCOL)


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

            filename = '{}_{}_{}_test_results.pkl'.format(flags.name, flags.problem, flags.mode)
            with open(filename, 'wb') as res_file:
                d = {
                    'mode': 'testing',
                    'problem': flags.problem,
                    'n_steps': flags.n_steps,
                    'results': results,
                    'n_functions': flags.n_batches,
                }
                pickle.dump(d, res_file, protocol=pickle.HIGHEST_PROTOCOL)

            
def run_plot(flags):
    with open(flags.filename, 'rb') as f:
        d = pickle.load(f)

    if d['mode'] == 'training':
        plotting.plot_training_results(flags, d)
    elif d['mode'] == 'testing':
        plotting.plot_test_results(flags, d)
    else:
        raise ValueError("Unknown mode")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', help='run model on CPU')
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
    parser_train.add_argument('--no-stop_grad', action='store_false', dest='stop_grad')

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

    ulags = parser.parse_args()
    pprint.pprint(flags)
    
    #shutil.rmtree('./{}_data/'.format(flags.name), ignore_errors=True)

    if flags.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(flags.gpu)

    flags.func(flags)
