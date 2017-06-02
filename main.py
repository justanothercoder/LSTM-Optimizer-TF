import os
import argparse
import pprint

import numpy as np
import tensorflow as tf
import basic_model

from lstm_opt import LSTMOpt
from sgd_opt import SgdOpt
from momentum_opt import MomentumOpt

import quadratic_optimizee
import rosenbrock_optimizee

import shutil

def main(flags):
    pprint.pprint(flags)

    shutil.rmtree('./lstm_opt.data/', ignore_errors=True)

    graph = tf.Graph()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(flags.gpu)

    with graph.as_default():
        with tf.Session(graph=graph) as session:
            #optimizee = quadratic_optimizee.Quadratic(low=50, high=100)
            optimizee = rosenbrock_optimizee.Rosenbrock(low=2, high=10)

            #opt = basic_model.BasicModel(optimizee, train_lr=flags.train_lr, n_bptt_steps=flags.n_bptt_steps, loss_type=flags.loss_type)
            opt = LSTMOpt(optimizee, train_lr=flags.train_lr, 
                                   n_bptt_steps=flags.n_bptt_steps, loss_type=flags.loss_type, 
                                   num_units=flags.num_units, num_layers=flags.num_layers, name='lstm_opt')

            #s_opts = [SgdOpt(optimizee, lr=16 * 2**(-i), name='sgd_opt_lr_{}'.format(4-i)) for i in range(0, 6)]
            s_opts = [MomentumOpt(optimizee, lr=16 * 2**(-i), name='momentum_opt_lr_{}'.format(4-i)) for i in range(0, 7)]

            optimizee.build()
            opt.build()

            for s_opt in s_opts:
                s_opt.build()

            session.run(tf.global_variables_initializer())

            if flags.mode == 'train':
                opt.train(n_epochs=flags.n_epochs, n_batches=flags.n_batches, n_steps=flags.n_steps, eid=flags.eid)
            else:
                if flags.eid == 0:
                    print("eid must be > 0 if mode is testing")
                    return

                st = np.random.get_state()

                opt.test(eid=flags.eid, n_batches=flags.n_batches, n_steps=flags.n_steps)

                for s_opt in s_opts:
                    np.random.set_state(st)
                    s_opt.test(eid=flags.eid, n_batches=flags.n_batches, n_steps=flags.n_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2, help='gpu id')
    parser.add_argument('--n_steps', type=int, default=100, help='number of steps')
    parser.add_argument('--n_bptt_steps', type=int, default=20, help='number of bptt steps')
    parser.add_argument('--n_batches', type=int, default=100, help='number of batches per epoch')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--train_lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--loss_type', type=str, choices=['log', 'sum', 'last'], default='log', help='loss function to use')
    parser.add_argument('--num_units', type=int, default=20, help='number of units in LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='number of lstm layers')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='mode: train or test')
    parser.add_argument('--eid', type=int, default=0, help='epoch id from which train/test optimizer')

    flags = parser.parse_args()
    main(flags)
