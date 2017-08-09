import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pathlib 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn

import random
import numpy as np
import tensorflow as tf
from lstm_optimizer import LSTMOptimizer

from tf_optimizees import sin_lstm as nn

import util


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def minimize(opt, session, n_batches=1):
    r = nn.SinLSTM(num_units=20, num_inputs=10, n_lstm=1)
    r.build(opt)

    batch_size = 128
        
    session.run(tf.global_variables_initializer())
    all_vars = r.get_vars()

    is_lstm = hasattr(opt, 'restore')
    if is_lstm:
        opt.restore()
        all_vars = set(all_vars) - set(opt.lstm_vars)

    results = []
    for batch in range(n_batches):
        print(batch)
        session.run(tf.variables_initializer(all_vars))

        result = r.train(n_batches=100, batch_size=128)
        results.append(result)
    return results


def main():
    model_path = pathlib.Path('models/add_skip_true/use_both')
    eid = 80

    lstm_opt = LSTMOptimizer(model_path, eid, clip_delta=1)
    #adam_opt = tf.train.AdamOptimizer(1e-3)
    
    adagrad_opt = tf.train.AdagradOptimizer(0.5)

    with tf.Session() as session:
        #with tf.variable_scope('lstm_opt'):
        
        state = np.random.get_state()
    
        losses_1 = minimize(lstm_opt, session, n_batches=100)
        losses_1 = np.array(losses_1)
        print(losses_1.shape)
        print(np.mean(losses_1))

        np.random.set_state(state)
        #with tf.variable_scope('adam_opt'):
        #    losses_2 = minimize(adam_opt, session)

        with tf.variable_scope('adagrad_opt'):
            losses_2 = minimize(adagrad_opt, session, n_batches=100)
            losses_2 = np.array(losses_2)

        print(losses_2.shape)
        print(np.mean(losses_2))

    #print(np.mean(losses_1))
    #print(np.mean(losses_2))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 12))
    ax.set_title('loss')
    ax.semilogy(np.mean(losses_1, axis=0), label='lstm')
    ax.semilogy(np.mean(losses_2, axis=0), label='adagrad_lr=1e-3')
    ax.legend(loc='best')
    fig.savefig('testruns/sin-lstm.png', format='png')


if __name__ == '__main__':
    main()
