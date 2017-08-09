import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pathlib 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn

import random
random.seed(42)

import numpy as np
np.random.seed(42)

import tensorflow as tf
tf.set_random_seed(42)

from optimizees import rosenbrock
from lstm_optimizer import LSTMOptimizer

import util


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def minimize(opt, session):
    #set_random_seed(42)

    r = rosenbrock.Rosenbrock(low=1, high=2)
    r.build()

    x = tf.Variable(r.get_initial_x()[0].astype(np.float32))
    f, _ = r.loss(x[None], 0)

    train_op = opt.minimize(f[0], var_list=[x])

    session.run(tf.global_variables_initializer())

    is_lstm = hasattr(opt, 'restore')
    if is_lstm:
        opt.restore()
        
    
    fs = []
    n_steps = 100

    feed_dict = r.get_new_params()
    for _ in range(n_steps):
        f_, _ = session.run([f, train_op], feed_dict=feed_dict)
        fs.append(f_)
        #print(f_)

        if is_lstm:
            lr = session.run(opt.state['loglr'], feed_dict=feed_dict)
            #print(lr)

    return fs


def minimize_usual(session):
    model_path = pathlib.Path('models/norm_grad_new')
    eid = 100
    
    r = rosenbrock.Rosenbrock(low=1, high=2)
    r.build()

    opt = util.load_opt(model_path)
    opt.build({'r': r}, inference_only=True, n_bptt_steps=1)

    session.run(tf.global_variables_initializer())
    results = opt.test(eid, 1, n_steps=100)[0]

    #print(results['lrs'][:10])

    return results['values']


def main():
    model_path = pathlib.Path('models/norm_grad_new')
    eid = 100

    lstm_opt = LSTMOptimizer(model_path, eid)
    adam_opt = tf.train.AdamOptimizer(1e-3)
        
    state = np.random.get_state()

    with tf.Graph().as_default(), tf.Session() as session:
        #np.random.set_state(state)
        #with tf.variable_scope('adam_opt'):
        #    fs_2 = minimize(adam_opt, session)
        pass

    with tf.Graph().as_default():
        tf.set_random_seed(42)

        np.random.set_state(state)
        with tf.Session() as session:
            #tf.set_random_seed(42)
            fs_3 = minimize_usual(session)

        np.random.set_state(state)
        with tf.Session() as session:
            #tf.set_random_seed(42)
            with tf.variable_scope('kek'):
                fs_1 = minimize(lstm_opt, session)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 12))
    ax.set_title('value')
    ax.semilogy(fs_1, label='lstm')
    #ax.semilogy(fs_2, label='adam_lr=1e-3')
    ax.semilogy(fs_3, label='lstm_nobug')
    ax.legend(loc='best')
    fig.savefig('testrun_2.png', format='png')


if __name__ == '__main__':
    main()
