import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pathlib 

import random
import numpy as np
import tensorflow as tf
from lstm_optimizer import LSTMOptimizer
from optimizees import rosenbrock


if __name__ == '__main__':
    session = tf.Session()

    random.seed(42)
    np.random.seed(42)
    tf.set_random_seed(42)

    r = rosenbrock.Rosenbrock(low=1, high=2)
    r.build()

    x = tf.Variable(r.get_initial_x()[0].astype(np.float32))
    f, g = r.loss(x[None], 0)

    feed_dict = r.get_new_params()

    #x = tf.Variable(tf.ones([]))
    #f = x**2

    model_path = pathlib.Path('models/add_skip_true/use_both')
    eid = 80

    lstm_opt = LSTMOptimizer(model_path, eid)
    train_op = lstm_opt.minimize(f, var_list=[x])
    session.run(tf.global_variables_initializer())
        
    n_steps = 100
    for _ in range(n_steps):
        x_, f_, loglr = session.run([x, f, lstm_opt.state['loglr']], feed_dict=feed_dict)
        print('(x, f, lr): ({}, {}, {})'.format(x_, f_, np.exp(loglr)))
        session.run(train_op, feed_dict=feed_dict)
