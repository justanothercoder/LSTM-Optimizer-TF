import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from tf_optimizees import mlp_classifier as nn
import util


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def minimize(opt, session):

    r = nn.MLPClassifier(num_units=100, num_layers=6)
    r.prepare_data(dataset_name='mnist')

    ops = r.build(opt)

    session.run(tf.global_variables_initializer())

    is_lstm = hasattr(opt, 'restore')
    if is_lstm:
        opt.restore()

    n_epochs = 30
    batch_size = 600

    accs = []
    losses = []

    print("Data size: ", r.X.shape[0])

    t = time.time()

    for i, (x, y) in enumerate(r.batch_iterator(n_epochs, batch_size)):
        feed_dict = {ops['x']: x, ops['y']: y}
        loss, acc = ops['loss'], ops['acc']
        train_op = ops['train_op']

        loss_, acc_, _ = session.run([loss, acc, train_op], feed_dict=feed_dict)
        accs.append(acc_)
        losses.append(loss_)

        if (i + 1) % 100 == 0:
            print("Batch: {}/{}".format(i + 1, n_epochs * r.X.shape[0] // batch_size))
            print("Loss: ", loss_)
            print("Accuracy: ", acc_)
            print("Batch time: ", (time.time() - t) / 100)
            t = time.time()

    return losses, accs

def main():
    model_path = pathlib.Path('models/add_skip_true/use_both')
    eid = 80

    lstm_opt = LSTMOptimizer(model_path, eid)
    adam_opt = tf.train.AdamOptimizer(1e-3)

    with tf.Session() as session:
        #with tf.variable_scope('lstm_opt'):
        
        state = np.random.get_state()

        losses_1, accs_1 = minimize(lstm_opt, session)

        np.random.set_state(state)
        with tf.variable_scope('adam_opt'):
            losses_2, accs_2 = minimize(adam_opt, session)

    #losses_1 = util.get_moving(losses_1)
    #losses_2 = util.get_moving(losses_2)
    #accs_1 = util.get_moving(accs_1)
    #accs_2 = util.get_moving(accs_2)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))
    axes[0].set_title('loss')
    axes[0].semilogy(losses_1, label='lstm')
    axes[0].semilogy(losses_2, label='adam_lr=1e-3')
    axes[1].set_title('accuracy')
    axes[1].semilogy(accs_1, label='lstm')
    axes[1].semilogy(accs_2, label='adam_lr=1e-3')
    
    axes[0].legend(loc='best')
    axes[1].legend(loc='best')
    fig.savefig('testrun.png', format='png')


if __name__ == '__main__':
    main()
