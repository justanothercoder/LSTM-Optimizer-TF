import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pathlib 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn

import pickle
import random
import numpy as np
import tensorflow as tf
from lstm_optimizer import LSTMOptimizer

from tf_optimizees import mlp_classifier as nn
from tf_optimizees import conv_classifier as nn_conv

import util


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def minimize(opt, session):
    r = nn_conv.ConvClassifier(num_filters=100, num_layers=5, activation='relu', dataset_name='cifar-10')
    r.build(opt)

    session.run(tf.global_variables_initializer())

    is_lstm = hasattr(opt, 'restore')
    if is_lstm:
        opt.restore()

    results = r.train(n_epochs=30, batch_size=500)
    return results


def main():
    model_path = pathlib.Path('models/add_skip_true/use_both')
    eid = 80

    lstm_opt = LSTMOptimizer(model_path, eid, clip_delta=1)
    adam_opt = tf.train.AdamOptimizer(1e-3)

    results = {}

    with tf.Session() as session:
        #with tf.variable_scope('lstm_opt'):
        
        state = np.random.get_state()

        result = minimize(lstm_opt, session)
        results['lstm'] = result
    
        losses_1 = result['losses']
        accs_1 = result['accs']

        val_losses_1 = result['val_losses']

        np.random.set_state(state)
        with tf.variable_scope('adam_opt'):
            result = minimize(adam_opt, session)
            results['adam'] = result

            losses_2 = result['losses']
            accs_2 = result['accs']
        
            val_losses_2 = result['val_losses']

    #losses_1 = util.get_moving(losses_1)
    #losses_2 = util.get_moving(losses_2)
    #accs_1 = util.get_moving(accs_1)
    #accs_2 = util.get_moving(accs_2)

    print(val_losses_1[-1], val_losses_2[-1])

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))
    axes[0].set_title('loss')
    axes[0].semilogy(losses_1, label='lstm')
    axes[0].semilogy(losses_2, label='adam_lr=1e-3')
    axes[1].set_title('accuracy')
    axes[1].semilogy(accs_1, label='lstm')
    axes[1].semilogy(accs_2, label='adam_lr=1e-3')
    
    axes[0].legend(loc='best')
    axes[1].legend(loc='best')
    fig.savefig('testruns/conv_cifar10_classifier_10_3.png', format='png')

    with open('testruns/conv_cifar10_classifier_10_3.pickle', 'wb') as f:
        pickle.dump(results, f, format='png')


if __name__ == '__main__':
    main()
