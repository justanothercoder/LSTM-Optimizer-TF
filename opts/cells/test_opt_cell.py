import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TF_CPP_LOG_MIN_LEVEL"] = '2'

import numpy as np
import tensorflow as tf

import lstm_opt_cell
import opt_cell

def main():
    x = tf.placeholder(tf.float32, [None, None])

    def loss_fn(x):
        f = tf.reduce_mean(x**2, axis=-1)
        g = tf.gradients(f, x)[0]
        return f, g

    n_steps = 23
    n_coords = tf.size(x)
    inputs = tf.zeros([n_coords, n_steps, 1])

    with tf.variable_scope('opt_scope/inference_scope') as scope:
        cell = lstm_opt_cell.LSTMOptCell(num_units=60, num_layers=3, residual=True, add_skip=True, use_both=True)
        cell(x, cell.zero_state(1))

        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
        saver = tf.train.Saver(var_list=all_vars)

        scope.reuse_variables()

    cell = opt_cell.OptFuncCell(cell, x, loss_fn)
    losses, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=cell.zero_state(n_coords), scope=scope)

    # losses.shape = [batch_size, n_steps, 1]
    loss = tf.reduce_mean(tf.log(losses + 1e-8) - tf.log(losses[:, :1] + 1e-8))
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=all_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
        
    #saver.restore(sess, '../models/use_both/snapshots/epoch-80')

    ema_loss = None

    for i in range(10000):
        dim = np.random.randint(low=1, high=11)
        x_val = np.random.uniform(size=(100, dim), low=-1., high=1.)
        
        _, loss_val = sess.run([train_op, loss], feed_dict={x: x_val})

        if ema_loss is None:
            ema_loss = loss_val
        else:
            ema_loss = 0.9 * ema_loss + 0.1 * loss_val

        print(ema_loss)

main()
