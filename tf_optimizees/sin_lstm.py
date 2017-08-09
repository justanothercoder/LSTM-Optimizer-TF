import time 
import numpy as np
import tensorflow as tf

class SinLSTM:
    def __init__(self, num_units=20, num_inputs=10, n_lstm=1, noise_scale=0.1, initial_param_scale=0.1):
        self.num_units = num_units
        self.num_inputs = num_inputs
        self.n_lstm = n_lstm
        self.noise_scale = noise_scale
        self.initial_param_scale = initial_param_scale


    def build(self, optimizer):
        x, y = self.inputs()
        pred = self.inference(x)
        loss = self.loss(pred, y)
        train_op = self.train_op(loss, optimizer)

        self.ops = {
            'x': x, 'y': y,
            'pred': pred,
            'loss': loss,
            'train_op': train_op
        }

        return self.ops


    def inputs(self):
        x = tf.placeholder(tf.float32, shape=[None, None, 1])
        y = tf.placeholder(tf.float32, shape=[None, 1])
        return x, y


    def inference(self, x):
        with tf.variable_scope('inference', initializer=tf.random_normal_initializer(stddev=self.initial_param_scale)) as self.scope:
            def lstm_cell():
                return tf.contrib.rnn.BasicLSTMCell(self.num_units)

            cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self.n_lstm)])
            x = tf.unstack(x, num=self.num_inputs, axis=1)
            outputs, state = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
            #outputs, state = tf.nn.dynamic_rnn(cell, x)
            
            pred = tf.layers.dense(outputs[-1], 1)

        return pred


    def get_vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)


    def loss(self, pred, y):
        loss = tf.reduce_mean(tf.square(pred - y))
        return loss


    def train_op(self, loss, optimizer='adam'):
        if optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(1e-3)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name)

            grads_and_vars = optimizer.compute_gradients(loss, var_list=all_vars)
            grads, _ = tf.clip_by_global_norm([g for g, v in grads_and_vars], 1.)
            train_op = optimizer.apply_gradients(list(zip(grads, all_vars)))

        return train_op


    def batch_iterator(self, n_batches, batch_size):
        for batch in range(n_batches):
            X = np.zeros((batch_size, self.num_inputs))
            y = np.zeros((batch_size,))

            phi = np.random.uniform(0.0, 2 * np.pi, size=batch_size)
            omega = np.random.uniform(0.0, np.pi / 2, size=batch_size)
            A = np.random.uniform(0.0, 10.0, size=batch_size)

            for k in range(self.num_inputs):
                X[:, k] = A * np.sin(k * omega + phi) + np.random.normal(scale=self.noise_scale)
            y = A * np.sin(self.num_inputs * omega + phi)

            yield X[..., None], y[..., None]


    def train(self, n_batches, batch_size):
        losses = []
        session = tf.get_default_session()

        t = time.time()
        for i, (x, y) in enumerate(self.batch_iterator(n_batches, batch_size)):
            feed_dict = {self.ops['x']: x, self.ops['y']: y}
            loss = self.ops['loss']
            train_op = self.ops['train_op']

            loss_, _ = session.run([loss, train_op], feed_dict=feed_dict)
            losses.append(loss_)

            #if (i + 1) % 100 == 0:
            #    print("Batch: ", i + 1)
            #    print("Loss: ", loss_)
            #    print("Batch time: ", (time.time() - t) / 100)
            #    t = time.time()

        return losses

