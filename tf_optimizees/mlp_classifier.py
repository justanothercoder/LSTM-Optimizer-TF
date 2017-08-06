import numpy as np
import tensorflow as tf

from sklearn.datasets import load_digits, fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn import utils

class MLPClassifier:
    def __init__(self, num_units=20, num_layers=1, activation='sigmoid'):
        self.num_units = num_units
        self.num_layers = num_layers
        self.activation = getattr(tf.nn, activation)


    def build(self, optimizer):
        x, y = self.inputs()
        pred = self.inference(x)
        loss, acc = self.loss(pred, y)
        train_op = self.train_op(loss, optimizer)

        self.ops = {
            'x': x, 'y': y,
            'pred': pred,
            'loss': self.ema.average(loss),
            'acc': self.ema.average(acc),
            'train_op': train_op
        }

        return self.ops


    def inputs(self):
        x = tf.placeholder(tf.float32, shape=[None, self.X.shape[1]])
        y = tf.placeholder(tf.int32, shape=[None])
        return x, y


    def inference(self, x):
        pred = x

        with tf.variable_scope('inference') as self.scope:
            for _ in range(self.num_layers):
                pred = tf.layers.dense(pred, self.num_units, activation=None)
                pred = tf.layers.batch_normalization(pred)
                pred = self.activation(pred)
            
            pred = tf.layers.dense(pred, 10)

        return pred


    def loss(self, logits, y):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits), axis=-1)
        p = tf.cast(tf.argmax(tf.nn.softmax(logits), axis=1), tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(p, y), tf.float32))

        self.ema = tf.train.ExponentialMovingAverage(decay=0.95)
        self.average_op = self.ema.apply([loss, acc])

        return loss, acc


    def train_op(self, loss, optimizer='adam'):
        if optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(1e-3)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope.name)
            train_op = optimizer.minimize(loss, var_list=all_vars)

        with tf.control_dependencies([train_op]):
            train_op = tf.group(self.average_op)

        return train_op


    def prepare_data(self, dataset_name):
        self.dataset_name = dataset_name

        if dataset_name == 'digits':
            dataset = load_digits(n_class=10)
        elif dataset_name == 'mnist':
            dataset = fetch_mldata('MNIST original', data_home='/srv/hd1/data/vyanush/')

        self.X, self.Y = dataset.data, dataset.target
        self.X, self.Y = utils.shuffle(self.X, self.Y)

        if dataset_name == 'mnist':
            self.X = self.X[:50000]
            self.Y = self.Y[:50000]

        self.X = StandardScaler().fit_transform(self.X.astype(np.float32))


    def batch_iterator(self, n_epochs, batch_size):
        for epoch in range(n_epochs):
            indices = np.arange(self.X.shape[0])
            np.random.shuffle(indices)

            for pos in range(0, self.X.shape[0] - batch_size + 1, batch_size):
                ind = indices[pos: pos + batch_size]
                yield self.X[ind], self.Y[ind]
