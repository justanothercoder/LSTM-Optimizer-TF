import time
import numpy as np
import tensorflow as tf

import pickle
from sklearn.datasets import load_digits, fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn import utils

HOME = '/srv/hd1/data/vyanush/'

def get_dataset(dataset_name):
    def standartize(X):
        return StandardScaler().fit_transform(X.astype(np.float32))

    if dataset_name == 'digits':
        dataset = load_digits(n_class=10)
        X, Y = dataset.data, dataset.target
        X = standartize(X)
        return X.reshape(-1, 8, 8, 1), np.array(Y)

    elif dataset_name == 'mnist':
        dataset = fetch_mldata('MNIST original', data_home=HOME)
        X, Y = dataset.data, dataset.target
        X = standartize(X)
        return X.reshape(-1, 28, 28, 1), np.array(Y)

    elif dataset_name == 'cifar-10':
        X = np.zeros((60000, 3072))
        Y = np.zeros((60000,))

        for i in range(5):
            with open(HOME + 'lstm_opt_tf/optimizees/' + 'cifar10/data_batch_{}'.format(i+1), 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                print(i, data[b'data'].shape)
                X[i*10000:(i+1)*10000] = data[b'data']
                Y[i*10000:(i+1)*10000] = data[b'labels']

        with open(HOME + 'lstm_opt_tf/optimizees/' + 'cifar10/test_batch', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            X[i*10000:(i+1)*10000] = data[b'data']
            Y[i*10000:(i+1)*10000] = data[b'labels']

        X = standartize(X)
        return X.reshape(-1, 32, 32, 3), np.array(Y)


class ConvClassifier:
    def __init__(self, num_filters=100, num_layers=1, activation='relu', dataset_name='mnist'):
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.activation = getattr(tf.nn, activation)
        self.dataset_name = dataset_name

        self.prepare_data(dataset_name)


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
        x = tf.placeholder(tf.float32, shape=[None, self.X.shape[1], self.X.shape[2], self.X.shape[3]])
        y = tf.placeholder(tf.int32, shape=[None])
        return x, y


    def inference(self, x):
        pred = x

        with tf.variable_scope('inference') as self.scope:
            for _ in range(self.num_layers):
                pred = tf.layers.conv2d(pred, self.num_filters, (3, 3), activation=None, padding='same')
                pred = tf.layers.batch_normalization(pred)
                pred = self.activation(pred)
                pred = tf.layers.conv2d(pred, self.num_filters, (3, 3), activation=None, padding='same')
                pred = tf.layers.batch_normalization(pred)
                pred = self.activation(pred)

                pred = tf.layers.max_pooling2d(pred, (2, 2), (2, 2))

            pred = tf.reduce_mean(pred, axis=(1,2))
            
            pred = tf.layers.dense(pred, 100, activation=None)
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

            grads_and_vars = optimizer.compute_gradients(loss, var_list=all_vars)
            grads, _ = tf.clip_by_global_norm([g for g, v in grads_and_vars], 1.)
            train_op = optimizer.apply_gradients(list(zip(grads, all_vars)))

        with tf.control_dependencies([train_op]):
            train_op = tf.group(self.average_op)

        return train_op


    def prepare_data(self, dataset_name):
        self.X, self.Y = get_dataset(dataset_name)
        self.X, self.Y = utils.shuffle(self.X, self.Y)

        if dataset_name in {'mnist', 'cifar-10'}:
            self.X_train = self.X[:50000]
            self.Y_train = self.Y[:50000]

            self.X_val = self.X[50000:]
            self.Y_val = self.Y[50000:]


    def batch_iterator(self, n_epochs, batch_size, is_train=True):
        if is_train:
            X, Y = self.X_train, self.Y_train
        else:
            X, Y = self.X_val, self.Y_val

        for epoch in range(n_epochs):
            indices = np.arange(X.shape[0])

            if is_train:
                np.random.shuffle(indices)

            for pos in range(0, X.shape[0] - batch_size + 1, batch_size):
                ind = indices[pos: pos + batch_size]
                yield X[ind], Y[ind]


    def val(self, batch_size):
        losses = []
        accs = []

        for i, (x, y) in enumerate(self.batch_iterator(1, batch_size, is_train=False)):
            feed_dict = {self.ops['x']: x, self.ops['y']: y}
            loss, acc = self.ops['loss'], self.ops['acc']

            session = tf.get_default_session()
            loss_, acc_ = session.run([loss, acc], feed_dict=feed_dict)
            accs.append(acc_)
            losses.append(loss_)

        loss_ = np.mean(losses)
        acc_ = np.mean(accs)

        return loss_, acc_


    def train(self, n_epochs, batch_size):
        session = tf.get_default_session()

        print("Data size: ", self.X_train.shape[0])

        results = {
            'accs': [],
            'losses': [],
            'val_losses': [],
            'val_accs': [],
        }

        batches_per_epoch = self.X_train.shape[0] // batch_size

        t = time.time()
        for i, (x, y) in enumerate(self.batch_iterator(n_epochs, batch_size)):
            feed_dict = {self.ops['x']: x, self.ops['y']: y}
            loss, acc = self.ops['loss'], self.ops['acc']
            train_op = self.ops['train_op']

            loss_, acc_, _ = session.run([loss, acc, train_op], feed_dict=feed_dict)
            results['accs'].append(acc_)
            results['losses'].append(loss_)

            if (i + 1) % 100 == 0:
                print("Batch: {}/{}".format(i + 1, n_epochs * batches_per_epoch))
                print("Loss: ", loss_)
                print("Accuracy: ", acc_)
                print("Batch time: ", (time.time() - t) / 100)
                t = time.time()

            epoch = (i + 1) // batches_per_epoch
            if (epoch + 1) % 5 == 0:
                loss_, acc_ = self.val(batch_size)

                results['val_losses'].append(loss_)
                results['val_accs'].append(acc_)

                print("Val loss: ", loss_)
                print("Val accuracy: ", acc_)

        return results
