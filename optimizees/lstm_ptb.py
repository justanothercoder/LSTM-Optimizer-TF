import numpy as np
import tensorflow as tf
from .ptb import reader
from . import optimizee

HOME = '/srv/hd1/data/vyanush/lstm_opt_tf/optimizees/'

class LSTM_PTB(optimizee.Optimizee):
    name = 'lstm_ptb'

    def __init__(self, num_layers=2, num_steps=20, hidden_size=200, batch_size=20, vocab_size=10000):
        self.init_scale = 0.1
        self.learning_rate = 1.0
        self.max_grad_norm = 5
        self.max_epoch = 4
        self.max_max_epoch = 13
        self.keep_prob = 1.0
        self.lr_decay = 0.5
        
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size

        self.x_len = None
        self.x_len_counted = False


    def get_x_dim(self):
        return self.dim


    def build(self):
        with tf.variable_scope('lstm_ptb'):
            self.dim = tf.placeholder(tf.int32, [], name='dim')
            self.x = tf.placeholder(tf.int32, [None, None, None, None], name='X') # n_bptt_steps * batch_size * n_steps * data_size
            self.y = tf.placeholder(tf.int32, [None, None, None, None], name='y') # n_bptt_steps * batch_size * n_steps * data_size
        
        data = reader.ptb_raw_data(data_path=HOME + 'ptb/simple-examples/data/')[0]
        self.input_data, self.targets = reader.ptb_producer(data, self.batch_size, self.num_steps)


    def custom_getter(self, getter, name, shape=None, *args, **kwargs):
        print('getter', name, shape)
        if shape is not None:
            dim = np.prod(shape)
            var = tf.reshape(self.w[self.s: self.s + dim], shape)
            self.s += dim
            return tf.identity(var, name=name)
        else:
            return getter(name, shape, *args, **kwargs)

 
    def loss(self, x, i):
        self.w = x
        self.s = 0
            
        def attn_cell():
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True)
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)

        with tf.variable_scope('lstm_ptb/loss', custom_getter=self.custom_getter):
            cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(self.num_layers)], state_is_tuple=True)
            _initial_state = cell.zero_state(self.batch_size, tf.float32)
        
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [self.vocab_size, self.hidden_size], dtype=tf.float32)
                inputs = tf.nn.embedding_lookup(embedding, self.x[i, 0])

            #inputs = tf.nn.dropout(inputs, self.keep_prob)
            #inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
        
            #outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=_initial_state)
            print(inputs.get_shape())
            outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=_initial_state)

            #output = tf.reshape(tf.stack(outputs, axis=1), [-1, self.hidden_size])
            output = tf.reshape(outputs, [-1, self.hidden_size])
            softmax_w = tf.get_variable("softmax_w", [self.hidden_size, self.vocab_size], dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b", [self.vocab_size], dtype=tf.float32)
            logits = tf.matmul(output, softmax_w) + softmax_b
            logits = tf.reshape(logits, [self.batch_size, self.num_steps, self.vocab_size])

            loss = tf.contrib.seq2seq.sequence_loss(
                logits,
                self.y[i, 0],
                tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
                average_across_timesteps=False,
                average_across_batch=True
            )
        
        if not self.x_len_counted:
            self.x_len = self.s
            self.x_len_counted = True

        g = self.grad(x, loss)
        return loss, g


    def get_initial_x(self, batch_size=1):
        w = np.random.normal(0, 0.01, size=(batch_size, self.x_len))
        return w
        

    def get_new_params(self, batch_size=1):
        return {
            self.dim: self.x_len
        }


    def get_next_dict(self, n_bptt_steps, batch_size=1):
        session = tf.get_default_session()

        x = np.zeros((n_bptt_steps, 1, self.batch_size, self.num_steps)) 
        y = np.zeros((n_bptt_steps, 1, self.batch_size, self.num_steps))
        
        for i in range(n_bptt_steps):
            x[i, 0] = session.run(self.input_data)
            y[i, 0] = session.run(self.targets)

        return { 
            self.x: x,
            self.y: y,
        } 

