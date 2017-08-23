import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell

class OptCell(RNNCell):
    def __init__(self):
        super(OptCell, self).__init__()


    @property
    def state_size(self):
        raise NotImplementedError


    @property
    def output_size(self):
        return 2


    def __call__(self, inputs, state):
        raise NotImplementedError


class OptFuncCell(OptCell):
    def __init__(self, cell, input_x, loss_fn, stop_grad=True):
        super(OptFuncCell, self).__init__()
        self.input_x = input_x
        self.loss_fn = loss_fn
        self.cell = cell
        self.stop_grad = stop_grad


    def zero_state(self, batch_size):
        return self.input_x, self.cell.zero_state(batch_size)


    @property
    def state_size(self):
        return self.input_x.get_shape(), self.cell.state_size


    def __call__(self, _, state):
        x, cell_state = state

        value, g = self.loss_fn(x)
        if self.stop_grad:
            g = tf.stop_gradient(g)

        g2_norm = tf.reduce_sum(tf.square(g), axis=-1)
        
        step, new_state = self.cell(g, cell_state)
        step = tf.reshape(step, tf.shape(x))
        new_state = (x + step, new_state)

        def reshape_out(val):
            val = tf.expand_dims(val, -1)
            if len(val.get_shape().as_list()) == 1:
                val = tf.expand_dims(val, 0)
            return val

        output = tf.concat([reshape_out(value), reshape_out(g2_norm)], axis=-1)
        return output, new_state
