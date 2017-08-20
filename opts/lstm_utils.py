import tensorflow as tf
from tensorflow.python.util import nest
#from tensorflow.python.ops.rnn_cell_impl import _state_size_with_prefix


def get_initial_cell_state(cell, initializer, batch_size, dtype):
    state_size = cell.state_size
    if nest.is_sequence(state_size):
        state_size_flat = nest.flatten(state_size)
        init_state_flat = [
            initializer(_state_size_with_prefix(s), batch_size, dtype, i)
                for i, s in enumerate(state_size_flat)]
        init_state = nest.pack_sequence_as(structure=state_size,
                                    flat_sequence=init_state_flat)
    else:
        init_state_size = _state_size_with_prefix(state_size)
        init_state = initializer(init_state_size, batch_size, dtype, None)

    return init_state


def make_variable_state_initializer(**kwargs):
    def variable_state_initializer(shape, batch_size, dtype, index):
        args = kwargs.copy()

        if args.get('name'):
            args['name'] = args['name'] + '_' + str(index)
        else:
            args['name'] = 'init_state_' + str(index)

        args['shape'] = shape
        args['dtype'] = dtype

        var = tf.get_variable(**args)
        var = tf.expand_dims(var, 0)
        var = tf.tile(var, tf.stack([batch_size] + [1] * len(shape)))
        #var.set_shape(_state_size_with_prefix(shape, prefix=[None]))
        var.set_shape([None] + shape.as_list())
        return var

    return variable_state_initializer
