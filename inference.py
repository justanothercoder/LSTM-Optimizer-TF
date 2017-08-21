import tensorflow as tf
from opts.cells import OptFuncCell


def static_inference(model, optimizee):
    state = model.input_state
    scope = model.inference_scope

    values = []
    norms = []

    for i in range(model.config.n_bptt_steps):
        info = model.step_with_func(optimizee.loss, i, state, model.config.stop_grad)
        state = info['state']

        values.append(info['value'])
        norms.append(info['gradient_norm'])

        if not scope.reuse:
            scope.reuse_variables()

    return dict(values=values, norms=norms, final_state=state)


def dynamic_inference(model, optimizee):
    def cond(sid, *loop_vars):
        return tf.less(sid, model.config.n_bptt_steps)


    def body(sid, vals, norms, *state):
        info = model.step_with_func(optimizee.loss, i, state, model.config.stop_grad)

        new_vals = tf.concat([vals, tf.expand_dims(info['value'], 0)], axis=0)
        new_norms = tf.concat([norms, tf.expand_dims(info['gradient_norm'], 0)], axis=0)

        out_state = (sid + 1, new_vals, new_norms) + info['state']
        return out_state

    vals_init = tf.zeros([0, tf.shape(model.input_state.x)[0]])
    norms_init = tf.zeros([0, tf.shape(model.input_state.x)[0]])

    i = tf.constant(0)
    in_state = (i, vals_init, norms_init) + model.input_state

    def get_shapes(t):
        shapes = []

        for i in t:
            if isinstance(i, tf.Tensor):
                shapes.append(i.get_shape())
            else:
                s = get_shapes(i)
                if type(i) in (tuple, list):
                    s = type(i)(s)
                else:
                    s = type(i)(c=s[0], h=s[1])
                shapes.append(s)
        
        return tuple(shapes)

    shape_invariants = (
                i.get_shape(),
                tf.TensorShape([None] + vals_init.get_shape().as_list()[1:]),
                tf.TensorShape([None] + norms_init.get_shape().as_list()[1:]),
            ) + get_shapes(state_init)

    _, vals, norms, *r = tf.while_loop(cond, body, in_state, shape_invariants=shape_invariants)

    vals  = tf.unstack(vals , num=model.config.n_bptt_steps, axis=0)
    norms = tf.unstack(norms, num=model.config.n_bptt_steps, axis=0)

    return dict(values=vals, norms=norms, final_state=r)
    

def cell_inference(model, optimizee):
    def decorator(f):
        c = 0
        def wrapper(x):
            nonlocal c
            ret = f(x, c)
            c += 1
            return ret
        return wrapper

    cell = OptFuncCell(model.cell, model.input_state.x, decorator(optimizee.loss), stop_grad=model.config.stop_grad)
    n_coords = tf.size(model.input_state.x)
    inputs = tf.zeros([n_coords, model.config.n_bptt_steps, 1])

    istate = cell.zero_state(n_coords)
    outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=istate, scope=model.inference_scope)
    values, norms = tf.unstack(outputs, num=2, axis=-1)

    values = tf.unstack(values, num=model.config.n_bptt_steps, axis=1)
    norms  = tf.unstack(norms , num=model.config.n_bptt_steps, axis=1)
    
    return dict(values=values, norms=norms, final_state=state, cell=cell, istate=istate)
