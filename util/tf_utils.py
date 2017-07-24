import functools
import tensorflow as tf


def get_devices(flags):
    if flags.gpu is not None:
        devices = ['/gpu:{}'.format(i) for i in range(len(flags.gpu))]
    else:
        devices = ['/cpu:0']

    return devices
    

def get_tf_config():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config


from tensorflow.python import debug as tf_debug

def with_tf_graph(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        graph = tf.Graph()
        with graph.as_default():
            config = get_tf_config()
            session = tf.Session(config=config, graph=graph)

            #session = tf_debug.LocalCLIDebugWrapperSession(session)
            #session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            with session.as_default():
                return func(*args, **kwargs)
    return wrapper
