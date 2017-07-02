import shutil
from collections import OrderedDict
import tensorflow as tf

import quadratic_optimizee, rosenbrock_optimizee
from sgd_opt import SgdOpt
from momentum_opt import MomentumOpt

from util import lstm_opt


def run_test(flags):
    graph = tf.Graph()
            
    optimizees = {
        'quadratic': quadratic_optimizee.Quadratic(low=50, high=100),
        'rosenbrock': rosenbrock_optimizee.Rosenbrock(low=2, high=10)
    }

    optimizee = {flags.problem: optimizees[flags.problem]}

    with graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4

        with tf.Session(config=config, graph=graph) as session:
            tests = {
                'rosenbrock': {
                    'sgd': [SgdOpt(optimizee, lr=2**(-i-5), name='sgd_lr_{}'.format(-i-9)) for i in range(1, 6)],
                    'momentum': [MomentumOpt(optimizee, lr=2**(-i-9), name='momentum_lr_{}'.format(-i-9)) for i in range(1, 3)],
                },
                'quadratic': {
                    'sgd': [SgdOpt(optimizee, lr=16 * 2**(-i), name='sgd_lr_{}'.format(4-i)) for i in range(0, 6)],
                    'momentum': [MomentumOpt(optimizee, lr=16 * 2**(-i), name='momentum_lr_{}'.format(4-i)) for i in range(0, 6)],
                }
            }


            #opt = LSTMOpt(optimizee, num_units=flags.num_units, num_layers=flags.num_layers, name=flags.name)
            opt = lstm_opt(optimizee, flags)

            s_opts = tests[flags.problem][flags.compare_with]
            
            optimizees[flags.problem].build()
            opt.build()

            for s_opt in s_opts:
                s_opt.build()

            session.run(tf.global_variables_initializer())

            if flags.eid == 0:
                raise ValueError("eid must be > 0 if mode is testing")

            st = np.random.get_state()

            results = OrderedDict()

            if flags.mode == 'many':
                for o in [opt] + s_opts:
                    np.random.set_state(st)
                    rets = o.test(eid=flags.eid, n_batches=flags.n_batches, n_steps=flags.n_steps)
                    results[o.name] = rets
            else:
                for eid in range(flags.start_eid, flags.eid + 1, flags.step):
                    np.random.set_state(st)
                    rets = opt.test(eid=eid, n_batches=flags.n_batches, n_steps=flags.n_steps)

                    name = '{name}_{eid}'.format(name=flags.name, eid=eid)
                    results[name] = rets

            util.dump_results(flags.name, results, phase='test', problem=flags.problem, mode=flags.mode)

            for o in s_opts:
                try:
                    shutil.rmtree('models/' + o.name)
                except:
                    pass
