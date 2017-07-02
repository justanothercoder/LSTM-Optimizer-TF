import copy
import tensorflow as tf

import quadratic_optimizee, rosenbrock_optimizee
from sgd_opt import SgdOpt
from momentum_opt import MomentumOpt

from util import lstm_opt


def run_train(flags):
    with open('models/{model_name}/train/config'.format(model_name=flags.name), 'w') as conf:
        d = copy.copy(vars(flags))
        del d['eid'], d['gpu'], d['cpu'], d['func']
        print(d)
        json.dump(d, conf, sort_keys=True, indent=4)

    graph = tf.Graph()

    with graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4

        session = tf.Session(config=config, graph=graph)
        with session.as_default():
                
            optimizees = {
                'quadratic': quadratic_optimizee.Quadratic(low=50, high=100),
                'rosenbrock': rosenbrock_optimizee.Rosenbrock(low=2, high=10)
            }

            if 'all' not in flags.optimizee:
                optimizees = {name: opt for name, opt in optimizees.items() if name in flags.optimizee}

            opt = lstm_opt(optimizees, flags)

            for optimizee in optimizees.values():
                optimizee.build()

            opt.build()

            session.run(tf.global_variables_initializer())
            train_rets, test_rets = opt.train(n_epochs=flags.n_epochs, n_batches=flags.n_batches, batch_size=flags.batch_size, n_steps=flags.n_steps, eid=flags.eid)
            
            util.dump_results(flags.name, (train_rets, test_rets), phase='train')

            for problem, rets in util.split_list(test_rets, lambda ret: ret['optimizee_name'])[0].items():
                util.dump_results(flags.name, rets, phase='test', problem=problem + '_training', mode='many')
