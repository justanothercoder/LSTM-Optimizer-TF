import json
import tensorflow as tf

from sgd_opt import SgdOpt
from momentum_opt import MomentumOpt

import util
from util import lstm_opt, get_optimizees


def run_train(flags):
    #with open('models/{model_name}/train/config'.format(model_name=flags.name), 'w') as conf:
    
    conf_path = flags.model_path / 'train'/ 'config'
    with conf_path.open('w') as conf:
        d = vars(flags).copy()
        del d['eid'], d['gpu'], d['cpu'], d['func'], d['model_path']
        print(d)
        json.dump(d, conf, sort_keys=True, indent=4)

    graph = tf.Graph()

    with graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4

        with tf.Session(config=config, graph=graph) as session:
            optimizees = get_optimizees()
            if 'all' not in flags.optimizee:
                optimizees = {name: opt for name, opt in optimizees.items() if name in flags.optimizee}

            opt = lstm_opt(optimizees, flags)

            for optimizee in optimizees.values():
                optimizee.build()

            opt.build()

            session.run(tf.global_variables_initializer())
            train_rets, test_rets = opt.train(n_epochs=flags.n_epochs, n_batches=flags.n_batches, batch_size=flags.batch_size, n_steps=flags.n_steps, eid=flags.eid)
            
            util.dump_results(flags.model_path, (train_rets, test_rets), phase='train')

            for problem, rets in util.split_list(test_rets, lambda ret: ret['optimizee_name'])[0].items():
                util.dump_results(flags.model_path, rets, phase='test', problem=problem + '_training', mode='many')
