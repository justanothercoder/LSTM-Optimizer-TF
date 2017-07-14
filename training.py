import re
import json
import tensorflow as tf

from sgd_opt import SgdOpt
from momentum_opt import MomentumOpt

import util
    
training_options = {
    'batch_size', 'enable_random_scaling', 'loss_type', 
    'n_batches', 'n_bptt_steps', 'n_epochs', 'n_steps', 
    'optimizee', 'train_lr', 'momentum', 'optimizer', 'lambd'
}

def save_train_config(flags):

    d = {k: v for k, v in vars(flags).items() if k in training_options}
    print('Training config: ', d)

    conf_path = util.get_model_path(flags.name) / 'train'/ 'config'
    with conf_path.open('w') as conf:
        json.dump(d, conf, sort_keys=True, indent=4)


def select_optimizees(flags):
    optimizees = util.get_optimizees(clip_by_value=True, random_scale=flags.enable_random_scaling, noisy_grad=flags.noisy_grad)
    if 'all' not in flags.optimizee:
        optimizees = {name: opt for name, opt in optimizees.items() if name in flags.optimizee}

    return optimizees


def train_opt(opt, flags):
    train_options = {
        'n_epochs': flags.n_epochs,
        'n_batches': flags.n_batches,
        'batch_size': flags.batch_size,
        'n_steps': flags.n_steps,
        'eid': flags.eid,
        'train_lr': flags.train_lr,
        'momentum': flags.momentum,
        'verbose': flags.verbose,
    }
    return opt.train(**train_options)


def run_train(flags):
    save_train_config(flags)

    model_path = util.get_model_path(flags.name)

    r = re.compile('epoch-(?P<eid>\d+).index')
    eids = [r.match(p.split('/')[-1]).group('eid') for p in map(lambda s: str(s).split('/')[-1], (model_path / 'tf_data').iterdir()) if r.match(p)]
    if eids:
        max_eid = max(map(int, eids))
    else:
        max_eid = None

    if not flags.force and max_eid is not None and flags.eid < max_eid:
        print("You will overwrite existing checkpoints. Add -f to force it.")
        return
            
    opt = util.load_opt(flags.name)
    optimizees = select_optimizees(flags)

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(config=util.get_tf_config(), graph=graph) as session:
            for optimizee in optimizees.values():
                optimizee.build()

            if flags.gpu is not None:
                devices = ['/gpu:{}'.format(d) for d in map(int, flags.gpu)]
            else:
                devices = ['/cpu:0']
            opt.build(optimizees, n_bptt_steps=flags.n_bptt_steps, loss_type=flags.loss_type, optimizer=flags.optimizer, lambd=flags.lambd, devices=devices)

            session.run(tf.global_variables_initializer(), {opt.train_lr: flags.train_lr, opt.momentum: flags.momentum})
            train_rets, test_rets = train_opt(opt, flags)
            
            model_path = util.get_model_path(flags.name)
            util.dump_results(model_path, (train_rets, test_rets), phase='train', tag=flags.tag)

            #for problem, rets in util.split_list(test_rets, lambda ret: ret['optimizee_name'])[0].items():
            #    util.dump_results(util.get_model_path(flags.name), rets, phase='test', problem=problem + '_training', mode='many')
