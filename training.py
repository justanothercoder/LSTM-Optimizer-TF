import re
import json
import subprocess, shlex

import tensorflow as tf

from opts.sgd_opt import SgdOpt
from opts.momentum_opt import MomentumOpt

import util
    
def save_train_config(flags):
    training_options = {
        'batch_size', 'enable_random_scaling', 'loss_type', 
        'n_batches', 'n_bptt_steps', 'n_epochs', 'n_steps', 
        'optimizee', 'train_lr', 'momentum', 'optimizer', 'lambd'
    }

    d = {k: v for k, v in vars(flags).items() if k in training_options}
    print('Training config: ', d)

    conf_path = util.get_model_path(flags.name) / 'train'/ 'config'
    with conf_path.open('w') as conf:
        json.dump(d, conf, sort_keys=True, indent=4)


def select_optimizees(flags):
    optimizees = util.get_optimizees(clip_by_value=False, random_scale=flags.enable_random_scaling, noisy_grad=flags.noisy_grad)
    if 'all' not in flags.optimizee:
        optimizees = {name: opt for name, opt in optimizees.items() if name in flags.optimizee}

    return optimizees


def build_opt(opt, optimizees, flags):
    if flags.gpu is not None:
        devices = ['/gpu:{}'.format(i) for i in range(len(flags.gpu))]
    else:
        devices = ['/cpu:0']

    build_options = {
        'n_bptt_steps': flags.n_bptt_steps,
        'loss_type': flags.loss_type,
        'optimizer': flags.optimizer,
        'lambd': flags.lambd,
        'devices': devices
    }
    opt.build(optimizees, **build_options)


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


def check_snapshots(flags):
    model_path = util.get_model_path(flags.name)

    save_path = 'snapshots'
    if flags.tag is not None:
        save_path += '_' + flags.tag

    subprocess.call(shlex.split('mkdir -p {}'.format(model_path / save_path)))

    r = re.compile('epoch-(?P<eid>\d+).index')

    files = [str(s).split('/')[-1] for s in (model_path / save_path).iterdir()]
    eids = [r.match(p).group('eid') for p in files if r.match(p)]
    if eids:
        max_eid = max(map(int, eids))
    
    if not flags.force and eids and flags.eid < max_eid:
        print("You will overwrite existing checkpoints. Add -f to force it.")
        return False

    return True


def run_train(flags):
    if not check_snapshots(flags):
        return 
    
    save_path = 'checkpoint'
    if flags.tag is not None:
        save_path += '_' + flags.tag

    save_train_config(flags)

    opt = util.load_opt(flags.name, save_path=save_path)
    optimizees = select_optimizees(flags)

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(config=util.get_tf_config(), graph=graph) as session:
            for optimizee in optimizees.values():
                optimizee.build()

            build_opt(opt, optimizees, flags)

            session.run(tf.global_variables_initializer(), {opt.train_lr: flags.train_lr, opt.momentum: flags.momentum})
            train_rets, test_rets = train_opt(opt, flags)
            
            model_path = util.get_model_path(flags.name)
            util.dump_results(model_path, (train_rets, test_rets), phase='train', tag=flags.tag)
