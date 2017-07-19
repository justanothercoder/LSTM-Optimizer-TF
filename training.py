"""
This module defines run_train function which setups everything for training
of the model.
"""

import re
import json
import subprocess
import shlex
import tensorflow as tf
import util
import optimizees as optim

def save_train_config(flags):
    """This function dump training config to directory where model lies."""

    training_options = {
        'batch_size', 'enable_random_scaling', 'loss_type',
        'n_batches', 'n_bptt_steps', 'n_epochs', 'n_steps',
        'optimizee', 'train_lr', 'momentum', 'optimizer', 'lambd'
    }

    training_config = {k: v for k, v in vars(flags).items() if k in training_options}
    print('Training config: ', training_config)

    conf_path = util.get_model_path(flags.name) / 'train'/ 'config'
    with conf_path.open('w') as conf:
        json.dump(training_config, conf, sort_keys=True, indent=4)


def build_opt(opt, optimizees, flags):
    """This function setups and runs model building."""
    if flags.gpu is not None:
        devices = ['/gpu:{}'.format(i) for i in range(len(flags.gpu))]
    else:
        devices = ['/cpu:0']
        #devices = ['/cpu:%d' % i for i in range(util.get_tf_config().device_count["CPU"])]

    opt.build(optimizees,
              n_bptt_steps=flags.n_bptt_steps,
              loss_type=flags.loss_type,
              optimizer=flags.optimizer,
              lambd=flags.lambd,
              devices=devices)


def train_opt(opt, flags):
    """This function extracts relevant flags and runs training of optimizer."""
    train_options = {
        'n_epochs', 'n_batches', 'batch_size',
        'n_steps', 'eid', 'train_lr', 'momentum',
        'verbose'
    }

    train_options = {k: v for k, v in vars(flags).items() if k in train_options}
    return opt.train(**train_options)


def check_snapshots(flags):
    """This function checks whether snapshots will be overwritten by running training."""
    model_path = util.get_model_path(flags.name)

    save_path = 'snapshots'
    if flags.tag is not None:
        save_path += '_' + flags.tag

    subprocess.call(shlex.split('mkdir -p {}'.format(model_path / save_path)))

    snapshot_regex = re.compile(r'epoch-(?P<eid>\d+).index')

    files = [str(s).split('/')[-1] for s in (model_path / save_path).iterdir()]
    eids = [snapshot_regex.match(p).group('eid') for p in files if snapshot_regex.match(p)]
    if eids:
        max_eid = max(int(eid) for eid in eids)

    if not flags.force and eids and flags.eid < max_eid:
        print("You will overwrite existing checkpoints. Add -f to force it.")
        return False

    return True


def run_train(flags):
    """This function runs training of optimizer."""
    if not check_snapshots(flags):
        return

    save_path = 'snapshots'
    if flags.tag is not None:
        save_path += '_' + flags.tag

    save_train_config(flags)

    opt = util.load_opt(flags.name, save_path=save_path)
    optimizees = optim.get_optimizees(flags.optimizee,
                                      clip_by_value=True,
                                      random_scale=flags.enable_random_scaling,
                                      noisy_grad=flags.noisy_grad)

    graph = tf.Graph()
    session = tf.Session(config=util.get_tf_config(), graph=graph)
    with graph.as_default(), session:
        for optimizee in optimizees.values():
            optimizee.build()

        build_opt(opt, optimizees, flags)

        feed_dict = {opt.train_lr: flags.train_lr, opt.momentum: flags.momentum}
        session.run(tf.global_variables_initializer(), feed_dict=feed_dict)
        train_rets, test_rets = train_opt(opt, flags)

        model_path = util.get_model_path(flags.name)
        util.dump_results(model_path, (train_rets, test_rets), phase='train', tag=flags.tag)
