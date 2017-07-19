"""
This module defines run_train function which setups everything for training
of the model.
"""

import re
import json
import tensorflow as tf
import optimizees as optim
import paths
import util


def save_train_config(flags, experiment_path):
    """This function dump training config to directory where model lies."""

    training_options = {
        'batch_size', 'enable_random_scaling', 'loss_type',
        'n_batches', 'n_bptt_steps', 'n_epochs', 'n_steps',
        'optimizee', 'train_lr', 'momentum', 'optimizer', 'lambd'
    }

    training_config = {k: v for k, v in vars(flags).items() if k in training_options}
    print('Training config: ', training_config)

    with (experiment_path / 'config').open('w') as conf:
        json.dump(training_config, conf, sort_keys=True, indent=4)


def train_opt(opt, flags):
    """This function extracts relevant flags and runs training of optimizer."""
    train_options = {
        'n_epochs', 'n_batches', 'batch_size',
        'n_steps', 'eid', 'train_lr', 'momentum',
        'verbose'
    }

    train_options = {k: v for k, v in vars(flags).items() if k in train_options}
    return opt.train(**train_options)


def will_overwrite_snapshots(snapshots_path, eid):
    """This function checks whether snapshots will be overwritten by running training."""
    if not snapshots_path.exists():
        return False

    snapshot_regex = re.compile(r'epoch-(?P<eid>\d+).index')

    files = [str(s).split('/')[-1] for s in snapshots_path.iterdir()]
    eids = [snapshot_regex.match(p).group('eid') for p in files if snapshot_regex.match(p)]
    if eids:
        max_eid = max(int(eid) for eid in eids)

    if eids and eid < max_eid:
        print("You will overwrite existing checkpoints. Add -f to force it.")
        return True

    return False


def run_train(flags):
    """This function runs training of optimizer."""
    model_path = paths.model_path(flags.name)
    experiment_path = paths.experiment_path(flags.name, flags.experiment_name, 'train')
    snapshots_path = paths.snapshots_path(experiment_path)

    print("Running experiment: ", flags.experiment_name)
    print("Experiment path: ", experiment_path)
    print("Snapshots path: ", snapshots_path)

    if not flags.force and will_overwrite_snapshots(snapshots_path, flags.eid):
        return

    paths.make_dirs(experiment_path, snapshots_path)
    save_train_config(flags, experiment_path)

    optimizees = optim.get_optimizees(flags.optimizee,
                                      clip_by_value=True,
                                      random_scale=flags.enable_random_scaling,
                                      noisy_grad=flags.noisy_grad)

    graph = tf.Graph()
    session = tf.Session(config=util.get_tf_config(), graph=graph)
    with graph.as_default(), session:
        for optimizee in optimizees.values():
            optimizee.build()

        opt = util.load_opt(model_path, experiment_path)
        opt.build(optimizees,
                  n_bptt_steps=flags.n_bptt_steps,
                  loss_type=flags.loss_type,
                  optimizer=flags.optimizer,
                  lambd=flags.lambd,
                  devices=util.get_devices(flags))

        feed_dict = {
            opt.train_lr: flags.train_lr, 
            opt.momentum: flags.momentum
        }
        session.run(tf.global_variables_initializer(), feed_dict=feed_dict)
        rets = train_opt(opt, flags)

        util.dump_results(experiment_path, rets)
