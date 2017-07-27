"""
This module defines run_train function which setups everything for training
of the model.
"""

import re
import inspect
import tensorflow as tf
import optimizees as optim

import util
import util.paths as paths
import util.tf_utils as tf_utils

from opts import model_trainer, distributed


def save_train_config(flags, experiment_path):
    """This function dumps training config to directory where model lies."""
    training_options = {
        'batch_size', 'enable_random_scaling', 'loss_type',
        'n_batches', 'n_bptt_steps', 'n_epochs', 'n_steps',
        'optimizee', 'train_lr', 'momentum', 'optimizer', 'lambd',
        'noisy_grad'
    }
    util.dump_config(experiment_path / 'config', flags, training_options)


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


def setup_experiment(flags):
    """Setups directories and loads optimizer"""
    model_path = paths.model_path(flags.name)
    experiment_path = paths.experiment_path(flags.name, flags.experiment_name, 'train')
    snapshots_path = paths.snapshots_path(experiment_path)

    print("Running experiment: ", flags.experiment_name)
    print("Experiment path: ", experiment_path)
    print("Snapshots path: ", snapshots_path)

    if not flags.force and will_overwrite_snapshots(snapshots_path, flags.eid):
        return None

    paths.make_dirs(experiment_path, snapshots_path)
    save_train_config(flags, experiment_path)

    opt = util.load_opt(model_path, experiment_path)
    opt.debug = flags.debug
    return experiment_path, opt


@tf_utils.with_tf_graph
def training(flags, opt):
    """This function runs training of optimizer."""
    optimizees = optim.get_optimizees(flags.optimizee,
                                      clip_by_value=True,
                                      random_scale=flags.enable_random_scaling,
                                      noisy_grad=flags.noisy_grad)

    for optimizee in optimizees.values():
        optimizee.build()

    opt = distributed.DistributedModel(opt, tf_utils.get_devices(flags))

    kwargs = util.get_kwargs(opt.build, flags)
    opt.build(optimizees, **kwargs)

    feed_dict = {
        opt.train_lr: flags.train_lr,
        opt.momentum: flags.momentum
    }
    session = tf.get_default_session()
    session.run(tf.global_variables_initializer(), feed_dict=feed_dict)
    
    trainer = model_trainer.Trainer()
    try:
        kwargs = util.get_kwargs(trainer.train, flags)
        rets = trainer.setup_and_run(opt, 'train', session=session, **kwargs)
    except tf.errors.InvalidArgumentError as e:
        print("Op: ", e.op)
        print("Input: ", e.op.inputs)
        print(e)
        raise

    return rets


def run_train(flags):
    """Entry-point function: setups and runs experiment."""
    out = setup_experiment(flags)
    if out is None:
        return

    experiment_path, opt = out

    rets = training(flags, opt)
    util.dump_results(experiment_path, rets)
