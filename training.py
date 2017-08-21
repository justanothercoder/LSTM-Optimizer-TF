import re
import sys
import json
import tensorflow as tf
import optimizees as optim

import util
import util.paths as paths
import util.tf_utils as tf_utils

from opts import model_trainer, distributed
from opts.basic_model import BuildConfig


def will_overwrite_snapshots(snapshots_path, eid):
    if not snapshots_path.exists():
        return False

    snapshot_regex = re.compile(r'epoch-(?P<eid>\d+).index')

    files = [str(s).split('/')[-1] for s in snapshots_path.iterdir()]
    eids = [snapshot_regex.match(p).group('eid') for p in files if snapshot_regex.match(p)]
    if eids:
        max_eid = max(int(eid) for eid in eids)

    if eids and eid < max_eid:
        return True

    return False


def setup_experiment(flags):
    model_path = paths.model_path(flags.name)

    print("Training model: ", flags.name)
    print("Model path: ", model_path) 
    print("Snapshots path: ", model_path / 'snapshots')

    if not flags.force and will_overwrite_snapshots(model_path / 'snapshots', flags.eid):
        print("You will overwrite existing checkpoints. Add -f to force it.")
        sys.exit(1)

    if flags.eid >= flags.n_epochs:
        print("Error: eid >= n_epochs")
        sys.exit(1)

    open_mode = 'w' if flags.force else 'a'
    with (model_path / 'train_config.json').open(open_mode) as conf:
        bad_kws = {'name', 'experiment_name', 'gpu', 'cpu', 'command_name', 'debug', 'force', 'verbose'}
        training_options = {k: v for k, v in vars(flags).items() if k not in bad_kws}
        json.dump(training_options, conf, sort_keys=True, indent=4)
    
    opt = util.load_opt(model_path)
    opt.debug = flags.debug
    return opt, model_path


def build_opt(opt, flags):
    optimizees = optim.get_optimizees(flags.optimizee,
                                      clip_by_value=True,
                                      random_scale=flags.enable_random_scaling,
                                      noisy_grad=flags.noisy_grad)

    for optimizee in optimizees.values():
        optimizee.build()

    opt = distributed.distribute(opt, tf_utils.get_devices(flags))

    #kwargs = util.get_kwargs(opt.build, flags)
    #opt.build(optimizees, **kwargs)

    kwargs = util.get_kwargs(BuildConfig, flags)
    build_config = BuildConfig(**kwargs)
    opt.build(optimizees, build_config)


def training(opt, flags):
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
    except tf.errors.InvalidArgumentError as error:
        print("Op: ", error.op)
        print("Input: ", error.op.inputs)
        print(error)
        raise

    return rets


@tf_utils.with_tf_graph
def run_train(flags):
    opt, model_path = setup_experiment(flags)

    build_opt(opt, flags)
    rets = training(opt, flags)

    util.dump_results(model_path, rets)
