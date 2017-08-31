import re
import json
import tensorflow as tf
import optimizees as optim

import util
import util.paths as paths
import util.tf_utils as tf_utils

from opts import BuildConfig, TrainConfig


def will_overwrite_snapshots(snapshots_path, eid):
    if not snapshots_path.exists():
        return False

    snapshot_regex = re.compile(r'epoch-(?P<eid>\d+).index')

    eids = []

    for d in snapshots_path.iterdir():
        filename = str(d).split('/')[-1]
        m = snapshot_regex.match(filename)
        if m:
            e = m.group('eid')
            eids.append(e)

    if eids:
        max_eid = max(int(e) for e in eids)
        print(eid, max_eid)
        return int(eid) < max_eid

    return False


def setup_experiment(flags):
    model_path = paths.model_path(flags.name)

    print("Training model: ", flags.name)
    print("Model path: ", model_path) 
    print("Snapshots path: ", model_path / 'snapshots')

    if not flags.force and will_overwrite_snapshots(model_path / 'snapshots', flags.eid):
        raise RuntimeError("You will overwrite existing checkpoints. Add -f to force it.")

    if flags.eid >= flags.n_epochs:
        raise ValueError("Error: eid >= n_epochs")

    with (model_path / 'train_config.json').open('w' if flags.force else 'a') as conf:
        train_config = TrainConfig.from_namespace(flags)
        json.dump(vars(train_config), conf, sort_keys=True, indent=4)
    
    opt = util.load_opt(model_path)
    return opt, model_path


def build_opt(opt, flags):
    optimizees = optim.get_optimizees(flags.optimizee,
                                      clip_by_value=True,
                                      random_scale=flags.enable_random_scaling,
                                      noisy_grad=flags.noisy_grad)

    for optimizee in optimizees.values():
        optimizee.build()

    build_config = BuildConfig.from_namespace(flags)
    opt.build(optimizees, build_config)


def training(opt, flags):
    feed_dict = {
        opt.train_lr: flags.train_lr,
        opt.momentum: flags.momentum
    }
    session = tf.get_default_session()
    session.run(tf.global_variables_initializer(), feed_dict=feed_dict)

    try:
        train_config = TrainConfig.from_namespace(flags)
        rets = opt.train(train_config)
    except tf.errors.InvalidArgumentError as error:
        print("Op: ", error.op)
        print("Input: ", error.op.inputs)
        print(error)
        raise

    return rets


def run_train(flags):
    opt, model_path = setup_experiment(flags)

    with tf.Graph().as_default():
        build_opt(opt, flags)
        tf.set_random_seed(util.get_seed())

        with tf.Session(config=tf_utils.get_tf_config()).as_default():
            rets = training(opt, flags)

    util.dump_results(model_path, rets)
