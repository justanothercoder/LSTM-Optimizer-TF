import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from collections import OrderedDict
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn

import numpy as np
import tensorflow as tf

import util
from util import paths
from util import tf_utils
import optimizees as optim

@tf_utils.with_tf_graph
def build(flags, opt):
    optimizees = optim.get_optimizees(['rosenbrock'],
                                      clip_by_value=True,
                                      random_scale=False,
                                      noisy_grad=False)

    for o in optimizees.values():
        o.build()

    opt.build(optimizees, n_bptt_steps=1, inference_only=True)
    opt.restore(flags.eid)

    sess = tf.get_default_session()
    return sess.run(OrderedDict([(v.name, v) for v in opt.all_vars]))


def plot_hist(flags, weights, opt):
    ncols = 2 if opt.use_both else 1
    nrows = 7 if opt.with_log_features else 5

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 12))

    for i in range(nrows):
        for j in range(ncols):
            axes[i][j].set_ylim(0, 1)

    for name, v in weights.items():
        w = np.ones(60) / 60

        axes[0][0].hist(v[0:1, :60].reshape(-1), bins=100, weights=w)
        axes[0][0].set_title('g')
        axes[1][0].hist(v[1:2, :60].reshape(-1), bins=100, weights=w)
        axes[1][0].set_title('g2')
        axes[2][0].hist(v[2:3, :60].reshape(-1), bins=100, weights=w)
        axes[2][0].set_title('m')
        axes[3][0].hist(v[3:4, :60].reshape(-1), bins=100, weights=w)
        axes[3][0].set_title('v')
        axes[4][0].hist(v[4:5, :60].reshape(-1), bins=100, weights=w)
        axes[4][0].set_title('s')

        d = 0
        if opt.with_log_features:
            axes[5][0].hist(v[5:6, :60].reshape(-1), bins=100, weights=w)
            axes[5][0].set_title('log_g2')
            axes[6][0].hist(v[6:7, :60].reshape(-1), bins=100, weights=w)
            axes[6][0].set_title('log_v')
            d = 2


        if opt.use_both:
            axes[0][1].hist(v[5 + d:6 + d, :60].reshape(-1), bins=100, weights=w)
            axes[0][1].set_title('g_norm')
            axes[1][1].hist(v[6 + d:7 + d, :60].reshape(-1), bins=100, weights=w)
            axes[1][1].set_title('g2_norm')
            axes[2][1].hist(v[7 + d:8 + d, :60].reshape(-1), bins=100, weights=w)
            axes[2][1].set_title('m_norm')
            axes[3][1].hist(v[8 + d:9 + d, :60].reshape(-1), bins=100, weights=w)
            axes[3][1].set_title('v_norm')
            axes[4][1].hist(v[9 + d:10 + d, :60].reshape(-1), bins=100, weights=w)
            axes[4][1].set_title('s_norm')

            if opt.with_log_features:
                axes[5][1].hist(v[10 + d:11 + d, :60].reshape(-1), bins=100, weights=w)
                axes[5][1].set_title('log_g2_norm')
                axes[6][1].hist(v[11 + d:12 + d, :60].reshape(-1), bins=100, weights=w)
                axes[6][1].set_title('log_v_norm')

        path = pathlib.Path('hists/{}/eid={}/'.format(flags.name, flags.eid))
        paths.make_dirs(path)

        filename = str(path / (name.replace('/', '_') + '_.png'))
        fig.tight_layout()
        fig.savefig(filename, format='png')

        break


def run_explore(flags):
    model_path = paths.model_path(flags.name)
    opt = util.load_opt(model_path)

    weights = build(flags, opt)
    plot_hist(flags, weights, opt)

