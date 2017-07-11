#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pprint

import cli


def run_train(flags):
    import training
    training.run_train(flags)


def run_test(flags):
    import testing
    testing.run_test(flags)


def run_plot(flags):
    import plotting
    plotting.run_plot(flags)


def run_cv(flags):
    import cv
    cv.run_cv(flags)


def run_new(flags):
    import util
    util.run_new(flags)


if __name__ == '__main__':
    commands = {
        'train': run_train,
        'test' : run_test,
        'cv'   : run_cv,
        'plot' : run_plot,
        'new'  : run_new
    }
    parser = cli.make_parser(commands)

    flags = parser.parse_args()
    pprint.pprint(vars(flags))

    if hasattr(flags, 'cpu') and flags.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif hasattr(flags, 'gpu') and flags.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, flags.gpu))

    flags.func(flags)
