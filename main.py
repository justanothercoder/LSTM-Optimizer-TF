#!/usr/bin/env python

import os, pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pprint
import subprocess, shlex

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


if __name__ == '__main__':
    parser = cli.make_parser(run_train=run_train, run_test=run_test, run_plot=run_plot, run_cv=run_cv)

    flags = parser.parse_args()
    pprint.pprint(vars(flags))

    path = pathlib.Path('models') / flags.name
    flags.model_path = path
    
    subprocess.call(shlex.split('mkdir -p {}'.format(path / 'train')))
    subprocess.call(shlex.split('mkdir -p {}'.format(path / 'test')))
    subprocess.call(shlex.split('mkdir -p {}'.format(path / 'cv' / 'snapshots')))

    #subprocess.call(shlex.split('mkdir -p models/{model_name}/train/'.format(model_name=flags.name)))
    #subprocess.call(shlex.split('mkdir -p models/{model_name}/test/'.format(model_name=flags.name)))
    #subprocess.call(shlex.split('mkdir -p models/{model_name}/cv/'.format(model_name=flags.name)))

    if flags.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(flags.gpu)

    flags.func(flags)
