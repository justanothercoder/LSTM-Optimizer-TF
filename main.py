#!/usr/bin/env python
"""
This script is the entry point of program.
It builds and runs parser on passed command-line arguments.
Also it handles some environemnt variables.
"""
import pprint
import os
import cli


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    parser = cli.make_parser()

    flags = parser.parse_args()
    pprint.pprint(vars(flags))

    if hasattr(flags, 'cpu') and flags.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif hasattr(flags, 'gpu') and flags.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in flags.gpu)

    if flags.command_name == 'train':
        import training
        training.run_train(flags)
    elif flags.command_name == 'test':
        import testing
        testing.run_test(flags)
    elif flags.command_name == 'cv':
        import cv
        cv.run_cv(flags)
    elif flags.command_name == 'plot':
        import plotting
        plotting.run_plot(flags)
    elif flags.command_name == 'new':
        import util
        util.run_new(flags)
    elif flags.commad_name == 'explore':
        import explore
        explore.run_explore(flags)
    else:
        parser.print_help()
