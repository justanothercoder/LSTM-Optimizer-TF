#!/usr/bin/env python
import os
import pprint
import random
import numpy as np
import cli


if __name__ == '__main__':
    parser = cli.make_parser()

    flags = parser.parse_args()
    pprint.pprint(vars(flags))

    if not hasattr(flags, 'debug') or not flags.debug:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if hasattr(flags, 'seed') and flags.seed is not None:
        random.seed(flags.seed)
        np.random.seed(flags.seed)

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
    elif flags.command_name == 'explore':
        import explore
        explore.run_explore(flags)
    else:
        parser.print_help()
