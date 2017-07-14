import argparse

problems = ['quadratic', 'rosenbrock', 'mixed', 'logreg', 'stoch_logreg', 'stoch_linear', 'mixed_stoch']


def make_train_parser(parser_train, run_train):
    parser_train.add_argument('--eid', type=int, default=0, help='epoch id from which train optimizer')
    parser_train.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'momentum', 'yellowfin'], help='optimizer to train LSTM')
    parser_train.add_argument('--train_lr', type=float, default=1e-2, help='learning rate')
    parser_train.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser_train.add_argument('--loss_type', type=str, choices=['log', 'sum', 'last'], default='log', help='loss function to use')
    parser_train.add_argument('--lambd', type=float, default=1e-5)

    parser_train.add_argument('--n_steps', type=int, default=100, help='number of steps')
    parser_train.add_argument('--n_bptt_steps', type=int, default=20, help='number of bptt steps')
    
    parser_train.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    parser_train.add_argument('--n_batches', type=int, default=100, help='number of batches per epoch')
    parser_train.add_argument('--batch_size', type=int, default=100, help='batch size')
    
    parser_train.add_argument('--optimizee', type=str, nargs='+', choices=problems, default='all', help='space separated list of optimizees or all')
    parser_train.add_argument('--enable_random_scaling', action='store_true', help='enable random scaling of problems')
    parser_train.add_argument('--noisy_grad', action='store_true', help='add normal noise to gradients of non-stochastic problems')
    parser_train.add_argument('-f', '--force', action='store_true', help='force overwrite of checkpoint')

    parser_train.set_defaults(func=run_train)
    return parser_train


def make_test_parser(parser_test, run_test):
    parser_test.add_argument('eid', type=int, help='epoch id from which test optimizer')
    parser_test.add_argument('problem', choices=problems, help='problem to run test on')
    parser_test.add_argument('mode', type=str, choices=['many', 'cv'], help='which mode to run')
    parser_test.add_argument('--enable_random_scaling', action='store_true', help='enable random scaling of problems')
    parser_test.add_argument('--noisy_grad', action='store_true', help='add normal noise to gradients of non-stochastic problems')

    parser_test.add_argument('--n_steps', type=int, default=100, help='number of steps')
    parser_test.add_argument('--n_batches', type=int, default=100, help='number of batches per epoch')

    parser_test.add_argument('--compare_with', type=str, default='momentum', choices=['sgd', 'momentum'], help='baseline for optimizer')
    
    parser_test.add_argument('--start_eid', type=int, default=100, help='epoch from which start to run cv')
    parser_test.add_argument('--step', type=int, default=100, help='step in number of epochs for cv')

    parser_test.set_defaults(func=run_test)
    return parser_test


def make_plot_parser(parser_plot, run_plot):
    parser_plot.add_argument('name', type=str, help='name of the model')
    parser_plot.add_argument('phase', type=str, choices=['train', 'test', 'cv'], help='train or test phase')
    parser_plot.add_argument('--plot_lr', action='store_true', help='plot learning rate')
    parser_plot.add_argument('--plot_moving', action='store_true', help='plot moving loss')
    parser_plot.add_argument('-p', '--problem', type=str, help='optimizee name')
    parser_plot.add_argument('-m', '--mode', type=str, choices=['many', 'cv'], help='mode of testing')
    parser_plot.add_argument('-f', '--frac', type=float, default=1.0, help='fraction of data to plot')
    parser_plot.add_argument('-t', '--tag', type=str, help='tag')
    parser_plot.add_argument('-s', '--stochastic', action='store_true', help='whether problem is stochastic')

    parser_plot.set_defaults(func=run_plot)
    return parser_plot


def make_cv_parser(parser_cv, run_cv):
    parser_cv.add_argument('config', type=str, help='path to parameter grid')
    parser_cv.add_argument('--method', type=str, choices=['grid', 'random', 'bayesian'], default='grid', help='type of tuning')
    parser_cv.add_argument('--num_tries', type=int, default=5, help='number of tries for random cv')

    parser_cv.set_defaults(func=run_cv)
    return parser_cv


def make_new_parser(parser_new, run_new):
    parser_new.add_argument('name', type=str, help='name of the model')
    parser_new.add_argument('num_units', type=int, help='number of units in LSTM')
    parser_new.add_argument('num_layers', type=int, help='number of lstm layers')
    parser_new.add_argument('--layer_norm', action='store_true', help='enable layer normalization')
    parser_new.add_argument('--add_skip', action='store_true', help='add adam output to LSTM output')
    parser_new.add_argument('--no_stop_grad', action='store_false', dest='stop_grad', help='whether to compute second derivatives')

    parser_new.set_defaults(func=run_new)
    return parser_new


def make_parser_for_command_with_run():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('name', type=str, help='name of the model')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cpu', action='store_true', help='run model on CPU')
    group.add_argument('--gpu', type=int, nargs='+', help='gpu id')

    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=1)
    parser.add_argument('--tag', type=str, help='tag denoting run purpose/parameters')
    return parser
    

def make_parser(commands):
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    subparsers = parser.add_subparsers(help='command to run')

    run_parser = make_parser_for_command_with_run()

    parser_new = subparsers.add_parser('new', help='add new model')
    parser_new = make_new_parser(parser_new, commands['new'])

    parser_train = subparsers.add_parser('train', parents=[run_parser], help='train optimizer on a set of functions')
    parser_train = make_train_parser(parser_train, commands['train'])
    
    parser_test = subparsers.add_parser('test', parents=[run_parser], help='run trained optimizer on some problem')
    parser_test = make_test_parser(parser_test, commands['test'])

    parser_cv = subparsers.add_parser('cv', parents=[parser_train], add_help=False, help='tune hyperparameters by validation')
    parser_cv = make_cv_parser(parser_cv, commands['cv'])
    
    parser_plot = subparsers.add_parser('plot', help='plot dumped results')
    parser_plot = make_plot_parser(parser_plot, commands['plot'])

    return parser
