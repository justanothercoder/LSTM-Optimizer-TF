import argparse


def make_train_parser(parser_train, run_train):
    parser_train.add_argument('name', type=str, help='name of the model')
    parser_train.add_argument('--optimizee', type=str, nargs='+', default='all', help='space separated list of optimizees or all')
    parser_train.add_argument('--n_steps', type=int, default=100, help='number of steps')
    parser_train.add_argument('--n_bptt_steps', type=int, default=20, help='number of bptt steps')
    parser_train.add_argument('--n_batches', type=int, default=100, help='number of batches per epoch')
    parser_train.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    parser_train.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser_train.add_argument('--train_lr', type=float, default=1e-2, help='learning rate')
    parser_train.add_argument('--loss_type', type=str, choices=['log', 'sum', 'last'], default='log', help='loss function to use')
    parser_train.add_argument('--no_stop_grad', action='store_false', dest='stop_grad', help='whether to count second derivatives')

    parser_train.set_defaults(func=run_train)
    return parser_train


def make_test_parser(parser_test, run_test):
    parser_test.add_argument('name', type=str, help='name of the model')
    parser_test.add_argument('problem', choices=['quadratic', 'rosenbrock', 'mixed', 'logreg'], help='problem to run test on')
    parser_test.add_argument('mode', type=str, choices=['many', 'cv'], help='which mode to run')
    parser_test.add_argument('--n_steps', type=int, default=100, help='number of steps')
    parser_test.add_argument('--n_batches', type=int, default=100, help='number of batches per epoch')
    parser_test.add_argument('--start_eid', type=int, default=100, help='epoch from which start to run cv')
    parser_test.add_argument('--step', type=int, default=100, help='step in number of epochs for cv')
    parser_test.add_argument('--compare_with', type=str, default='momentum', choices=['sgd', 'momentum'], help='baseline for optimizer')

    parser_test.set_defaults(func=run_test)
    return parser_test


def make_plot_parser(parser_plot, run_plot):
    parser_plot.add_argument('name', type=str, help='name of the model')
    parser_plot.add_argument('phase', type=str, choices=['train', 'test', 'cv'], help='train or test phase')
    parser_plot.add_argument('--problem', type=str, help='optimizee name')
    parser_plot.add_argument('--mode', type=str, choices=['many', 'cv'], help='mode of testing')
    parser_plot.add_argument('--plot_lr', action='store_true', help='enable plotting of learning rate')
    parser_plot.add_argument('--frac', type=float, default=1.0, help='fraction of data to plot')
    parser_plot.add_argument('--plot_moving', action='store_true', help='plot moving loss')

    parser_plot.set_defaults(func=run_plot)
    return parser_plot


def make_cv_parser(parser_cv, run_cv):
    parser_cv.add_argument('name', type=str, help='name of the model')
    parser_cv.add_argument('config', type=str, help='path to parameter grid')
    parser_cv.add_argument('--method', type=str, choices=['grid', 'random', 'bayesian'], default='grid', help='type of tuning')
    parser_cv.add_argument('--n_steps', type=int, default=100, help='number of steps')
    parser_cv.add_argument('--n_bptt_steps', type=int, default=20, help='number of bptt steps')
    parser_cv.add_argument('--n_batches', type=int, default=100, help='number of batches per epoch')
    parser_cv.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    parser_cv.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser_cv.add_argument('--train_lr', type=float, default=1e-2, help='learning rate')
    parser_cv.add_argument('--loss_type', type=str, choices=['log', 'sum', 'last'], default='log', help='loss function to use')

    parser_cv.set_defaults(func=run_cv)
    return parser_cv


def make_parser(*, run_train, run_test, run_plot, run_cv):
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--cpu', action='store_true', help='run model on CPU')
    parser.add_argument('--gpu', type=int, default=2, help='gpu id')
    parser.add_argument('--eid', type=int, default=0, help='epoch id from which train/test optimizer')
    parser.add_argument('--num_units', type=int, default=20, help='number of units in LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='number of lstm layers')
    parser.add_argument('--layer_norm', action='store_true', help='enable layer normalization')
    parser.add_argument('--add_skip', action='store_true', help='add adam output to LSTM output')
    parser.add_argument('--enable_random_scaling', action='store_true', help='enable random scaling of problems')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=1)

    subparsers = parser.add_subparsers(help='mode: train or test')

    parser_train = subparsers.add_parser('train', help='train optimizer on a set of functions')
    parser_train = make_train_parser(parser_train, run_train)
    
    parser_test = subparsers.add_parser('test', help='run trained optimizer on some problem')
    parser_test = make_test_parser(parser_test, run_test)

    parser_plot = subparsers.add_parser('plot', help='plot dumped results')
    parser_plot = make_plot_parser(parser_plot, run_plot)

    parser_cv = subparsers.add_parser('cv', help='tune hyperparameters by validation')
    parser_cv = make_cv_parser(parser_cv, run_cv)

    return parser
