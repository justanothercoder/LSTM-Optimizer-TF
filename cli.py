import argparse

def make_train_parser(parser):
    parser.add_argument('--eid', type=int, default=0, help='epoch id from which start training')

    parser.add_argument('--optimizer', type=str,
                        default='adam', choices=['adam', 'momentum', 'yellowfin'],
                        help='optimizer to train LSTM')

    parser.add_argument('--normalize_lstm_grads', action='store_true')

    parser.add_argument('--train_lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--grad_clip', type=float, default=1.)

    parser.add_argument('--loss_type', type=str,
                        choices=['log', 'sum', 'last'], default='log',
                        help='loss function to use')

    parser.add_argument('--lambd', type=float, default=0)
    parser.add_argument('--lambd-l1', type=float, default=0)

    parser.add_argument('--n_steps', type=int, default=100, help='number of steps')
    parser.add_argument('--n_bptt_steps', type=int, default=20, help='number of bptt steps')

    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--n_batches', type=int, default=100, help='number of batches per epoch')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')

    parser.add_argument('-opt', '--optimizee', type=str, nargs='+', default='all',
                        help='space separated list of optimizees or all')

    parser.add_argument('--enable_random_scaling', action='store_true',
                        help='enable random scaling of problems')

    parser.add_argument('--noisy_grad', action='store_true',
                        help='add normal noise to gradients of non-stochastic problems')
    parser.add_argument('--no_stop_grad', action='store_false', dest='stop_grad',
                        help='whether to compute second derivatives')

    parser.add_argument('-f', '--force', action='store_true', help='force overwrite of checkpoint')

    return parser


def make_test_parser(parser):
    parser.add_argument('experiment_name', type=str, help='name of the experiment')
    parser.add_argument('eid', type=int, help='epoch id from which test optimizer')
    parser.add_argument('mode', type=str, choices=['many', 'cv'], help='which mode to run')
    parser.add_argument('problems', type=str, nargs='+', help='problem to run test on')

    parser.add_argument('--enable_random_scaling', action='store_true',
                        help='enable random scaling of problems')
    parser.add_argument('--noisy_grad', action='store_true',
                        help='add normal noise to gradients of non-stochastic problems')

    parser.add_argument('--n_steps', type=int, default=1000, help='number of steps')
    parser.add_argument('--n_batches', type=int, default=20, help='number of batches per epoch')

    parser.add_argument('--compare_with', type=str, default='adam',
                        choices=['sgd', 'momentum', 'adam', 'adamng'], help='baseline for optimizer')
    parser.add_argument('--adam_only', action='store_true', help="enable: step from ADAM, learning rate from LSTM")

    parser.add_argument('--start_eid', type=int, default=100,
                        help='epoch from which start to run cv')
    parser.add_argument('--step', type=int, default=100, help='step in number of epochs for cv')
    parser.add_argument('-f', '--force', action='store_true', help='force overwrite of results')
    parser.add_argument('--use-moving-averages', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='random seed to set in the beginning (for the reproducibility)')

    return parser


def make_plot_parser(parser):
    parser.add_argument('phase', type=str, choices=['train', 'test', 'cv'],
                        help='train or test phase')
    parser.add_argument('name', type=str, help='name of the model')
    parser.add_argument('--no-plot-lr', action='store_false', dest='plot_lr',
                        help='plot learning rate')
    parser.add_argument('--no-plot-moving', action='store_false', dest='plot_moving',
                        help='plot moving loss')
    parser.add_argument('-p', '--problems', type=str, nargs='+',
                        help='optimizee name')
    parser.add_argument('-m', '--mode', type=str, choices=['many', 'cv'], default='many',
                        help='mode of testing')
    parser.add_argument('-f', '--frac', type=float, default=1.0,
                        help='fraction of data to plot')
    parser.add_argument('-s', '--stochastic', action='store_true',
                        help='whether problem is stochastic')
    parser.add_argument('--compare_with', type=str, default='adam')
    parser.add_argument('--enable_random_scaling', action='store_true')

    return parser


def make_cv_parser(parser):
    methods = ['grid', 'random', 'bayesian']

    parser.add_argument('config', type=str,
                        help='path to parameter grid')
    parser.add_argument('--method', type=str, choices=methods, default='grid',
                        help='type of tuning')
    parser.add_argument('--num_tries', type=int, default=5,
                        help='number of tries for random cv')

    return parser


def make_new_parser(parser):
    parser.add_argument('name', type=str, help='name of the model')
    parser.add_argument('num_units', type=int, help='number of units in LSTM')
    parser.add_argument('num_layers', type=int, help='number of lstm layers')
    parser.add_argument('--no-layer-norm', action='store_false', dest='layer_norm',
                        help='enable layer normalization')
    parser.add_argument('--no-add-skip', action='store_false', dest='add_skip',
                        help='add adam output to LSTM output')
    parser.add_argument('--rnn-type', type=str, choices=['lstm', 'gru'], default='lstm',
                        help='cell to use: LSTM or GRU')
    parser.add_argument('--no-residual', action='store_false', dest='residual',
                        help='add residual connections in lstm')
    parser.add_argument('--learn-init', action='store_true',
                        help='learn initial hidden state')
    parser.add_argument('--with-log-features', action='store_true',
                        help='add logarithmic features')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--normalize_gradients', action='store_true',
                       help='normalize_gradients')
    group.add_argument('--rmsprop_gradients', action='store_true',
                       help='rmsprop_gradients')
    group.add_argument('--use_both', action='store_true',
                       help='use both normalized and unnormalized gradients')

    parser.add_argument('-f', '--force', action='store_true')
    return parser


def make_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    subparsers = parser.add_subparsers(help='command to run', dest='command_name')

    run_parser = argparse.ArgumentParser(add_help=False)
    run_parser.add_argument('name', type=str, help='name of the model')

    group = run_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cpu', action='store_true', help='run model on CPU')
    group.add_argument('--gpu', type=int, nargs='+', help='gpu id')

    run_parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=1)
    run_parser.add_argument('--debug', action='store_true')

    parser_new = subparsers.add_parser('new', help='add new model')
    parser_train = subparsers.add_parser('train', parents=[run_parser],
                                         help='train optimizer on a set of functions')
    parser_test = subparsers.add_parser('test', parents=[run_parser],
                                        help='run trained optimizer on some problem')
    parser_plot = subparsers.add_parser('plot', help='plot dumped results')

    parser_new = make_new_parser(parser_new)
    parser_train = make_train_parser(parser_train)
    parser_test = make_test_parser(parser_test)

    parser_cv = subparsers.add_parser('cv', parents=[parser_train], add_help=False,
                                      help='tune hyperparameters by validation')
    parser_cv = make_cv_parser(parser_cv)
    parser_plot = make_plot_parser(parser_plot)

    return parser
