from .quadratic import Quadratic
from .rosenbrock import Rosenbrock
from .beale import Beale
from .booth import Booth
from .matyas import Matyas
from .logistic_regression import LogisticRegression
from .stochastic_logistic_regression import StochasticLogisticRegression
from .stochastic_linear_regression import StochasticLinearRegression

from .correct_stoch_logreg import CorrectStochLogreg
from .correct_stoch_linreg import CorrectStochLinreg

from .mlp_classifier import MLPClassifier
from .conv_classifier import ConvClassifier
from .lstm_ptb import LSTM_PTB

from . import transformers

from . import from_rnnprop
from .rnnprop_adapt import RNNPropAdapter

problems = [
    'quadratic', 'rosenbrock', 'logreg',
    'stoch_logreg', 'stoch_linear',
    'noisy_stoch_logreg', 'noisy_stoch_linear',
    'correct_stoch_logreg', 'correct_stoch_linear',
    'small_correct_stoch_logreg', 'small_correct_stoch_linear',

    'stoch_logreg_10', 
    'mixed', 'mixed_stoch', 'mixed_nonstoch',
    'correct_mixed_stoch',

    '_digits_classifier',

    'conv_digits_classifier', 'conv_digits_classifier_2',
    'beale', 'booth', 'matyas', 'stoch_only',
    'vgg-cifar-10',
    'lstm_ptb',

    'digits_classifier',
    'digits_classifier_2',
    'digits_classifier_3',
    'digits_classifier_6',
    'digits_classifier_12',
    'digits_classifier_18',

    'digits_classifier_relu',
    'digits_classifier_relu_2',
    'digits_classifier_relu_3',
    'digits_classifier_relu_6',
    'digits_classifier_relu_12',
    'digits_classifier_relu_18',
    
    '_mnist_classifier',
    'mnist_classifier',
    'mnist_classifier_3',
    'mnist_classifier_6',
    'mnist_classifier_12',
    'mnist_classifier_18',
    
    'mnist_classifier_relu',
    'mnist_classifier_relu_3',
    'mnist_classifier_relu_6',
    'mnist_classifier_relu_12',
    'mnist_classifier_relu_18',
]


rnnprop_problems = [
    'mnist-nn-l2-sigmoid-100',
    'mnist-nn-l2-relu-100',
    'mnist-nn-l2-elu-100',
    'mnist-nn-l2-tanh-100',

    'mnist-nn-sigmoid-100',
    'mnist-nn-relu-100',
    'mnist-nn-elu-100',
    'mnist-nn-tanh-100',
    'mnist-nn-l2-sigmoid-100',
    'mnist-nn-l3-sigmoid-100',
    'mnist-nn-l4-sigmoid-100',
    'mnist-nn-l5-sigmoid-100',
    'mnist-nn-l6-sigmoid-100',
    'mnist-nn-l7-sigmoid-100',
    'mnist-nn-l8-sigmoid-100',
    'mnist-nn-l9-sigmoid-100',
    'mnist-nn-l10-sigmoid-100',
    'vgg-mnist-fc1-conv2-pool1-100',
    'vgg-cifar-fc1-conv2-pool1-100',
    'vgg-mnist-fc2-conv4-pool2-100',
    'vgg-cifar-fc2-conv4-pool2-100',
    'sin_lstm',
    'sin_lstm-x2',
    'sin_lstm-no001',
    
    '_mnist-nn-sigmoid-100',
    '_mnist-nn-relu-100',
    '_mnist-nn-elu-100',

    'digits_classifier_0',
    '_digits_classifier_0',
    'digits_classifier_0_random',
    '_digits_classifier_0_random'
]

problems.extend(rnnprop_problems)


def get_optimizees(problems_list, clip_by_value=False, random_scale=False, noisy_grad=False):
    optimizees = {
        'quadratic'   : Quadratic(low=50, high=100),
        'rosenbrock'  : Rosenbrock(low=2, high=10),
        'beale'       : Beale(low=2, high=10),
        'booth'       : Booth(low=2, high=10),
        'matyas'      : transformers.UniformRandomScaling(Matyas(low=2, high=10), r=3.0),
        'logreg'      : LogisticRegression(max_data_size=1000, max_features=100),
        'stoch_logreg': StochasticLogisticRegression(max_data_size=1000, max_features=100),
        'stoch_linear': StochasticLinearRegression(max_data_size=1000, max_features=100),
        
        'correct_stoch_logreg': CorrectStochLogreg(max_data_size=1000, max_features=100),
        'correct_stoch_linear': CorrectStochLinreg(max_data_size=1000, max_features=100),

        'small_correct_stoch_logreg': CorrectStochLogreg(max_data_size=100, max_features=10, min_data_size=10),
        'small_correct_stoch_linear': CorrectStochLinreg(max_data_size=100, max_features=10, min_data_size=10),

        'noisy_1_correct_stoch_logreg': transformers.NormalNoisyGrad(CorrectStochLogreg(max_data_size=1000, max_features=100), stddev=1e-1),
        'noisy_1_correct_stoch_linear': transformers.NormalNoisyGrad(CorrectStochLinreg(max_data_size=1000, max_features=100), stddev=1e-1),
        'noisy_2_correct_stoch_logreg': transformers.NormalNoisyGrad(CorrectStochLogreg(max_data_size=1000, max_features=100), stddev=1e-2),
        'noisy_2_correct_stoch_linear': transformers.NormalNoisyGrad(CorrectStochLinreg(max_data_size=1000, max_features=100), stddev=1e-2),
        'noisy_3_correct_stoch_logreg': transformers.NormalNoisyGrad(CorrectStochLogreg(max_data_size=1000, max_features=100), stddev=1e-3),
        'noisy_3_correct_stoch_linear': transformers.NormalNoisyGrad(CorrectStochLinreg(max_data_size=1000, max_features=100), stddev=1e-3),
        
        'noisy_stoch_logreg': transformers.NormalNoisyGrad(StochasticLogisticRegression(max_data_size=1000, max_features=100), stddev=1e-3),
        'noisy_stoch_linear': transformers.NormalNoisyGrad(StochasticLinearRegression(max_data_size=1000, max_features=100), stddev=1e-3),
        
        'stoch_logreg_10': StochasticLogisticRegression(max_data_size=400, max_features=10),
        
        '_digits_classifier': MLPClassifier(num_units=100, num_layers=1, dataset_name='digits', return_func=True),
        'lstm_ptb': LSTM_PTB(num_layers=1, hidden_size=50, batch_size=1, vocab_size=3000),
        
        'digits_classifier_0': MLPClassifier(num_units=100, num_layers=0, dataset_name='digits'),
        '_digits_classifier_0': MLPClassifier(num_units=100, num_layers=0, dataset_name='digits', return_func=True),
        'digits_classifier_0_random': MLPClassifier(num_units=100, num_layers=0, dataset_name='random'),
        '_digits_classifier_0_random': MLPClassifier(num_units=100, num_layers=0, dataset_name='random'),
        
        'digits_classifier': MLPClassifier(num_units=100, num_layers=1, dataset_name='digits'),
        'digits_classifier_2': MLPClassifier(num_units=100, num_layers=2, dataset_name='digits'),
        'digits_classifier_3': MLPClassifier(num_units=100, num_layers=3, dataset_name='digits'),
        'digits_classifier_6': MLPClassifier(num_units=100, num_layers=6, dataset_name='digits'),
        'digits_classifier_12': MLPClassifier(num_units=100, num_layers=12, dataset_name='digits'),
        'digits_classifier_18': MLPClassifier(num_units=100, num_layers=18, dataset_name='digits'),

        '_mnist_classifier': MLPClassifier(num_units=100, num_layers=1, dataset_name='mnist', return_func=True),
        'mnist_classifier': MLPClassifier(num_units=100, num_layers=1, dataset_name='mnist'),
        'mnist_classifier_3': MLPClassifier(num_units=100, num_layers=3, dataset_name='mnist'),
        'mnist_classifier_6': MLPClassifier(num_units=100, num_layers=6, dataset_name='mnist'),
        'mnist_classifier_12': MLPClassifier(num_units=100, num_layers=12, dataset_name='mnist'),
        'mnist_classifier_18': MLPClassifier(num_units=100, num_layers=18, dataset_name='mnist'),
        
        'mnist_classifier_relu': MLPClassifier(num_units=100, num_layers=1, dataset_name='mnist', activation='relu'),
        'mnist_classifier_relu_3': MLPClassifier(num_units=100, num_layers=3, dataset_name='mnist', activation='relu'),
        'mnist_classifier_relu_6': MLPClassifier(num_units=100, num_layers=6, dataset_name='mnist', activation='relu'),
        'mnist_classifier_relu_12': MLPClassifier(num_units=100, num_layers=12, dataset_name='mnist', activation='relu'),
        'mnist_classifier_relu_18': MLPClassifier(num_units=100, num_layers=18, dataset_name='mnist', activation='relu'),
        
        'digits_classifier_relu': MLPClassifier(num_units=100, num_layers=1, dataset_name='digits', activation='relu'),
        'digits_classifier_relu_2': MLPClassifier(num_units=100, num_layers=2, dataset_name='digits', activation='relu'),
        'digits_classifier_relu_3': MLPClassifier(num_units=100, num_layers=3, dataset_name='digits', activation='relu'),
        'digits_classifier_relu_6': MLPClassifier(num_units=100, num_layers=6, dataset_name='digits', activation='relu'),
        'digits_classifier_relu_12': MLPClassifier(num_units=100, num_layers=12, dataset_name='digits', activation='relu'),
        'digits_classifier_relu_18': MLPClassifier(num_units=100, num_layers=18, dataset_name='digits', activation='relu'),
        
        'conv_digits_classifier': ConvClassifier(num_filters=100, num_layers=1, dataset_name='digits'),
        'conv_digits_classifier_2': ConvClassifier(num_filters=100, num_layers=2, dataset_name='digits'),
        'vgg-cifar-10': ConvClassifier(dataset_name='cifar-10', arch='vgg19'),

        'conv_mnist_classifier': ConvClassifier(num_filters=100, num_layers=1, dataset_name='mnist'),
        'conv_mnist_classifier_3': ConvClassifier(num_filters=100, num_layers=3, dataset_name='mnist'),
        'conv_mnist_classifier_6': ConvClassifier(num_filters=100, num_layers=6, dataset_name='mnist'),
        'conv_mnist_classifier_12': ConvClassifier(num_filters=100, num_layers=12, dataset_name='mnist'),
        'conv_mnist_classifier_18': ConvClassifier(num_filters=100, num_layers=18, dataset_name='mnist'),
    }


    optimizees.update({
        '_mnist-nn-sigmoid-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='sigmoid'), reshape_f=True),
        '_mnist-nn-relu-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='relu'), reshape_f=True),
        '_mnist-nn-elu-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='elu'), reshape_f=True),

        'mnist-nn-sigmoid-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='sigmoid')),
        'mnist-nn-relu-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='relu')),
        'mnist-nn-elu-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='elu')),
        'mnist-nn-tanh-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='tanh')),

        'mnist-nn-l2-sigmoid-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='sigmoid', n_l=2)),
        'mnist-nn-l2-relu-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='relu', n_l=2)),
        'mnist-nn-l2-elu-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='elu', n_l=2)),
        'mnist-nn-l2-tanh-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='tanh', n_l=2)),

        'mnist-nn-l2-sigmoid-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='sigmoid', n_l=2)),
        'mnist-nn-l3-sigmoid-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='sigmoid', n_l=3)),
        'mnist-nn-l4-sigmoid-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='sigmoid', n_l=4)),
        'mnist-nn-l5-sigmoid-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='sigmoid', n_l=5)),
        'mnist-nn-l6-sigmoid-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='sigmoid', n_l=6)),
        'mnist-nn-l7-sigmoid-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='sigmoid', n_l=7)),
        'mnist-nn-l8-sigmoid-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='sigmoid', n_l=8)),
        'mnist-nn-l9-sigmoid-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='sigmoid', n_l=9)),
        'mnist-nn-l10-sigmoid-100': RNNPropAdapter(from_rnnprop.mnist.MnistLinearModel(activation='sigmoid', n_l=10)),
        'vgg-mnist-fc1-conv2-pool1-100': RNNPropAdapter(from_rnnprop.vgg.VGGModel(input_data='mnist', n_batches=128, fc_num=1, conv_num=2, pool_num=1)),
        'vgg-cifar-fc1-conv2-pool1-100': RNNPropAdapter(from_rnnprop.vgg.VGGModel(input_data='cifar10', n_batches=128, fc_num=1, conv_num=2, pool_num=1)),
        'vgg-mnist-fc2-conv4-pool2-100': RNNPropAdapter(from_rnnprop.vgg.VGGModel(input_data='mnist', n_batches=128, fc_num=2, conv_num=4, pool_num=2)),
        'vgg-cifar-fc2-conv4-pool2-100': RNNPropAdapter(from_rnnprop.vgg.VGGModel(input_data='cifar10', n_batches=128, fc_num=2, conv_num=4, pool_num=2)),
        'sin_lstm': RNNPropAdapter(from_rnnprop.lstm.SinLSTMModel()),
        'sin_lstm-x2': RNNPropAdapter(from_rnnprop.lstm.SinLSTMModel(n_lstm=2)),
        'sin_lstm-no001': RNNPropAdapter(from_rnnprop.lstm.SinLSTMModel(noise_scale=0.01)),
    })
    

    optimizees['mixed'] = transformers.ConcatAndSum([
        optimizees['quadratic'],
        optimizees['rosenbrock'],
    ])

    optimizees['mixed_nonstoch'] = transformers.ConcatAndSum([
        optimizees['quadratic'],
        optimizees['rosenbrock'],
        optimizees['beale'],
        optimizees['matyas'],
        optimizees['booth'],
        optimizees['logreg']
    ])
    
    optimizees['mixed_stoch'] = transformers.ConcatAndSum([
        optimizees['quadratic'],
        optimizees['rosenbrock'],
        optimizees['stoch_logreg'],
        optimizees['stoch_linear'],
    ])
    
    optimizees['correct_mixed_stoch'] = transformers.ConcatAndSum([
        optimizees['quadratic'],
        optimizees['rosenbrock'],
        optimizees['correct_stoch_logreg'],
        optimizees['correct_stoch_linear'],
    ])

    optimizees['stoch_only'] = transformers.ConcatAndSum([
        optimizees['stoch_logreg'],
        optimizees['stoch_linear'],
    ])

    for name in optimizees:
        opt = optimizees[name]

        if random_scale:
            opt = transformers.UniformRandomScaling(opt, r=3.0)

        if clip_by_value:
            opt = transformers.ClipByValue(opt, clip_low=0, clip_high=10**10)

        if noisy_grad: # and not name.startswith('stoch'):
            opt = transformers.NormalNoisyGrad(opt, stddev=0.01)

        optimizees[name] = opt

    if 'all' != problems_list and 'all' not in problems_list:
        return {problem: optimizees[problem] for problem in problems_list}
    else:
        return optimizees
