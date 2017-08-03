from .quadratic import Quadratic
from .rosenbrock import Rosenbrock
from .beale import Beale
from .booth import Booth
from .matyas import Matyas
from .logistic_regression import LogisticRegression
from .stochastic_logistic_regression import StochasticLogisticRegression
from .stochastic_linear_regression import StochasticLinearRegression
from .digits_classifier import DIGITSClassifier
from .conv_classifier import ConvClassifier

from . import transformers

problems = [
    'quadratic', 'rosenbrock', 'logreg',
    'stoch_logreg', 'stoch_linear',
    'mixed', 'mixed_stoch', 'mixed_nonstoch',
    'digits_classifier', 'digits_classifier_2',
    'digits_classifier_relu', 'digits_classifier_relu_2',
    'mnist_classifier',
    'conv_digits_classifier', 'conv_digits_classifier_2',
    'beale', 'booth', 'matyas', 'stoch_only',
    'digits_classifier_3', 'digits_classifier_relu_3',
    'vgg-cifar-10'
]


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
        'digits_classifier': DIGITSClassifier(num_units=100, num_layers=1, dataset_name='digits'),
        'digits_classifier_2': DIGITSClassifier(num_units=100, num_layers=2, dataset_name='digits'),
        'digits_classifier_relu': DIGITSClassifier(num_units=100, num_layers=1, dataset_name='digits', activation='relu'),
        'digits_classifier_relu_2': DIGITSClassifier(num_units=100, num_layers=2, dataset_name='digits', activation='relu'),
        'mnist_classifier': DIGITSClassifier(num_units=100, num_layers=1, dataset_name='mnist'),
        'conv_digits_classifier': ConvClassifier(num_filters=100, num_layers=1, dataset_name='digits'),
        'conv_digits_classifier_2': ConvClassifier(num_filters=100, num_layers=2, dataset_name='digits'),
        'digits_classifier_3': DIGITSClassifier(num_units=100, num_layers=3, dataset_name='digits'),
        'digits_classifier_relu_3': DIGITSClassifier(num_units=100, num_layers=3, dataset_name='digits', activation='relu'),
        'vgg-cifar-10': ConvClassifier(dataset_name='cifar-10', arch='vgg19')
    }

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

        if noisy_grad and not name.startswith('stoch'):
            opt = transformers.NormalNoisyGrad(opt, stddev=0.1)

        optimizees[name] = opt

    if 'all' != problems_list and 'all' not in problems_list:
        return {problem: optimizees[problem] for problem in problems_list}
    else:
        return optimizees
