from .quadratic import Quadratic
from .rosenbrock import Rosenbrock
from .beale import Beale
from .booth import Booth
from .matyas import Matyas
from .logistic_regression import LogisticRegression
from .stochastic_logistic_regression import StochasticLogisticRegression
from .stochastic_linear_regression import StochasticLinearRegression

from . import transformers


def get_optimizees(clip_by_value=False, random_scale=False, noisy_grad=False):
    optimizees = {
        'quadratic'   : Quadratic(low=50, high=100),
        'rosenbrock'  : Rosenbrock(low=2, high=10),
        'beale'       : Beale(low=2, high=10),
        'booth'       : Booth(low=2, high=10),
        'matyas'      : transformers.UniformRandomScaling(Matyas(low=2, high=10), r=3.0),
        'logreg'      : LogisticRegression(max_data_size=1000, max_features=100),
        'stoch_logreg': StochasticLogisticRegression(max_data_size=1000, max_features=100),
        'stoch_linear': StochasticLinearRegression(max_data_size=1000, max_features=100)
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

    for name in optimizees:
        opt = optimizees[name]

        if random_scale:
            opt = transformers.UniformRandomScaling(opt, r=3.0)

        if clip_by_value:
            opt = transformers.ClipByValue(opt, clip_low=0, clip_high=10**10)

        if noisy_grad and not name.startswith('stoch'):
            opt = transformers.NormalNoisyGrad(opt, stddev=0.1)

        optimizees[name] = opt

    return optimizees
