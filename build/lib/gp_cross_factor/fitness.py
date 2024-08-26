"""
Metrics to evaluate the fitness of a program.

The :mod:`gplearn.fitness` module contains some metric with which to evaluate
the computer programs created by the :mod:`gplearn.genetic` module.
"""

import numbers
import numpy as np
import pandas as pd
from joblib import wrap_non_picklable_objects
from scipy.stats import rankdata

__all__ = ['make_fitness']


class _Fitness(object):

    """A metric to measure the fitness of a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting floating point score quantifying the quality of the program's
    representation of the true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """

    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args):
        return self.function(*args)


def make_fitness(*, function, greater_is_better, wrap=True):
    """Make a fitness measure, a metric scoring the quality of a program's fit.

    This factory function creates a fitness measure object which measures the
    quality of a program's fit and thus its likelihood to undergo genetic
    operations into the next generation. The resulting object is able to be
    called with NumPy vectorized arguments and return a resulting floating
    point score quantifying the quality of the program's representation of the
    true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom metrics is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'
                         % type(greater_is_better))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))
    if function.__code__.co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w),'
                         ' got %d.' % function.__code__.co_argcount)
    if not isinstance(function(np.array([1, 1]),
                      np.array([2, 2]),
                      np.array([1, 1])), numbers.Number):
        raise ValueError('function must return a numeric.')

    if wrap:
        return _Fitness(function=wrap_non_picklable_objects(function),
                        greater_is_better=greater_is_better)
    return _Fitness(function=function,
                    greater_is_better=greater_is_better)


def _weighted_pearson(y, y_pred, w):
    """Calculate the weighted Pearson correlation coefficient."""
    with np.errstate(divide='ignore', invalid='ignore'):
        y_pred_demean = y_pred - np.average(y_pred, weights=w)
        y_demean = y - np.average(y, weights=w)
        corr = ((np.sum(w * y_pred_demean * y_demean) / np.sum(w)) /
                np.sqrt((np.sum(w * y_pred_demean ** 2) *
                         np.sum(w * y_demean ** 2)) /
                        (np.sum(w) ** 2)))
    if np.isfinite(corr):
        return np.abs(corr)
    return 0.


def _weighted_spearman(y, y_pred, w):
    """Calculate the weighted Spearman correlation coefficient."""
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y)
    return _weighted_pearson(y_pred_ranked, y_ranked, w)


def _mean_absolute_error(y, y_pred, w):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=w)


def _mean_square_error(y, y_pred, w):
    """Calculate the mean square error."""
    return np.average(((y_pred - y) ** 2), weights=w)


def _root_mean_square_error(y, y_pred, w):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))


def _log_loss(y, y_pred, w):
    """Calculate the log loss."""
    eps = 1e-15
    inv_y_pred = np.clip(1 - y_pred, eps, 1 - eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    score = y * np.log(y_pred) + (1 - y) * np.log(inv_y_pred)
    return np.average(-score, weights=w)

weighted_pearson = _Fitness(function=_weighted_pearson,
                            greater_is_better=True)
weighted_spearman = _Fitness(function=_weighted_spearman,
                             greater_is_better=True)
mean_absolute_error = _Fitness(function=_mean_absolute_error,
                               greater_is_better=False)
mean_square_error = _Fitness(function=_mean_square_error,
                             greater_is_better=False)
root_mean_square_error = _Fitness(function=_root_mean_square_error,
                                  greater_is_better=False)
log_loss = _Fitness(function=_log_loss,
                    greater_is_better=False)

_fitness_map = {'pearson': weighted_pearson,
                'spearman': weighted_spearman,
                'mean absolute error': mean_absolute_error,
                'mse': mean_square_error,
                'rmse': root_mean_square_error,
                'log loss': log_loss
}

def calculate_nan_rate_2d(array): # 计算空值率 
    total_elements = np.size(array)  
    nan_elements = np.count_nonzero(np.isnan(array))  
    
    nan_rate = nan_elements / total_elements 
    return nan_rate

def compute_IC(y, y_pred, w, rank_ic=True):
    y_pred = y_pred.copy()
    y_pred[np.isinf(y_pred)] = np.nan
    if calculate_nan_rate_2d(y) > 0.3: # 保证空值率小于0.3
        return 

    y = y[w.astype(bool)]
    y_pred = y_pred[w.astype(bool)]
    if rank_ic:
        ic = pd.DataFrame(y_pred).corrwith(pd.DataFrame(y),axis = 1, method = "spearman")
    else:
        ic = pd.DataFrame(y_pred).corrwith(pd.DataFrame(y),axis = 1, method = "pearson")
    return ic 

def _rank_IC(y, y_pred, w):
    ics = compute_IC(y, y_pred, w)
    if ics is None:
        return 0
        
    ic = ics.mean()
    if np.isnan(ic):
        return 0
    else:
        return abs(ic)

def _rank_ICIR(y, y_pred, w):
    ics = compute_IC(y, y_pred, w)
    if ics is None:
        return 0
        
    ic = ics.mean()
    ic_std = ics.std()
    icir = ic / ic_std
    if np.isnan(icir):
        return 0
    else:
        return abs(icir)
    
def compute_quantile10_rets(y, y_pred, w):
    y_pred = y_pred[w.astype(bool)]
    y = y[w.astype(bool)]
    if np.all(np.isnan(y_pred)):
        return None
    
    quantiles = 10
    annulization = 252 #默认按日频年化收益
    groups = np.array(range(quantiles)) + 1

    factor_quantiles = pd.DataFrame(y_pred).rank(axis=1,method='first').dropna(axis=0, how='all').apply(pd.qcut, q=quantiles, labels = groups,axis=1)

    rets = pd.DataFrame(y)
    return_series = {}
    for group in groups:
        returns_group = rets[factor_quantiles == group]
        return_series[group] = (returns_group.sum(axis=1) / returns_group.count(axis=1)).mean() * annulization # scale holding to 1 ; equal weights
    return return_series

def _quantile10_max(y, y_pred, w):
    res = compute_quantile10_rets(y, y_pred, w)
    if res is None:
        return 0
    else:
        return max(res.values())
    
def measure_monotonicity(data):
    ranks = [sorted(data).index(x) + 1 for x in data]
    rank_differences = [ranks[i] - ranks[i-1] for i in range(1, len(ranks))]
    positive_differences = sum(1 for diff in rank_differences if diff > 0)
    negative_differences = sum(1 for diff in rank_differences if diff < 0)
    monotonicity_score = abs(positive_differences - negative_differences) / len(data)
    return monotonicity_score

def _quantile10_monotonicity(y, y_pred, w):
    res = compute_quantile10_rets(y, y_pred, w)
    if res is None:
        return 0
    else:
        return measure_monotonicity(res.values())
    
weighted_rank_ic = _Fitness(function=_rank_IC,greater_is_better=True)
weighted_rank_icir = _Fitness(function=_rank_ICIR,greater_is_better=True)
weighted_quantile_max = _Fitness(function=_quantile10_max,greater_is_better=True)
weighted_quantile_mono = _Fitness(function=_quantile10_monotonicity,greater_is_better=True)

_extra_map = {
    "rank_ic":weighted_rank_ic,
    "rank_icir":weighted_rank_icir,
    "quantile_max":weighted_quantile_max,
    "quantile_mono":weighted_quantile_mono,
}

_fitness_map = dict(_fitness_map, **_extra_map)


