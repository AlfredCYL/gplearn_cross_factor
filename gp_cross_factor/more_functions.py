import copy

import numpy as np
import pandas as pd

from functions import _Function

def error_state_decorator(func):
    def wrapper(A, *args, **kwargs):
        with np.errstate(over='ignore', under='ignore'):
            return func(A, *args, **kwargs)
    return wrapper

@error_state_decorator
def scale(A, scaler=1):
    ret = pd.DataFrame(A)
    factor = scaler * ret.div(ret.sum(axis=1), axis=0).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rank(A):
    ret = pd.DataFrame(A)
    factor = ret.rank(axis=1).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def delay(A, window=1):
    ret = pd.DataFrame(A)
    factor = ret.shift(window).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def delta(A, window=1):
    ret = pd.DataFrame(A)
    factor = ret.diff(window).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_nanmean(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window,min_periods=2).mean().to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_nanstd(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window,min_periods=2).std().to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_max(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window,min_periods=2).max()
    factor = factor.to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_min(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window,min_periods=2).min()
    factor = factor.to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_correlation(A, B, window=5):
    ret1 = pd.DataFrame(A)
    ret2 = pd.DataFrame(B)
    factor = ret1.rolling(window,min_periods=2).corr(ret2).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_argmin(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window,min_periods=2).apply(lambda x: np.argmin(x), raw=True).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_argmax(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window,min_periods=2).apply(lambda x: np.argmax(x), raw=True).to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def rolling_rank(A, window=5):
    ret = pd.DataFrame(A)
    factor = ret.rolling(window,min_periods=2).rank().to_numpy(dtype=np.double)
    return factor

@error_state_decorator
def pow(A, pow=2):
    return np.power(A,pow)

_extra_function_map = {
    'ts_std_5': _Function(function=lambda x: rolling_nanstd(x, 5), name='ts_std_5', arity=1),
    'ts_mean_5': _Function(function=lambda x: rolling_nanmean(x, 5), name='ts_mean_5', arity=1),
    'ts_max_5': _Function(function=lambda x: rolling_max(x, 5), name='ts_max_5', arity=1),
    'ts_min_5': _Function(function=lambda x: rolling_min(x, 5), name='ts_min_5', arity=1),
    'ts_correlation_5': _Function(function=lambda x, y: rolling_correlation(x, y, 5), name='ts_correlation_5', arity=2),
    'ts_argmin_5': _Function(function=lambda x: rolling_argmin(x, 5), name='ts_argmin_5', arity=1),
    'ts_argmax_5': _Function(function=lambda x: rolling_argmax(x, 5), name='ts_argmax_5', arity=1),
    'ts_rank_5': _Function(function=lambda x: rolling_rank(x, 5), name='ts_rank_5', arity=1),

    'ts_std_10': _Function(function=lambda x: rolling_nanstd(x, 10), name='ts_std_10', arity=1),
    'ts_mean_10': _Function(function=lambda x: rolling_nanmean(x, 10), name='ts_mean_10', arity=1),
    'ts_max_10': _Function(function=lambda x: rolling_max(x, 10), name='ts_max_10', arity=1),
    'ts_min_10': _Function(function=lambda x: rolling_min(x, 10), name='ts_min_10', arity=1),
    'ts_correlation_10': _Function(function=lambda x, y: rolling_correlation(x, y, 10), name='ts_correlation_10', arity=2),
    'ts_argmin_10': _Function(function=lambda x: rolling_argmin(x, 10), name='ts_argmin_10', arity=1),
    'ts_argmax_10': _Function(function=lambda x: rolling_argmax(x, 10), name='ts_argmax_10', arity=1),
    'ts_rank_10': _Function(function=lambda x: rolling_rank(x, 10), name='ts_rank_10', arity=1),

    'ts_std_20': _Function(function=lambda x: rolling_nanstd(x, 20), name='ts_std_20', arity=1),
    'ts_mean_20': _Function(function=lambda x: rolling_nanmean(x, 20), name='ts_mean_20', arity=1),
    'ts_max_20': _Function(function=lambda x: rolling_max(x, 20), name='ts_max_20', arity=1),
    'ts_min_20': _Function(function=lambda x: rolling_min(x, 20), name='ts_min_20', arity=1),
    'ts_correlation_20': _Function(function=lambda x, y: rolling_correlation(x, y, 20), name='ts_correlation_20', arity=2),
    'ts_argmin_20': _Function(function=lambda x: rolling_argmin(x, 20), name='ts_argmin_20', arity=1),
    'ts_argmax_20': _Function(function=lambda x: rolling_argmax(x, 20), name='ts_argmax_20', arity=1),
    'ts_rank_20': _Function(function=lambda x: rolling_rank(x, 20), name='ts_rank_20', arity=1),

    'ts_std_40': _Function(function=lambda x: rolling_nanstd(x, 40), name='ts_std_40', arity=1),
    'ts_mean_40': _Function(function=lambda x: rolling_nanmean(x, 40), name='ts_mean_40', arity=1),
    'ts_max_40': _Function(function=lambda x: rolling_max(x, 40), name='ts_max_40', arity=1),
    'ts_min_40': _Function(function=lambda x: rolling_min(x, 40), name='ts_min_40', arity=1),
    'ts_correlation_40': _Function(function=lambda x, y: rolling_correlation(x, y, 40), name='ts_correlation_40', arity=2),
    'ts_argmin_40': _Function(function=lambda x: rolling_argmin(x, 40), name='ts_argmin_40', arity=1),
    'ts_argmax_40': _Function(function=lambda x: rolling_argmax(x, 40), name='ts_argmax_40', arity=1),
    'ts_rank_40': _Function(function=lambda x: rolling_rank(x, 40), name='ts_rank_40', arity=1),
    
    'ts_std_60': _Function(function=lambda x: rolling_nanstd(x, 60), name='ts_std_60', arity=1),
    'ts_mean_60': _Function(function=lambda x: rolling_nanmean(x, 60), name='ts_mean_60', arity=1),
    'ts_max_60': _Function(function=lambda x: rolling_max(x, 60), name='ts_max_60', arity=1),
    'ts_min_60': _Function(function=lambda x: rolling_min(x, 60), name='ts_min_60', arity=1),
    'ts_correlation_60': _Function(function=lambda x, y: rolling_correlation(x, y, 60), name='ts_correlation_60', arity=2),
    'ts_argmin_60': _Function(function=lambda x: rolling_argmin(x, 60), name='ts_argmin_60', arity=1),
    'ts_argmax_60': _Function(function=lambda x: rolling_argmax(x, 60), name='ts_argmax_60', arity=1),
    'ts_rank_60': _Function(function=lambda x: rolling_rank(x, 60), name='ts_rank_60', arity=1),
    
    'delay_1': _Function(function=lambda x: delay(x, 1), name = 'delay_1', arity=1),
    'delay_2': _Function(function=lambda x: delay(x, 2), name = 'delay_2', arity=1),
    'delay_5': _Function(function=lambda x: delay(x, 5), name = 'delay_5', arity=1),
    'delay_10': _Function(function=lambda x: delay(x, 10), name = 'delay_10', arity=1),
    'delay_20': _Function(function=lambda x: delay(x, 20), name = 'delay_20',arity=1),
    'delay_40': _Function(function=lambda x: delay(x, 40), name = 'delay_40', arity=1),
    'delay_60': _Function(function=lambda x: delay(x, 60), name = 'delay_60', arity=1),
    
    'delta_1': _Function(function=lambda x: delta(x, 1), name = 'delta_1', arity=1),
    'delta_2': _Function(function=lambda x: delta(x, 2), name = 'delta_2', arity=1),
    'delta_5': _Function(function=lambda x: delta(x, 5), name = 'delta_5', arity=1),
    'delta_10': _Function(function=lambda x: delta(x, 10), name = 'delta_10', arity=1),
    'delta_20': _Function(function=lambda x: delta(x, 20), name = 'delta_20', arity=1),
    'delta_40': _Function(function=lambda x: delta(x, 40), name = 'delta_40', arity=1),
    'delta_60': _Function(function=lambda x: delta(x, 40), name = 'delta_60', arity=1),
    
    'rank': _Function(function=lambda x: rank(x), name = 'rank', arity=1),
    'scale_1': _Function(function=lambda x: scale(x, 1), name = 'scale_1', arity=1),
    'power_2': _Function(function=lambda x: pow(x, 2), name = 'power_2', arity=1),
    'power_3': _Function(function=lambda x: pow(x, 3), name = 'power_3', arity=1),
}