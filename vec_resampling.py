"""
Performs downsampling of a vector, using various techniques
"""

# TODO: candidate for removal, as scipy.interpolate is a better substitute

import math
from typing import List
from enum import Enum
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import torch
from torch import nn

def vec_downsample(vec: np.array, divisor: int, mode: str = 'linear') -> np.ndarray:
    """Downsample an ndarray by an integer ratio
    
    mode may be: 'none', 'nearest', 'linear', 'quadratic', 'cubic', 'max-pool',
    'min-pool', 'abs-max-pool', 'abs-min-pool', 'mean-pool', 'median-pool'
    """

    xold = lambda v: np.linspace(0, len(v), len(v))
    xnew = lambda v: np.linspace(0, len(v), len(v) // divisor)
    intp = lambda v, m: interp1d(xold(v), v, kind=m)(xnew(v))
    split = lambda v, n: [v[i:min(i+n, len(v))] for i in range(0, len(v), n)]
    
    
    if mode == 'none':
        return intp(vec, 'previous')

    if mode == 'nearest':
        return intp(vec, 'nearest')
    
    if mode == 'linear':
        return intp(vec, 'linear')

    if mode == 'quadratic':
        return intp(vec, 'quadratic')

    if mode == 'cubic':
        return intp(vec, 'cubic')

    if mode == 'max-pool':
        return np.array(list(map(max, split(vec, divisor))))
    
    if mode == 'min-pool':
        return np.array(list(map(min, split(vec, divisor))))
    
    if mode == 'abs-max-pool':
        def abs_max(v):
            running_max = 0
            running_max_sign = 1
            for item in v:
                if np.abs(item) > running_max:
                    running_max = abs(item)
                    running_max_sign = np.sign(item)
            return running_max * running_max_sign

        return np.array(list(map(abs_max, split(vec, divisor))))
    
    if mode == 'abs-min-pool':
        def abs_min(v):
            running_min = float('inf')
            running_min_sign = 1
            for item in v:
                if np.abs(item) < running_min:
                    running_min = abs(item)
                    running_min_sign = np.sign(item)
            return running_min * running_min_sign

        return np.array(list(map(abs_min, split(vec, divisor))))
    
    if mode == 'mean-pool':
        return np.array(list(map(np.mean, split(vec, divisor))))

    if mode == 'median-pool':
        return np.array(list(map(np.median, split(vec, divisor))))
    
    return None
    

def vec_upsample(vec: np.array, multiplier: int, mode: str = 'linear') -> np.ndarray:
    """Usample an ndarray by an integer ratio
    
    mode may be: 'none', 'nearest', 'linear', 'quadratic', 'cubic'
    """
    xold = lambda v: np.linspace(0, len(v), len(v))
    xnew = lambda v: np.linspace(0, len(v), len(v) * multiplier)
    intp = lambda v, m: interp1d(xold(v), v, kind=m)(xnew(v))

    if mode == 'none':
        return intp(vec, 'previous')

    if mode == 'nearest':
        return intp(vec, 'nearest')
    
    if mode == 'linear':
        return intp(vec, 'linear')

    if mode == 'quadratic':
        return intp(vec, 'quadratic')

    if mode == 'cubic':
        return intp(vec, 'cubic')
    


# testing
if __name__ == '__main__':
    x = np.linspace(0, 10, 1000)
    y = np.sin((0.5*x)**2)
    ynew = vec_downsample(y, 100, 'median-pool')
    ynewnew = vec_upsample(ynew, 100, 'none')
    xnew = np.linspace(0, 10, len(ynew))
    plt.plot(x,y)
    plt.plot(xnew,ynew)
    plt.plot(x,ynewnew)
    plt.show()