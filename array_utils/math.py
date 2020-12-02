import numpy as np
from numba import jit


@jit(nopython=True, parallel=True)
def normalized_ratio(a, b):
    return (a - b) / (a + b)


def rescale(arr, min_val, max_val):
    arr += -(np.min(arr))
    arr /= np.max(arr) / (max_val - min_val)
    arr += min_val
    return arr