from typing import Callable

import numpy as np
import scipy.optimize
from numpy.typing import ArrayLike


def minimize(fun: Callable, x0: ArrayLike, *args, **kwargs) -> np.ndarray:
    x0 = np.asarray(x0)
    x0_shape = x0.shape

    def fwrap(x):
        x = x.reshape(x0_shape)
        val = fun(x)
        val_shape = np.asarray(val).shape
        if val_shape != ():
            raise ValueError(
                "Objective function must return a true scalar. "
                f"Got array of shape {val_shape}."
            )
        return val

    out = scipy.optimize.minimize(fwrap, x0, *args, **kwargs)
    out.x = out.x.reshape(x0.shape)
    return out


def leastsq(fun: Callable, x0: ArrayLike, *args, **kwargs):
    x0 = np.asarray(x0)
    x0_shape = x0.shape

    def fwrap(x):
        x = x.reshape(x0_shape)
        return fun(x)

    out = scipy.optimize.leastsq(fwrap, x0, *args, **kwargs)
    return (out[0].reshape(x0.shape), *out[1:])
