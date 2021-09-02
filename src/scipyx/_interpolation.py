import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PPoly, lagrange


def interp_rolling_lagrange(x: ArrayLike, y: ArrayLike, order: int) -> PPoly:
    x = np.asarray(x)
    y = np.asarray(y)

    # make sure x is sorted
    assert np.all(x[:-1] < x[1:])
    assert len(x) > order

    if order % 2 == 1:
        # The intervals are between the points
        coeffs = []
        for k in range(len(x) - 1):
            idx = np.arange(k - order // 2, k + order // 2 + 2)
            while idx[0] < 0:
                idx += 1
            while idx[-1] > len(x) - 1:
                idx -= 1

            lag = lagrange(x[idx] - x[k], y[idx])
            c = lag.coefficients
            if len(c) < order + 1:
                # Prepend zeros if necessary; see
                # <https://github.com/scipy/scipy/issues/14681>
                c = np.concatenate([np.zeros(order + 1 - len(c)), c])

            coeffs.append(c)

        pp = PPoly(np.array(coeffs).T, x)

    else:
        # The intervals are around the points
        breakpoints = np.concatenate([[x[0]], (x[:-1] + x[1:]) / 2, [x[-1]]])

        coeffs = []
        for k in range(len(x)):
            idx = np.arange(k - order // 2, k + order // 2 + 1)
            while idx[0] < 0:
                idx += 1
            while idx[-1] > len(x) - 1:
                idx -= 1

            lag = lagrange(x[idx] - breakpoints[k], y[idx])
            c = lag.coefficients
            if len(c) < order + 1:
                # Prepend zeros if necessary; see
                # <https://github.com/scipy/scipy/issues/14681>
                c = np.concatenate([np.zeros(order + 1 - len(c)), c])
            coeffs.append(c)

        pp = PPoly(np.array(coeffs).T, breakpoints)

    return pp
