import numpy as np
import pytest

import scipyx as spx


def test_0d():
    def f(x):
        return (x ** 2 - 2) ** 2

    x0 = 1.5
    out = spx.minimize(f, x0)

    assert out.x.shape == np.asarray(x0).shape
    assert np.asarray(out.fun).shape == ()


def test_2d():
    def f(x):
        return (np.sum(x ** 2) - 2) ** 2

    x0 = np.ones((4, 3), dtype=float)
    out = spx.minimize(f, x0, method="Powell")

    assert out.x.shape == np.asarray(x0).shape
    assert np.asarray(out.fun).shape == ()


def test_error():
    def f(x):
        return x - 2

    x0 = [1.5]
    with pytest.raises(ValueError):
        spx.minimize(f, x0)


def test_0d_leastsq():
    def f(x):
        assert x.shape == ()
        return x

    x0 = 0.0
    x, _ = spx.leastsq(f, x0)
    assert x.shape == ()


def test_1d_leastsq():
    def f(x):
        assert x.shape == (2,)
        print(x.shape)
        return x

    x0 = [1.0, 0.0]
    x, _ = spx.leastsq(f, x0)
    assert x.shape == (2,)
