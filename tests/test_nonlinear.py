import numpy as np

import scipyx


def test_bisect():
    def f(x):
        return x ** 2 - 2

    tol = 1.0e-12
    a, b = scipyx.bisect(f, 0.0, 5.0, tol)
    assert b - a < tol
    assert abs(np.sqrt(2) - a) < 2 * tol


def test_regula_falsi():
    def f(x):
        return x ** 2 - 2

    tol = 1.0e-12
    a, b = scipyx.regula_falsi(f, 0.0, 5.0, tol)
    assert b - a < tol
    assert abs(np.sqrt(2) - a) < 2 * tol
