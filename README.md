# scipyx

[![PyPi Version](https://img.shields.io/pypi/v/scipyx.svg?style=flat-square)](https://pypi.org/project/scipyx/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/scipyx.svg?style=flat-square)](https://pypi.org/project/scipyx/)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/scipyx.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/scipyx)
[![PyPi downloads](https://img.shields.io/pypi/dm/scipyx.svg?style=flat-square)](https://pypistats.org/packages/scipyx)

[![gh-actions](https://img.shields.io/github/workflow/status/nschloe/scipyx/ci?style=flat-square)](https://github.com/nschloe/scipyx/actions?query=workflow%3Aci)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/scipyx.svg?style=flat-square)](https://app.codecov.io/gh/nschloe/scipyx)
[![LGTM](https://img.shields.io/lgtm/grade/python/github/nschloe/scipyx.svg?style=flat-square)](https://lgtm.com/projects/g/nschloe/scipyx)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

[SciPy](https://www.scipy.org/) is large library used everywhere in scientific
computing. That's why breaking backwards-compatibility comes as a significant cost and
is almost always avoided, even if the API of some methods is arguably lacking. This
package provides drop-in wrappers "fixing" those.

[npx](https://github.com/nschloe/npx) does the same for [NumPy](https://numpy.org/).

If you have a fix for a SciPy method that can't go upstream for some reason, feel free
to PR here.

#### Krylov methods

```python
import numpy as np
import scipy.sparse
import scipyx as spx

# create tridiagonal (-1, 2, -1) matrix
n = 100
data = -np.ones((3, n))
data[1] = 2.0
A = scipy.sparse.spdiags(data, [-1, 0, 1], n, n)
A = A.tocsr()
b = np.ones(n)


sol, info = spx.cg(A, b, tol=1.0e-10)
sol, info = spx.minres(A, b, tol=1.0e-10)
sol, info = spx.gmres(A, b, tol=1.0e-10)
sol, info = spx.bicg(A, b, tol=1.0e-10)
sol, info = spx.bicgstab(A, b, tol=1.0e-10)
sol, info = spx.cgs(A, b, tol=1.0e-10)
sol, info = spx.qmr(A, b, tol=1.0e-10)
```

`sol` is the solution of the linear system `A @ x = b` (or `None` if no convergence),
and `info` contains some useful data, e.g., `info.resnorms`. The solution `sol` and all
callback `x` have the shape of `x0`/`b`.
The methods are wrappers around [SciPy's iterative
solvers](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html).

Relevant issues:

- [inconsistent number of callback calls between cg, minres](https://github.com/scipy/scipy/issues/13936)

#### Optimization

```python
import scipyx as spx


def f(x):
    return (x ** 2 - 2) ** 2


x0 = 1.5
out = spx.minimize(f, x0)
print(out.x)

x0 = -3.2
x, _ = spx.leastsq(f, x0)
print(x)
```

In scipyx, all intermediate values `x` and the result from a minimization `out.x` will
have the same shape as `x0`. (In SciPy, they always have shape `(n,)`, no matter the
input vector.)

Relevant issues:

- [optimization: let out.x have the same shape as
  x0](https://github.com/scipy/scipy/issues/13869)


#### Rolling Lagrange interpolation

```python
import numpy as np
import scipyx as spx


x = np.linspace(0.0, 1.0, 11)
y = np.sin(7.0 * x)

poly = spx.interp_rolling_lagrange(x, y, order=3)
```

Given an array of coordinates `x` and an array of values `y`, you can use scipyx to
compute a piecewise polynomial Lagrange approximation. The `order + 1` closest
coordinates x/y are considered for each interval.

| <img src="https://nschloe.github.io/scipyx/interp-0.svg" width="100%"> | <img src="https://nschloe.github.io/scipyx/interp-1.svg" width="100%"> | <img src="https://nschloe.github.io/scipyx/interp-2.svg" width="100%"> |
| :--------------------------------------------------------------------: | :--------------------------------------------------------------------: | :--------------------------------------------------------------------: |
|                                Order 0                                 |                                Order 1                                 |                                Order 2                                 |

#### Jacobi elliptic functions with complex argument

SciPy supports
[Jacobi elliptic functions](https://en.wikipedia.org/wiki/Jacobi_elliptic_functions) as
[ellipj](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipj.html).
Unfortunately, only real-valued argument `u` and parameter `m` are allowed. scipyx
expands support to complex-valued argument `u`.

```python
import scipyx as spx

u = 1.0 + 2.0j
m = 0.8
# sn, cn, dn, ph = scipy.special.ellipj(x, m)  # not working
sn, cn, dn, ph = spx.ellipj(u, m)
```

Relevant bug reports:

- [Jacobian elliptic function with complex argument
  #12226](https://github.com/scipy/scipy/issues/12226)

### License

This software is published under the [BSD-3-Clause
license](https://spdx.org/licenses/BSD-3-Clause.html).
