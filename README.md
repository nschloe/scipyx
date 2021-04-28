# scipyx

[![PyPi Version](https://img.shields.io/pypi/v/scipyx.svg?style=flat-square)](https://pypi.org/project/scipyx/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/scipyx.svg?style=flat-square)](https://pypi.org/project/scipyx/)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/scipyx.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/scipyx)
[![PyPi downloads](https://img.shields.io/pypi/dm/scipyx.svg?style=flat-square)](https://pypistats.org/packages/scipyx)

[![gh-actions](https://img.shields.io/github/workflow/status/nschloe/scipyx/ci?style=flat-square)](https://github.com/nschloe/scipyx/actions?query=workflow%3Aci)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/scipyx.svg?style=flat-square)](https://app.codecov.io/gh/nschloe/scipyx)
[![LGTM](https://img.shields.io/lgtm/grade/python/github/nschloe/scipyx.svg?style=flat-square)](https://lgtm.com/projects/g/nschloe/scipyx)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

[NumPy](https://numpy.org/) and [SciPy](https://www.scipy.org/) are large libraries used
everywhere in scientific computing. That's why breaking backwards-compatibility comes as
a significant cost and is almost always avoided, even if the API of some methods is
arguably lacking. This package provides drop-in wrappers "fixing" those.

If you have a fix for a NumPy method that can't go upstream for some reason, feel free
to PR here.


#### `dot`
```python
scipyx.dot(a, b)
```
Forms the dot product between the last axis of `a` and the _first_ axis of `b`.

(Not the second-last axis of `b` as `numpy.dot(a, b)`.)


#### `np.solve`
```python
scipyx.solve(A, b)
```
Solves a linear equation system with a matrix of shape `(n, n)` and an array of shape
`(n, ...)`. The output has the same shape as the second argument.


#### `sum_at`/`add_at`
```python
scipyx.sum_at(a, idx, minlength=0)
scipyx.add_at(out, idx, a)
```
Returns an array with entries of `a` summed up at indices `idx` with a minumum length of
`minlength`. `idx` can have any shape as long as it's matching `a`. The output shape is
`(minlength,...)`.

The numpy equivalent `numpy.add.at` is _much_
slower:

<img alt="memory usage" src="https://nschloe.github.io/scipyx/perf-add-at.svg" width="50%">

[Corresponding issue report](https://github.com/numpy/numpy/issues/11156)


#### `unique_rows`
```python
scipyx.unique_rows(a, return_inverse=False, return_counts=False)
```
Returns the unique rows of the integer array `a`. The numpy alternative `np.unique(a,
axis=0)` is slow.

[Corresponding issue report](https://github.com/numpy/numpy/issues/11136)


#### `isin_rows`
```python
scipyx.isin_rows(a, b)
```
Returns a boolean array of length `len(a)` specifying if the rows `a[k]` appear in `b`.
Similar to NumPy's own `np.isin` which only works for scalars.


#### SciPy Krylov methods
```python
sol, info = scipyx.cg(A, b, tol=1.0e-10)
sol, info = scipyx.minres(A, b, tol=1.0e-10)
sol, info = scipyx.gmres(A, b, tol=1.0e-10)
sol, info = scipyx.bicg(A, b, tol=1.0e-10)
sol, info = scipyx.bicgstab(A, b, tol=1.0e-10)
sol, info = scipyx.cgs(A, b, tol=1.0e-10)
sol, info = scipyx.qmr(A, b, tol=1.0e-10)
```
`sol` is the solution of the linear system `A @ x = b` (or `None` if no convergence),
and `info` contains some useful data, e.g., `info.resnorms`. The methods are wrappers
around [SciPy's iterative
solvers](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html).

Relevant issues:
 * [inconsistent number of callback calls between cg, minres](https://github.com/scipy/scipy/issues/13936)


#### SciPy minimization
```python
def f(x):
    return (x ** 2 - 2) ** 2


x0 = 1.5
out = scipyx.minimize(f, x0)
```
In SciPy, the result from a minimization `out.x` will _always_ have shape `(n,)`, no
matter the input vector. scipyx changes this to respect the input vector shape.

[Corresponding issue report](https://github.com/scipy/scipy/issues/13869)


### License
scipyx is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
