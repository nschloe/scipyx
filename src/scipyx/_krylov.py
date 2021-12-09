from __future__ import annotations

from collections import namedtuple
from typing import Callable

import numpy as np
import scipy
import scipy.sparse.linalg
from numpy.typing import ArrayLike

Info = namedtuple("KrylovInfo", ["success", "xk", "numsteps", "resnorms", "errnorms"])


def _norm(r: ArrayLike, M=None):
    Mr = r if M is None else M @ r

    if len(r.shape) == 1:
        rMr = np.dot(r.conj(), Mr)
    else:
        rMr = np.einsum("i...,i...->...", r.conj(), Mr)

    if np.any(rMr.imag) > 1.0e-13:
        raise RuntimeError(
            "Got nonnegative imaginary part. " "Is the preconditioner not self-adjoint?"
        )
    return np.sqrt(rMr.real)


def assert_shapes(A, b: np.ndarray, x0: np.ndarray):
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square matrix, not A.shape = {A.shape}.")
    if x0.shape != b.shape:
        raise ValueError(
            f"x0 and b need to have the same shape, not x0.shape = {x0.shape}, "
            f"b.shape = {b.shape}"
        )
    if b.shape[0] != b.size:
        raise ValueError("Can only deal with one right-hand side at a time.")
    assert A.shape[1] == b.shape[0]


def _wrapper(
    method: Callable,
    A,
    b: ArrayLike,
    x0: ArrayLike | None = None,
    tol: float = 1e-05,
    maxiter: int | None = None,
    M=None,
    callback: Callable | None = None,
    atol: float | None = 0.0,
    exact_solution: ArrayLike | None = None,
):
    if x0 is None:
        x0 = np.zeros_like(b)
    else:
        x0 = np.asarray(x0)

    assert_shapes(A, b, x0)

    x0_shape = x0.shape

    # initial residual
    r = b - A @ x0
    resnorms = [_norm(r, M)]

    if exact_solution is None:
        errnorms = None
    else:
        err = exact_solution - x0
        errnorms = [_norm(err)]

    num_steps = 0

    def cb(xk):
        nonlocal num_steps
        num_steps += 1

        xk = xk.reshape(x0_shape)

        if callback is not None:
            callback(xk)

        r = b - A @ xk
        resnorms.append(_norm(r, M))

        if exact_solution is not None:
            errnorms.append(_norm(exact_solution - x0))

    x, info = method(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M, atol=atol, callback=cb)
    success = info == 0
    x = x.reshape(x0_shape)
    return x if success else None, Info(success, x, num_steps, resnorms, errnorms)


def cg(*args, **kwargs):
    return _wrapper(scipy.sparse.linalg.cg, *args, **kwargs)


def bicg(*args, **kwargs):
    return _wrapper(scipy.sparse.linalg.bicg, *args, **kwargs)


def bicgstab(*args, **kwargs):
    return _wrapper(scipy.sparse.linalg.bicgstab, *args, **kwargs)


def cgs(*args, **kwargs):
    return _wrapper(scipy.sparse.linalg.cgs, *args, **kwargs)


def gmres(
    A,
    b: ArrayLike,
    x0: ArrayLike | None = None,
    tol: float = 1e-05,
    restart: int | None = None,
    maxiter: int | None = None,
    M=None,
    callback: Callable | None = None,
    atol: float | None = 0.0,
    exact_solution: ArrayLike | None = None,
):
    if x0 is None:
        x0 = np.zeros(A.shape[1])
    else:
        x0 = np.asarray(x0)

    assert_shapes(A, b, x0)

    x0_shape = x0.shape

    # scipy.gmres() apparently calls the callback before the start of the iteration such
    # that the initial residual is automatically contained
    resnorms = []
    num_steps = -1

    if exact_solution is None:
        errnorms = None
    else:
        errnorms = []

    def cb(xk):
        nonlocal num_steps
        num_steps += 1

        xk = xk.reshape(x0_shape)

        if callback is not None:
            callback(xk)

        r = b - A @ xk
        resnorms.append(_norm(r, M))

        if exact_solution is not None:
            errnorms.append(_norm(exact_solution - x0))

    x, info = scipy.sparse.linalg.gmres(
        A,
        b,
        x0=x0,
        tol=tol,
        restart=restart,
        maxiter=maxiter,
        M=M,
        atol=atol,
        callback=cb,
        callback_type="x",
    )
    success = info == 0
    x = x.reshape(x0_shape)
    return x if success else None, Info(success, x, num_steps, resnorms, errnorms)


# Need a special minres wrapper since it doesn't have atol.
# <https://github.com/scipy/scipy/issues/13935>
def minres(
    A,
    b: ArrayLike,
    x0: ArrayLike | None = None,
    shift: float = 0.0,
    tol: float = 1e-05,
    maxiter: int | None = None,
    M=None,
    callback: Callable | None = None,
    exact_solution=None,
):
    if x0 is None:
        x0 = np.zeros(A.shape[1])
    else:
        x0 = np.asarray(x0)

    assert_shapes(A, b, x0)

    x0_shape = x0.shape

    # initial residual
    resnorms = []
    r = b - A @ x0
    resnorms.append(_norm(r, M))

    if exact_solution is None:
        errnorms = None
    else:
        err = exact_solution - x0
        errnorms = [_norm(err)]

    num_steps = 0

    def cb(xk):
        nonlocal num_steps
        num_steps += 1

        xk = xk.reshape(x0_shape)

        if callback is not None:
            callback(xk)

        r = b - A @ xk
        resnorms.append(_norm(r, M))

        if exact_solution is not None:
            errnorms.append(_norm(exact_solution - x0))

    x, info = scipy.sparse.linalg.minres(
        A, b, x0=x0, shift=shift, tol=tol, maxiter=maxiter, M=M, callback=cb
    )
    success = info == 0
    x = x.reshape(x0_shape)
    return x if success else None, Info(success, x, num_steps, resnorms, errnorms)


# treat qmr separately because it can deal with two preconditioners, M1 and M2 (left and
# right)
def qmr(
    A,
    b: ArrayLike,
    x0: ArrayLike | None = None,
    tol: float = 1e-05,
    maxiter: int | None = None,
    M1=None,
    M2=None,
    callback: Callable | None = None,
    atol: float | None = 0.0,
    exact_solution: ArrayLike | None = None,
):
    if x0 is None:
        x0 = np.zeros(A.shape[1])

    x0_shape = x0.shape

    assert_shapes(A, b, x0)

    # initial residual
    resnorms = []
    r = b - A @ x0
    resnorms.append(_norm(r, M1))

    if exact_solution is None:
        errnorms = None
    else:
        err = exact_solution - x0
        errnorms = [_norm(err)]

    num_steps = 0

    def cb(xk):
        nonlocal num_steps
        num_steps += 1

        xk = xk.reshape(x0_shape)

        if callback is not None:
            callback(xk)

        r = b - A @ xk
        resnorms.append(_norm(r, M1))

        if exact_solution is not None:
            errnorms.append(_norm(exact_solution - x0))

    x, info = scipy.sparse.linalg.qmr(
        A, b, x0=x0, tol=tol, maxiter=maxiter, M1=M1, M2=M2, atol=atol, callback=cb
    )
    success = info == 0
    x = x.reshape(x0_shape)
    return x if success else None, Info(success, x, num_steps, resnorms, errnorms)
