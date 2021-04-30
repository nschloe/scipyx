from collections import namedtuple
from typing import Callable, Optional

import numpy as np
import scipy
import scipy.sparse.linalg

Info = namedtuple("KrylovInfo", ["success", "xk", "numsteps", "resnorms", "errnorms"])


def _norm(r, M=None):
    Mr = r if M is None else M @ r
    rMr = np.dot(r.conj(), Mr)
    if np.any(rMr.imag) > 1.0e-13:
        raise RuntimeError("Got nonnegative imaginary part.")
    return np.sqrt(rMr.real)


def _wrapper(
    method,
    A,
    b,
    x0=None,
    tol: float = 1e-05,
    maxiter: Optional[int] = None,
    M=None,
    callback: Optional[Callable] = None,
    atol: Optional[float] = 0.0,
    exact_solution=None,
):
    if x0 is None:
        x0 = np.zeros(A.shape[1])

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

        if callback is not None:
            callback(xk)

        r = b - A @ xk
        resnorms.append(_norm(r, M))

        if exact_solution is not None:
            errnorms.append(_norm(exact_solution - x0))

    x, info = method(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M, atol=atol, callback=cb)

    success = info == 0

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
    b,
    x0=None,
    tol: float = 1e-05,
    restart: Optional[int] = None,
    maxiter: Optional[int] = None,
    M=None,
    callback: Optional[Callable] = None,
    atol: Optional[float] = 0.0,
    exact_solution=None,
):
    if x0 is None:
        x0 = np.zeros(A.shape[1])

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

    return x if success else None, Info(success, x, num_steps, resnorms, errnorms)


# Need a special minres wrapper since it doesn't have atol.
# <https://github.com/scipy/scipy/issues/13935>
def minres(
    A,
    b,
    x0=None,
    shift: float = 0.0,
    tol: float = 1e-05,
    maxiter: Optional[int] = None,
    M=None,
    callback: Optional[Callable] = None,
    exact_solution=None,
):
    if x0 is None:
        x0 = np.zeros(A.shape[1])

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

    return x if success else None, Info(success, x, num_steps, resnorms, errnorms)


# treat qmr separately because it can deal with two preconditioners, M1 and M2 (left and
# right)
def qmr(
    A,
    b,
    x0=None,
    tol: float = 1e-05,
    maxiter: Optional[int] = None,
    M1=None,
    M2=None,
    callback: Optional[Callable] = None,
    atol: Optional[float] = 0.0,
    exact_solution=None,
):
    if x0 is None:
        x0 = np.zeros(A.shape[1])

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

    return x if success else None, Info(success, x, num_steps, resnorms, errnorms)
