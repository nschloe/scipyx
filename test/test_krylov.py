import numpy as np
import pytest
import scipy.sparse.linalg

import scipyx


def _sym():
    n = 10
    data = -np.ones((3, n))
    data[1] = 2.0
    A = scipy.sparse.spdiags(data, [-1, 0, 1], n, n)
    A = A.tocsr()
    b = np.ones(n)
    return A, b, None


def _sym_prec():
    n = 10
    data = -np.ones((3, n))
    data[1] = 2.0
    A = scipy.sparse.spdiags(data, [-1, 0, 1], n, n)
    A = A.tocsr()
    b = np.ones(n)
    #
    M = scipy.sparse.linalg.LinearOperator(
        (n, n), matvec=lambda x: 0.5 * x, rmatvec=lambda x: 0.5 * x
    )
    return A, b, M


@pytest.mark.parametrize(
    "method, system",
    [
        (scipyx.cg, _sym()),
        (scipyx.cg, _sym_prec()),
        (scipyx.gmres, _sym()),
        (scipyx.gmres, _sym_prec()),
        (scipyx.minres, _sym()),
        (scipyx.minres, _sym_prec()),
        (scipyx.bicg, _sym()),
        (scipyx.bicg, _sym_prec()),
        (scipyx.bicgstab, _sym()),
        (scipyx.bicgstab, _sym_prec()),
        (scipyx.cgs, _sym()),
        (scipyx.cgs, _sym_prec()),
        (scipyx.qmr, _sym()),
        # (scipyx.qmr, _sym_prec()),
    ],
)
def test_run(method, system, tol=1.0e-13):
    A, b, M = system

    exact_solution = scipy.sparse.linalg.spsolve(A, b)

    x0 = np.zeros(A.shape[1])
    if M is None:
        sol, info = method(
            A, b, x0, exact_solution=exact_solution, callback=lambda _: None
        )
    else:
        sol, info = method(
            A, b, x0, exact_solution=exact_solution, M=M, callback=lambda _: None
        )
    assert sol is not None
    assert info.success
    assert len(info.resnorms) == info.numsteps + 1
    assert len(info.errnorms) == info.numsteps + 1

    # make sure the initial resnorm and errnorm are correct
    if M is None:
        assert abs(np.linalg.norm(A @ x0 - b) - info.resnorms[0]) < tol
    else:
        r = A @ x0 - b
        resnorm0 = np.sqrt(np.dot(r, M @ r))
        assert abs(resnorm0 - info.resnorms[0]) < 1.0e-13
    assert abs(np.linalg.norm(x0 - exact_solution) - info.errnorms[0]) < tol
