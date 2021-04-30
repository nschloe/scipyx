import numpy as np
import pytest
import scipy.sparse.linalg

import scipyx


def _spd():
    n = 10
    data = -np.ones((3, n))
    data[1] = 2.0
    A = scipy.sparse.spdiags(data, [-1, 0, 1], n, n)
    A = A.tocsr()
    b = np.ones(n)
    return A, b, None


def _spd_prec():
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


def _hpd():
    a = np.array(np.linspace(1.0, 2.0, 5), dtype=complex)
    a[0] = 5.0
    a[-1] = 1.0e-1
    A = np.diag(a)
    A[-1, 0] = 1.0e-1j
    A[0, -1] = -1.0e-1j

    b = np.ones(5, dtype=complex)
    return A, b, None


@pytest.mark.parametrize(
    "method, system",
    [
        (scipyx.cg, _spd()),
        (scipyx.cg, _spd_prec()),
        (scipyx.cg, _hpd()),
        #
        (scipyx.gmres, _spd()),
        (scipyx.gmres, _spd_prec()),
        (scipyx.gmres, _hpd()),
        #
        (scipyx.minres, _spd()),
        (scipyx.minres, _spd_prec()),
        # (scipyx.minres, _hpd()),  ERR minres can't deal with hermitian?
        #
        (scipyx.bicg, _spd()),
        (scipyx.bicg, _spd_prec()),
        (scipyx.bicg, _hpd()),
        #
        (scipyx.bicgstab, _spd()),
        (scipyx.bicgstab, _spd_prec()),
        (scipyx.bicgstab, _hpd()),
        #
        (scipyx.cgs, _spd()),
        (scipyx.cgs, _spd_prec()),
        (scipyx.cgs, _hpd()),
        #
        (scipyx.qmr, _spd()),
        # (scipyx.qmr, _spd_prec()),
        (scipyx.qmr, _hpd()),
    ],
)
def test_run(method, system, tol=1.0e-13):
    A, b, M = system

    if isinstance(A, np.ndarray):
        exact_solution = np.linalg.solve(A, b)
    else:
        exact_solution = np.linalg.solve(A.toarray(), b)

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

    print(info.resnorms)
    assert np.asarray(info.resnorms).dtype == float
    assert np.asarray(info.errnorms).dtype == float

    # make sure the initial resnorm and errnorm are correct
    if M is None:
        assert abs(np.linalg.norm(A @ x0 - b) - info.resnorms[0]) < tol
    else:
        r = A @ x0 - b
        resnorm0 = np.sqrt(np.dot(r, M @ r))
        assert abs(resnorm0 - info.resnorms[0]) < 1.0e-13
    assert abs(np.linalg.norm(x0 - exact_solution) - info.errnorms[0]) < tol
