import numpy as np
import scipy.sparse.linalg

import scipyx


def _run(method, resnorms1, resnorms2, tol=1.0e-13):
    n = 10
    data = -np.ones((3, n))
    data[1] = 2.0
    A = scipy.sparse.spdiags(data, [-1, 0, 1], n, n)
    A = A.tocsr()
    b = np.ones(n)

    exact_solution = scipy.sparse.linalg.spsolve(A, b)

    x0 = np.zeros(A.shape[1])
    sol, info = method(A, b, x0, exact_solution=exact_solution, callback=lambda _: None)
    assert sol is not None
    assert info.success
    print(info)
    assert len(info.resnorms) == info.numsteps + 1
    assert len(info.errnorms) == info.numsteps + 1
    print(info.resnorms)
    print()
    resnorms1 = np.asarray(resnorms1)
    assert np.all(np.abs(info.resnorms - resnorms1) < tol * (1 + resnorms1))
    # make sure the initial resnorm and errnorm are correct
    assert abs(np.linalg.norm(A @ x0 - b, 2) - info.resnorms[0]) < 1.0e-13
    assert abs(np.linalg.norm(x0 - exact_solution, 2) - info.errnorms[0]) < 1.0e-13

    # # with "preconditioning"
    # M = scipy.sparse.linalg.LinearOperator(
    #     (n, n), matvec=lambda x: 0.5 * x, rmatvec=lambda x: 0.5 * x
    # )
    # sol, info = method(A, b, M=M)

    # assert sol is not None
    # assert info.success
    # print(info.resnorms)
    # resnorms2 = np.asarray(resnorms2)
    # assert np.all(np.abs(info.resnorms - resnorms2) < tol * (1 + resnorms2))


def test_cg():
    _run(
        scipyx.cg,
        [
            3.1622776601683795,
            6.324555320336759,
            4.898979485566356,
            3.4641016151377544,
            2.0,
            0.0,
        ],
        [
            2.23606797749979,
            4.47213595499958,
            3.4641016151377544,
            2.449489742783178,
            1.4142135623730951,
            0.0,
        ],
    )


def test_gmres():
    _run(
        scipyx.gmres,
        [3.162277660168380e00, 7.160723346098895e-15],
        [2.236067977499790e00, 5.063396036227354e-15],
    )


def test_minres():
    _run(
        scipyx.minres,
        [
            3.1622776601683795,
            2.8284271247461903,
            2.449489742783178,
            2.0,
            1.4142135623730951,
            8.747542958250513e-15,
        ],
        [
            2.23606797749979,
            2.0,
            1.7320508075688772,
            1.4142135623730951,
            1.0,
            5.475099487534308e-15,
        ],
    )


def test_bicg():
    _run(
        scipyx.bicg,
        [
            3.1622776601683795,
            6.324555320336759,
            4.898979485566356,
            3.4641016151377544,
            2.0,
            0.0,
        ],
        [
            2.23606797749979,
            4.47213595499958,
            3.4641016151377544,
            2.449489742783178,
            1.4142135623730951,
            0.0,
        ],
    )


def test_bicgstab():
    _run(
        scipyx.bicgstab,
        [
            3.1622776601683795,
            2.87802343074627,
            1.920916697864717,
            0.8206213643164431,
            0.1632431286483802,
            4.699798436761765e-15,
        ],
        [
            2.23606797749979,
            2.0350698842944595,
            1.3582932231546119,
            0.5802669314947132,
            0.1154303232493776,
            3.3232593448441795e-15,
        ],
    )


def test_cgs():
    _run(
        scipyx.cgs,
        [
            3.1622776601683795,
            67.23094525588644,
            64.06246951218786,
            35.832945734337834,
            9.16515138991168,
            0.0,
        ],
        [
            2.23606797749979,
            47.53945729601885,
            45.2990066116245,
            25.337718918639855,
            6.48074069840786,
            0.0,
        ],
    )


def test_qmr():
    _run(
        scipyx.qmr,
        [
            3.1622776601683795,
            2.8284271247461903,
            2.449489742783178,
            2.0,
            1.4142135623730951,
            7.53644380168212e-15,
        ],
        [
            2.23606797749979,
            47.53945729601885,
            45.2990066116245,
            25.337718918639855,
            6.48074069840786,
            0.0,
        ],
    )


if __name__ == "__main__":
    test_gmres()
