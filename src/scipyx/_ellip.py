from __future__ import annotations

import numpy as np
import scipy.special
from numpy.typing import ArrayLike


def ellipj(
    u: ArrayLike, m: complex
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sn, cn, dn, ph = scipy.special.ellipj(np.real(u), m)

    if np.all(np.imag(u) == 0.0):
        return sn, cn, dn, ph

    # formulas (57) ff. from
    # <https://mathworld.wolfram.com/JacobiEllipticFunctions.html>
    # or
    # <https://paramanands.blogspot.com/2011/01/elliptic-functions-complex-variables.html>

    # k = np.sqrt(m)
    # k_ = np.sqrt(1 - k ** 2)
    # m_ = k_ ** 2

    m_ = 1.0 - m
    sn_, cn_, dn_, _ = scipy.special.ellipj(np.imag(u), m_)

    D = 1.0 - (dn * sn_) ** 2

    sni = (sn * dn_ + 1j * (cn * dn * sn_ * cn_)) / D
    cni = (cn * cn_ - 1j * (sn * dn * sn_ * dn_)) / D
    dni = (dn * cn_ * dn_ - 1j * m * (sn * cn * sn_)) / D

    # Evaluating Jacobi elliptic functions in the complex domain
    # <http://www.peliti.org/Notes/elliptic.pdf>
    X0 = sn * dn_
    X1 = cn * cn_
    Y = sn_ * dn
    K = scipy.special.ellipk(m)
    nx = np.floor((np.real(u) + 2 * K) / (4 * K))
    phi = np.arctan2(X0, X1) + 1j * np.arctanh(Y) + 2 * np.pi * nx

    return sni, cni, dni, phi


# For weierstrass_p, we'd need ellipj with complex-valued modulus `m`.
# def weierstrass_p(z: ArrayLike, g2: float, g3: float) -> np.ndarray:
#
#     g2 = 2.0
#     g3 = 2.0
#
#     # Compute the constants e{1,2,3}
#     # https://en.wikipedia.org/wiki/Weierstrass_elliptic_function#The_constants_e1,_e2_and_e3
#     e1, e2, e3 = np.roots([4.0, 0.0, -g2, -g3])
#
#     print("e")
#     print(e1)
#     print(e2)
#     print(e3)
#     print("sum(e)", e1 + e2 + e3)
#
#     # sum(e) == 0
#     # g2 = -4 * (e[0] * e[1] + e[1] * e[2] + e[2] * e[0])
#     # g3 = 4 * e[0] * e[1] * e[2]
#
#     print(z * np.sqrt(e1 - e3))
#     print("m", (e2 - e3) / (e1 - e3))
#     print("m", (e3 - e1) / (e2 - e1))
#     print("m", (e1 - e2) / (e3 - e2))
#
#     exit(1)
#
#     sn, _, _ = ellipj(z * np.sqrt(e1 - e3), (e2 - e3) / (e1 - e3))
#
#     return e3 + (e1 - e3) / sn ** 2
