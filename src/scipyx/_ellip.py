import numpy as np
import scipy.special


def ellipj(u, m):
    sn, cn, dn, _ = scipy.special.ellipj(np.real(u), m)

    if np.all(np.imag(u) == 0.0):
        return sn, cn, dn

    # formulas (57) ff. from
    # <https://mathworld.wolfram.com/JacobiEllipticFunctions.html>
    # or
    # <https://paramanands.blogspot.com/2011/01/elliptic-functions-complex-variables.html>

    # k = np.sqrt(m)
    # k_ = np.sqrt(1 - k ** 2)
    # m_ = k_ ** 2

    m_ = 1.0 - m
    sn_, cn_, dn_, _ = scipy.special.ellipj(np.imag(u), m_)

    e = 1.0 - (dn * sn_) ** 2

    sni = (sn * dn_ + 1j * (cn * dn * sn_ * cn_)) / e
    cni = (cn * cn_ - 1j * (sn * dn * sn_ * dn_)) / e
    dni = (dn * cn_ * dn_ - 1j * m * (sn * cn * sn_)) / e

    return sni, cni, dni
