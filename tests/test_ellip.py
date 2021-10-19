import pytest

import scipyx


# reference values from wolframalpha
@pytest.mark.parametrize(
    "x,m,ref",
    [
        (0.0, 0.8, (0.0, 1.0, 1.0)),
        (
            1.0,
            0.8,
            (
                0.7784765552276773412676842,
                0.6276736835019045390923315,
                0.7177599893896839554510713,
            ),
        ),
        (
            1.0j,
            0.8,
            (
                1.468016878814796144518175j,
                1.776252672477971493517182,
                1.650472309730796027504956,
            ),
        ),
        (
            1.0 + 2.0j,
            0.8,
            (
                +1.3249032709247202925 - 0.243227729431104000j,
                -0.3553341016168184437 - 0.906902019357462838j,
                -0.3675245790783220127 - 0.701456681043774595j,
            ),
        ),
    ],
)
def test_ellipj(x, m, ref):
    print(x, m)

    sn, cn, dn = scipyx.ellipj(x, m)
    print(sn)
    print(cn)
    print(dn)

    assert abs(sn - ref[0]) < 1.0e-13 * (1.0 + abs(sn))
    assert abs(cn - ref[1]) < 1.0e-13 * (1.0 + abs(cn))
    assert abs(dn - ref[2]) < 1.0e-13 * (1.0 + abs(dn))
