import numpy as np
import pytest

import scipyx as spx


@pytest.mark.parametrize(
    "order,ref",
    [
        (0, 3.149489690570211),
        (1, 3.699432132107516),
        (2, 3.760927690803583),
        (3, 3.8905430128124),
    ],
)
def test_interpolate(order, ref):
    x = np.linspace(0.0, 1.0, 11)
    y = np.sin(7.0 * x)

    pp = spx.interp_rolling_lagrange(x, y, order)

    xtarget = np.linspace(0.0, 1.0, 101)
    ytarget = pp(xtarget)
    print(order, np.sum(ytarget))

    if ref is not None:
        assert abs(np.sum(ytarget) - ref) < abs(ref) * 1.0e-13

    return x, y, xtarget, ytarget


# if __name__ == "__main__":
#     x, y, xtarget, ytarget = test_interpolate(1, 3.699432132107516)
#     import matplotlib.pyplot as plt
#
#     plt.plot(x, y, "o", label="data")
#     plt.plot(xtarget, ytarget, "+", label="interpolation")
#     plt.savefig("interp-1.svg", transparent=True, bbox_inches="tight")
#     plt.show()
