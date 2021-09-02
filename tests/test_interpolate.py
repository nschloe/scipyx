import numpy as np

import scipyx as spx


def test_interpolate():
    x = np.linspace(0.0, 1.0, 11)
    y = np.sin(2 * np.pi * x)

    pp = spx.interp_rolling_lagrange(x, y, 1)

    xtarget = np.linspace(0.0, 1.0, 101)
    ytarget = pp(xtarget)

    import matplotlib.pyplot as plt

    plt.plot(x, y, "o")
    plt.plot(xtarget, ytarget, "+")
    plt.show()


if __name__ == "__main__":
    test_interpolate()
