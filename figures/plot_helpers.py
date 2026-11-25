"""Helpers for Fig. 1 (make_network_figure.py): complex fields as colour, HG modes, the two SLM fields."""
import numpy as np
from numpy.polynomial.hermite import hermval
from matplotlib.colors import hsv_to_rgb


def complex_rgb(f, gamma=0.7):
    """Hue = phase (0..2 pi), fading to white where the amplitude is small; a lower gamma lifts weak parts."""
    a = np.abs(f) / np.abs(f).max()
    h = (np.angle(f) % (2 * np.pi)) / (2 * np.pi)
    return hsv_to_rgb(np.dstack([h, np.minimum(1, 1.0 * a ** gamma), 1 - 0.15 * a ** gamma]))


def hg(m, n, X, Y, w):
    hm = hermval(np.sqrt(2) * X / w, np.eye(m + 1)[m]); hn = hermval(np.sqrt(2) * Y / w, np.eye(n + 1)[n])
    f = hm * hn * np.exp(-(X ** 2 + Y ** 2) / w ** 2)
    return f / np.sqrt((np.abs(f) ** 2).sum())


def slm_fields(D, N=400, L=12e-3):
    """Signal and pump fields at their SLMs for Fig. 1's test image: the signal A exp(i pi v) (Gaussian 2.83 mm,
    digit 8.15 mm), the pump superpositions sum c HG (waist 2.0 mm) in class order 2, 4, 7. Returns X, Y, psi, Kp."""
    img = D['img'] / D['img'].max(); cc, dec, modes = D['cc'], np.atleast_1d(D['dec']).astype(int), D['modes']
    x = (np.arange(N) - N / 2) * L / N; X, Y = np.meshgrid(x, x)
    A = np.exp(-(X ** 2 + Y ** 2) / 2.83e-3 ** 2)
    ix = np.floor((X + 8.15e-3 / 2) / 8.15e-3 * 28).astype(int); iy = np.floor((Y + 8.15e-3 / 2) / 8.15e-3 * 28).astype(int)
    inside = (ix >= 0) & (ix < 28) & (iy >= 0) & (iy < 28)
    v = np.zeros_like(X); v[inside] = img[iy[inside], ix[inside]]
    psi = A * np.exp(1j * np.pi * v)
    order = [list(dec).index(c) for c in (2, 4, 7)]
    Kp = [sum(cc[k, j] * hg(int(modes[k, 0]), int(modes[k, 1]), X, Y, 2.0e-3) for k in range(cc.shape[0])) for j in order]
    return X, Y, psi, Kp
