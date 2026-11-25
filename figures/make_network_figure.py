"""Fig. 1: (a) a single-layer digital classifier; (b) the NLONN drawn with data of one test image of the main
result (its trained kernels, its recorded clicks and rates; results/network_figure.mat from network_figure_data.m).
The kernels race with equal dwells: click times as recorded, rates unweighted (Eq. 5)."""
import numpy as np
import scipy.io as sio
from numpy.polynomial.hermite import hermval
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.colors import hsv_to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# usage: python make_network_figure.py   (run code/network_figure_data.m first)
import os
HERE = os.path.dirname(os.path.abspath(__file__))
D = sio.loadmat(os.path.join(HERE, '..', 'results', 'network_figure.mat'), squeeze_me=True)
out = os.path.join(HERE, 'fig1_network.png')
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8984'
GREYFILL, BLUE = '#f2f2f0', '#2a78d6'
CLS = {2: '#2a78d6', 4: '#eb6834', 7: '#1baf7a'}          # categorical slots 1-3, fixed order 2, 4, 7
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8, 'pdf.fonttype': 42,
                     'axes.edgecolor': INK, 'axes.linewidth': 0.8, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'image.interpolation': 'none'})


from plot_helpers import complex_rgb, slm_fields


def focus(f, x, u):
    """Field at the points u of the lens's Fourier plane (f = 200 mm, 1545 nm): the Fourier sum as a matrix product."""
    E = np.exp(-2j * np.pi * np.outer(u, x) / (1545e-9 * 0.2))
    return E @ f @ E.T


# ---- data
cc, w, dec, img = D['cc'], np.atleast_1d(D['w']), np.atleast_1d(D['dec']).astype(int), D['img'] / D['img'].max()
modes = D['modes']; T = [np.atleast_1d(D['t1']), np.atleast_1d(D['t2']), np.atleast_1d(D['t3'])]
rates = np.atleast_1d(D['rates'])
order = [list(dec).index(c) for c in (2, 4, 7)]            # rows in class order 2, 4, 7

# signal at its SLM (Gaussian 2.83 mm, digit 8.15 mm, phase pi v) and pump at its SLM (sum c HG, waist 2.0 mm)
N = 1200; L = 12e-3
X, Y, psi, Kp = slm_fields(D, N, L)
# at the crystal (Fourier plane, f = 200 mm): signal, pump, fiber mode (45 um waist); the integrand of Eq. (2)
R = 110e-6; u = np.linspace(-R, R, 420); U, V = np.meshgrid(u, u)      # the field drawn, half-width 110 um
Es = focus(psi, X[0], u); phi_f = np.exp(-(U ** 2 + V ** 2) / 45e-6 ** 2)
prod = [focus(k, X[0], u) * Es for k in Kp]               # pump x signal at the crystal
amp2 = np.array([abs((p * phi_f).sum()) ** 2 for p in prod]); print('model |<K|psi>|^2 shares (2,4,7):', np.round(amp2 / amp2.sum(), 2))

# ---- figure
FW, FH = 7.2, 3.3
fig = plt.figure(figsize=(FW, FH), facecolor='white')
cols = np.array([0.055, 0.245, 0.435, 0.625, 0.815]); cw = 0.17
heads = ['Input layer', 'Trainable weights', 'Linear step', 'Nonlinearity', 'Output layer']
for c, t in zip(cols, heads):
    fig.text(c + cw / 2, 0.975, t, ha='center', va='top', fontsize=8.5, fontweight='bold', color=INK)
fig.add_artist(plt.Line2D([0.01, 0.99], [0.925, 0.925], color=INK, lw=0.6, transform=fig.transFigure))


def arrow(x0, x1, y, col):
    fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=8, color=col, lw=1.0, shrinkA=0, shrinkB=0))


# (a) digital classifier
fig.text(0.01, 0.895, '(a) Single-layer digital classifier', fontsize=8, fontweight='bold', color=INK, va='top')
abox = ['pixel values\n' + r'$\boldsymbol{x}\in\mathbb{R}^{784}$', 'weights & biases\n' + r'$W,\ \boldsymbol{b}$',
        'weighted sum\n' + r'$\boldsymbol{z}=W\boldsymbol{x}+\boldsymbol{b}$', 'activation\n' + r'$\sigma(\boldsymbol{z})$', 'softmax:\nlargest score']
ya, ha = 0.655, 0.15
for k, (c, t) in enumerate(zip(cols, abox)):
    fig.patches.append(FancyBboxPatch((c + 0.01, ya), cw - 0.02, ha, boxstyle='round,pad=0.003,rounding_size=0.008',
                                      transform=fig.transFigure, fc=GREYFILL, ec=INK, lw=0.9))
    fig.text(c + cw / 2, ya + ha / 2, t, ha='center', va='center', fontsize=7, linespacing=1.25)
    if k < 4: arrow(c + cw - 0.005, cols[k + 1] + 0.005, ya + ha / 2, INK)

# (b) NLONN
fig.text(0.01, 0.585, '(b) NLONN: Nonlinear Optical Neural Network', fontsize=8, fontweight='bold', color=BLUE, va='top')
yb, hb = 0.07, 0.34
sq_w = hb * FH / FW
mid = yb + hb / 2

# titles: one line for the name, one for the equation, at the same height in every column
def title(c, name, eq=''):
    fig.text(c + cw / 2, yb + hb + 0.085, name, ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    if eq: fig.text(c + cw / 2, yb + hb + 0.025, eq, ha='center', va='baseline', fontsize=7.5)


title(cols[0], 'Signal', r'$\psi_x\propto A\,e^{i\pi v_x}$')
title(cols[1], 'Pump', r'$\Sigma\, c^{(j)}_{mn}\,\mathrm{HG}_{mn}$')
title(cols[2], 'Crystal', r'$E_p^{(j)}\,E_s$')
title(cols[3], 'Si-APD', r'$P(j\,|\,x)\propto p_j(x)$')
title(cols[4], 'Photon arrivals')

# 1 input, with the phase colour bar on its left
ax = fig.add_axes([cols[0] + (cw - sq_w) / 2 - 0.004, yb, sq_w, hb]); ax.imshow(complex_rgb(psi)); ax.axis('off')
cax = fig.add_axes([0.03, yb + 0.02, 0.009, hb - 0.04])
cb = fig.colorbar(ScalarMappable(Normalize(0, 2 * np.pi), 'hsv'), cax=cax, ticks=[0, np.pi, 2 * np.pi])
cb.ax.set_yticklabels(['0', r'$\pi$', r'$2\pi$'], fontsize=6.5); cb.ax.yaxis.set_ticks_position('left'); cb.outline.set_linewidth(0.8); cb.solids.set_edgecolor('face')
fig.text(0.0345, yb + hb - 0.005, 'phase', fontsize=6.5, ha='center', va='bottom')


def stack(col, imgs, fiber=False):
    hs = hb / 3 * 0.92; ws = hs * FH / FW
    for q, f in enumerate(imgs):
        y0 = yb + (2 - q) * hb / 3 + (hb / 3 - hs) / 2
        a = fig.add_axes([col + (cw - ws) / 2 + 0.008, y0, ws, hs]); a.imshow(complex_rgb(f, 0.35 if fiber else 0.7)); a.axis('off')
        if fiber:
            r = 45e-6 / (2 * 110e-6) * f.shape[0]
            a.add_patch(Circle((f.shape[1] / 2, f.shape[0] / 2), r, fill=False, ec=INK, lw=0.8, ls=(0, (2, 1.5))))
        fig.text(col + (cw - ws) / 2 - 0.002, y0 + hs / 2, str((2, 4, 7)[q]), ha='right', va='center', fontsize=8, fontweight='bold')


# 2 weights, 3 linear step
stack(cols[1], [k[330:870, 330:870] for k in Kp])
stack(cols[2], prod, fiber=True)

# 4 non-linearity: share of the first click, from measured rates (equal dwells)
pr = np.array([rates[j] for j in order]); pr = pr / pr.sum()
ax = fig.add_axes([cols[3] + 0.045, yb + 0.02, cw - 0.06, hb - 0.04])
ax.barh([2, 1, 0], pr, color=INK2, height=0.55)
for yy, pp in zip([2, 1, 0], pr): ax.text(pp + 0.02, yy, f'{pp:.2f}', va='center', fontsize=7)
ax.set_yticks([2, 1, 0]); ax.set_yticklabels(['2', '4', '7'], fontsize=8); ax.set_xlim(0, 1.2); ax.set_xticks([]); ax.tick_params(axis='y', length=0)

# 5 output: recorded clicks at their recorded times
ax = fig.add_axes([cols[4] + 0.045, yb + 0.05, cw - 0.05, hb - 0.06])
tmax = 600e-6; first = (np.inf, None, None)
for yy, j, cl in zip([2, 1, 0], order, (2, 4, 7)):
    tt = T[j]; tt = tt[tt < tmax]
    ax.plot(tt * 1e6, np.full_like(tt, yy), '|', ms=9, mew=1.1, color=INK)
    if len(tt) and tt[0] < first[0]: first = (tt[0], yy, cl)
ax.plot(first[0] * 1e6, first[1], 'o', ms=11, mfc='none', mec='#e34948', mew=1.2, clip_on=False, zorder=5)
ax.set_ylim(-0.6, 2.6); ax.set_xlim(-60, 640); ax.set_yticks([2, 1, 0]); ax.set_yticklabels(['2', '4', '7'], fontsize=8)
ax.set_xticks([0, 300, 600]); ax.tick_params(axis='x', labelsize=6.5, top=True); ax.tick_params(axis='y', right=True); ax.set_xlabel('time (μs)', fontsize=6.5, labelpad=1)

# arrows: the same positions and length as in panel (a)
for k in range(4): arrow(cols[k] + cw - 0.005, cols[k + 1] + 0.005, mid, BLUE)

fig.savefig(out, dpi=600)
fig.savefig(out.replace('.png', '.pdf'), dpi=600)
print('wrote', out, '| first click', first[2])
