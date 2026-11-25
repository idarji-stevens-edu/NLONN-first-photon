"""Figures 3-11 of the paper (PRA column widths), from the CSV files in ../results written by the MATLAB scripts.

    python make_figures.py        # writes fig3_confusion ... fig11_other_work (.pdf and .png) here

Measured values are filled blue, simulated values hollow orange; confusion matrices on a single-hue blue ramp.
"""
import csv
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / 'results'
COL1, COL2 = 3.375, 7.0                       # PRA single / double column widths (inches)
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#dfdedb'
RAMP = LinearSegmentedColormap.from_list('blue', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'axes.edgecolor': INK, 'axes.linewidth': 0.8, 'xtick.color': INK,
    'ytick.color': INK, 'axes.labelcolor': INK, 'text.color': INK, 'savefig.dpi': 600,
    'pdf.fonttype': 42,
    # boxed axes with inward ticks on all four sides
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6, 'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
})


def LEGEND():
    """Figs. 4 and 5: measured = filled blue, simulated = hollow orange."""
    return [Line2D([], [], marker='o', ls='', ms=5, mfc=BLUE, mec=BLUE, label='measured'),
            Line2D([], [], marker='o', ls='', ms=5, mfc='white', mec=ORANGE, mew=1.2, label='simulated')]


LEGEND_BOX = dict(frameon=True, facecolor='white', edgecolor='none', framealpha=1.0, borderpad=0.4)


def heatmap(ax, C, **kw):
    """Matrix drawn as vector squares (row 0 on top): sharp at any zoom, unlike an embedded image."""
    im = ax.pcolormesh(np.arange(C.shape[1] + 1) - 0.5, np.arange(C.shape[0] + 1) - 0.5, C, edgecolors='face', linewidth=0.2, **kw)
    ax.set_aspect('equal'); ax.invert_yaxis()
    return im


def read_csv(name):
    with open(DATA / name, newline='') as f:
        return list(csv.DictReader(f))


def _np(name, skip=0):
    return np.loadtxt(DATA / name, delimiter=',', ndmin=2, skiprows=skip)


def _save(fig, out, dpi=300):
    fig.savefig(HERE / f'{out}.pdf'); fig.savefig(HERE / f'{out}.png', dpi=dpi); plt.close(fig)


def fig_confusion(out='fig3_confusion'):
    """Fig. 3: two 3 x 3 confusion matrices of the 450 test images, rows the true class, columns the class
    named: (a) one photon, repeated single-shot trials (three_class_order3_confusion.csv); (b) ten photons
    decided together by the naive-Bayes rule of App. C (confusion_photons10.csv)."""
    fig, axs = plt.subplots(1, 2, figsize=(COL1, 1.95))
    for ax, name, tag in zip(axs, ('three_class_order3_confusion.csv', 'confusion_photons10.csv'),
                             ('(a) one photon', '(b) ten photons')):
        C = np.array([[float(r[k]) for k in ('named_2', 'named_4', 'named_7')] for r in read_csv(name)])
        heatmap(ax, C, cmap=RAMP, vmin=0, vmax=0.8)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{C[i, j]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if C[i, j] > 0.45 else INK)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(['2', '4', '7']); ax.set_yticklabels(['2', '4', '7'])
        ax.set_xlabel('Predicted class'); ax.set_title(tag, fontsize=8, pad=3)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel('True class')
    fig.tight_layout(pad=0.3, w_pad=1.0)
    _save(fig, out)


def fig_comparison(out='fig4_comparison'):
    """Fig. 4: accuracy of the NLONN (measured, filled, 1 sigma) compared to single-photon detection after linear
    optics (simulated, hollow) on the same 450 test images; dashed line the random guess (comparison.csv)."""
    v = {r['line']: r for r in read_csv('comparison.csv')}
    rows = [  # label, key, NLONN
        ('Lens', 'lens', False),
        (r'Fixed $\pi$ phase filter', 'fixed_pi_filter', False),
        ('Object-plane phase\nlayer + lens', 'object_plane_layer', False),
        ('NLONN, repeated\nsingle-shot trials', 'nlonn_trials', True),
        ('NLONN, single shot', 'nlonn_single_shot', True),
        ('Fourier-plane\nphase layer', 'fourier_plane_layer', False),
        ('Ideal NLONN\n(ten-mode model)', 'ideal_nlonn', True)]
    fig, ax = plt.subplots(figsize=(COL1, 2.45))
    n = len(rows)
    for k, (lab, key, nl) in enumerate(rows):
        a, e, meas = float(v[key]['acc']), float(v[key]['err']), v[key]['measured'] == '1'
        yk = k; c = BLUE if meas else ORANGE                  # lens at the bottom, ideal NLONN at the top
        if e:
            ax.errorbar(a, yk, xerr=e, fmt='none', ecolor=c, elinewidth=1.0, capsize=2)
        ax.plot(a, yk, 'o', ms=5.5, mfc=c if meas else 'white', mec=c, mew=1.3, zorder=3)
        ax.text(a + e + 1.8, yk, f'{a:.1f}' + (f' ± {e:.1f}' if e else ''), va='center', fontsize=7, color=INK)
    ax.axvline(100 / 3, color=INK, lw=0.8, ls=(0, (3, 2)))
    ax.text(100 / 3 + 0.8, n - 0.55, 'random guess', fontsize=6.5, color=INK, va='top')
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows], fontsize=6.8, color=INK)
    ax.set_xlim(30, 85); ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel('Accuracy (%)'); ax.set_ylabel('Optics before detection', fontsize=7.5)
    ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(axis='y', length=0)
    ax.legend(handles=LEGEND(), loc='lower right', fontsize=6.8, handletextpad=0.2, **LEGEND_BOX)
    fig.tight_layout(pad=0.3)
    _save(fig, out)


def fig_orders(kmax=10, out='fig5_mode_order'):
    """Fig. 5: accuracy per detected photon against the highest pump-mode order k = m + n in the kernels, to k = kmax.
    Simulated (hollow, line): the ideal NLONN, one kernel per class optimised within all modes of order <= k with each
    mode's modelled conversion (ideal_nlonn_orders.csv), with each experiment's order marked on the curve
    (open circles). Measured (filled): single shot, 1 sigma (accuracy_vs_mode_order.csv)."""
    R = read_csv('accuracy_vs_mode_order.csv')
    model = [(int(r['max_order']), float(r['accuracy'])) for r in read_csv('ideal_nlonn_orders.csv')]
    model = sorted(m for m in model if m[0] <= kmax)
    curve = dict(model)
    fig, ax = plt.subplots(figsize=(COL1, 2.45))
    ax.axhline(100 / 3, color=INK, lw=0.8, ls=(0, (3, 2)))
    ax.text(kmax + 0.3, 100 / 3 + 0.6, 'random guess', color=INK, fontsize=6.5, ha='right', va='bottom')
    ax.plot([k for k, _ in model], [a for _, a in model], '-', color=ORANGE, lw=1.4)
    ax.text(kmax + 0.3, model[-1][1] + 1.2, 'ideal NLONN', color=ORANGE, fontsize=6.5, ha='right')
    for r in R:
        k = int(r['max_order'])
        ax.plot(k, curve[k], 'o', ms=5, mfc='white', mec=ORANGE, mew=1.2, zorder=5)
        ax.errorbar(k, float(r['single_shot']), yerr=float(r['single_shot_se']), fmt='o', ms=5, mfc=BLUE, mec=BLUE,
                    ecolor=BLUE, elinewidth=1.0, capsize=2, zorder=4)
    ax.legend(handles=LEGEND(), loc='lower right', fontsize=6.8, handletextpad=0.2, bbox_to_anchor=(1.0, 0.12), **LEGEND_BOX)
    ax.set_xlabel(r'Highest pump-mode order $m+n$ in the kernels')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(range(0, kmax + 1, 2)); ax.set_xticks(range(0, kmax + 1), minor=True)
    ax.set_ylim(30, 90); ax.set_xlim(-0.5, kmax + 0.5)
    ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    _save(fig, out)


CLASS_COLORS = {2: '#2a78d6', 4: '#1a9e77', 7: '#8e5bd0'}      # App B: one hue per digit (orange stays "simulated")


def fig_tomography(out='fig7_tomography'):
    """App B: <M_x>_d - <M_x> for d = 2, 4, 7 (rows), real and imaginary parts (columns), on one diverging scale
    (tomography_class_means.csv)."""
    T, Mo = _np('tomography_class_means.csv'), _np('tomography_modes.csv')
    labs = [f'{int(m)}{int(n)}' for m, n in Mo[:, 1:3]]; d = len(labs); cs = (2, 4, 7)
    Mc = {}
    for c in cs:
        R = T[T[:, 0] == c]; A = np.zeros((d, d), complex)
        A[R[:, 1].astype(int) - 1, R[:, 2].astype(int) - 1] = R[:, 3] + 1j * R[:, 4]; Mc[c] = A
    Mall = sum(Mc.values()) / len(cs); lim = 0.045
    div = LinearSegmentedColormap.from_list('div', ['#b2412b', '#e8a58f', '#f7f6f3', '#86b6ef', '#1c5cab'])
    fig, axs = plt.subplots(3, 2, figsize=(COL1, 4.05), gridspec_kw=dict(left=0.13, right=0.86, top=0.955, bottom=0.085, wspace=0.06, hspace=0.06))
    for k, c in enumerate(cs):
        for r, f in enumerate((np.real, np.imag)):
            ax = axs[k, r]; im = heatmap(ax, f(Mc[c] - Mall), cmap=div, vmin=-lim, vmax=lim)
            ax.set_xticks(range(d)); ax.set_yticks(range(d)); ax.tick_params(length=0, labelsize=5.2)
            ax.set_xticklabels(labs if k == 2 else [], rotation=90); ax.set_yticklabels(labs if r == 0 else [])
            for s in ax.spines.values():
                s.set_visible(False)
        axs[k, 0].set_ylabel(f'Digit {c}\n' + r'HG$_{mn}$', fontsize=7)
    for r, part in enumerate(('Re', 'Im')):
        axs[0, r].set_title(part, fontsize=8, pad=3); axs[2, r].set_xlabel(r'HG$_{mn}$', fontsize=7)
    cax = fig.add_axes([0.885, 0.25, 0.02, 0.5]); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.6); cb.solids.set_edgecolor('face')
    _save(fig, out)


def fig_coherence(out='fig8_coherence'):
    """App B: eigenvalue spectra of M_x / Tr M_x per class, median and interquartile range over the class's
    training digits (tomography_eigenvalues.csv)."""
    E = _np('tomography_eigenvalues.csv'); y, ev = E[:, 0], E[:, 1:]
    fig, ax = plt.subplots(figsize=(COL1, 2.2)); w = 0.26
    for k, c in enumerate((2, 4, 7)):
        v = ev[y == c]; med = np.median(v, 0)[:6]; lo, hi = np.percentile(v, 25, 0)[:6], np.percentile(v, 75, 0)[:6]
        xs = np.arange(1, 7) + (k - 1) * w
        ax.bar(xs, med, w, color=CLASS_COLORS[c], label=f'digit {c}', edgecolor='white', lw=0.4)
        ax.errorbar(xs, med, yerr=[med - lo, hi - med], fmt='none', ecolor=INK, elinewidth=0.7, capsize=1.2)
    ax.set_ylim(0, 1); ax.set_xticks(range(1, 7)); ax.tick_params(axis='x', length=0)
    ax.set_xlabel(r'Eigenvalue index $i$'); ax.set_ylabel(r'$\lambda_i/\mathrm{Tr}\,M_x$')
    ax.legend(fontsize=6.5, frameon=False, loc='upper right'); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3); _save(fig, out)


def fig_training(out='fig9_training'):
    """App B: (a) the kernel design's training accuracy per fminunc iteration, without and with the 1 % leak floor;
    (b) the ten-mode ideal NLONN's training accuracy per gradient step (design_history.csv, ideal_nlonn_training.csv)."""
    H, I = _np('design_history.csv'), _np('ideal_nlonn_training.csv', 1)
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.3))
    for fl, c, lab in ((0, BLUE, 'no leak floor'), (0.01, INK2, '1 % leak floor')):
        h = H[np.isclose(H[:, 0], fl)]; axs[0].plot(h[:, 1], 100 * h[:, 3], color=c, lw=1.3, label=lab)
    axs[0].set_xscale('symlog', linthresh=10); axs[0].set_title('(a) Kernel design on the measured tomography', fontsize=8, loc='left')
    axs[0].set_xlabel('Optimizer iteration'); axs[0].legend(fontsize=6.5, frameon=False, loc='lower right')
    axs[1].plot(I[:, 0], 100 * I[:, 1], color=ORANGE, lw=1.3)
    axs[1].set_xscale('symlog', linthresh=10); axs[1].set_title('(b) Ideal NLONN (simulation)', fontsize=8, loc='left')
    axs[1].set_xlabel('Optimizer iteration')
    for ax in axs:
        ax.set_ylabel('Training accuracy (%)'); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4); _save(fig, out)


def fig_phase_layers(out='fig6_phase_layers'):
    """App B: training and validation accuracy per epoch of the two phase layers trained in simulation
    (fourier_plane_layer_training.csv, object_plane_layer_training.csv)."""
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    for f, c, name in (('fourier_plane_layer_training.csv', ORANGE, 'Fourier-plane layer'), ('object_plane_layer_training.csv', INK2, 'object-plane layer')):
        p = _np(f, 1)
        ax.plot(p[:, 0], 100 * p[:, 1], color=c, lw=1.3, label=f'{name}, training')
        ax.plot(p[:, 0], 100 * p[:, 2], color=c, lw=1.0, ls=(0, (3, 2)), label=f'{name}, validation')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)'); ax.legend(fontsize=6, loc='lower right', **LEGEND_BOX)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3); _save(fig, out)


def fig_photons(out='fig10_photons'):
    """App C, Fig. 10: accuracy against detected photons per decision, the NLONN from its recorded clicks
    (filled, 1 sigma) and the ideal NLONN from simulated photons (hollow); photon_number.csv."""
    D = np.loadtxt(DATA / 'photon_number.csv', delimiter=',', skiprows=1, ndmin=2); k = D[:, 0]
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    ax.axhline(100 / 3, color=INK, lw=0.8, ls=(0, (3, 2)))
    ax.text(1.05, 100 / 3 + 0.8, 'random guess', color=INK, fontsize=6.5, va='bottom')
    ax.plot(k, D[:, 3], '-', color=ORANGE, lw=1.2, zorder=2)
    ax.plot(k, D[:, 3], 'o', ms=2.8, mfc='white', mec=ORANGE, mew=0.8, zorder=3)
    ax.errorbar(k, D[:, 1], yerr=D[:, 2], fmt='o', ms=4, color=BLUE, ecolor=BLUE, elinewidth=0.9, capsize=1.5, zorder=4)
    for y, lab, c, va in ((D[-1, 3] + 1.2, 'ideal NLONN', ORANGE, 'bottom'), (D[-1, 1] - 3.5, 'NLONN', BLUE, 'top')):
        ax.text(105, y, lab, color=c, fontsize=6.5, ha='right', va=va)
    ax.set_xscale('log'); ax.set_xlim(0.85, 115); ax.set_ylim(30, 92)
    ax.set_xticks([1, 10, 100]); ax.set_xticklabels(['1', '10', '100'])
    ax.set_xlabel('Detected photons per decision'); ax.set_ylabel('Accuracy (%)')
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(handles=LEGEND(), loc='lower right', fontsize=6.8, handletextpad=0.2, bbox_to_anchor=(1.0, 0.12), **LEGEND_BOX)
    fig.tight_layout(pad=0.3); _save(fig, out)


def fig_others(out='fig11_other_work'):
    """App C, Fig. 11: normalized accuracy (A - 1/C)/(1 - 1/C) of MNIST classifiers, stacked in one column:
    (a) against detected photons per decision, (b) against detector readouts per decision (pixels x
    measurement settings shown in turn). This work from photon_number.csv; other work from
    other_work.csv in this folder (source of each number in its 'source' column). Labels name the authors; the
    caption gives the citations."""
    nz = lambda a, C: (np.asarray(a, float) - 100 / C) / (100 - 100 / C)
    D = np.loadtxt(DATA / 'photon_number.csv', delimiter=',', skiprows=1, ndmin=2); O = read_csv(HERE / 'other_work.csv')
    series = {}
    for r in O:
        series.setdefault(r['work'], []).append((float(r['photons']), float(r['readouts']),
                                                 float(nz(float(r['accuracy']), int(r['classes']))), r['kind']))
    name = {'Fischbacher et al.': 'Fischbacher et al.', 'Ma et al. 2026': 'Ma et al.', 'Wang et al.': 'Wang et al.',
            'Li et al.': 'Li et al.', 'Fan et al.': 'Fan et al.'}
    # label anchor per panel: (point index, x factor, y offset, ha, va)
    labs = ({'Li et al.': (0, 0.72, 0, 'right', 'center'), 'Fan et al.': (0, 0.72, 0, 'right', 'center'),
             'Fischbacher et al.': (0, 0.72, 0.02, 'right', 'center'), 'Ma et al. 2026': (5, 1.4, 0, 'left', 'center'),
             'Wang et al.': (0, 1.0, -0.06, 'center', 'top')},
            {'Li et al.': (0, 1.3, 0, 'left', 'center'), 'Fan et al.': (0, 1.3, 0, 'left', 'center'),
             'Fischbacher et al.': (0, 1.0, -0.06, 'center', 'top'), 'Ma et al. 2026': (0, 0.77, 0, 'right', 'center'),
             'Wang et al.': (1, 1.3, -0.05, 'left', 'center')})
    fs = 6.5
    fig, axs = plt.subplots(2, 1, figsize=(COL1, 3.9))
    for p, ax in enumerate(axs):
        ax.axhline(0, color=INK, lw=0.8, ls=(0, (3, 2)))
        ours = nz(D[:, 1], 3)
        if p == 0:
            ax.plot(D[:, 0], ours, '-', color=BLUE, lw=1.1, zorder=3)
            ax.plot(D[:, 0], ours, 'o', ms=2.8, color=BLUE, zorder=4)
            ax.text(D[-1, 0] * 1.4, ours[-1], 'this work', color=BLUE, fontsize=fs, ha='left', va='center')
        else:   # three readouts at every k: only the first photon is shown (the caption gives k = 100)
            ax.plot([3], ours[:1], 'o', ms=3.4, color=BLUE, zorder=4)
            ax.text(3 / 1.3, ours[0], 'this work', color=BLUE, fontsize=fs, ha='right', va='center')
        for w, pts in series.items():
            x, y, z, kind = zip(*pts); xv = x if p == 0 else y
            if len(xv) > 1:
                ax.plot(xv, z, '-', color=INK2, lw=0.8, zorder=3)
            ax.plot(xv, z, 's', ms=3.8, mfc='white' if kind[0] == 'theory' else INK2, mec=INK2, mew=0.9, zorder=4)
            i, fx, dy, ha, va = labs[p][w]
            ax.text(xv[i] * fx, z[i] + dy, name[w], color=INK, fontsize=fs, ha=ha, va=va)
        ax.set_xscale('log'); ax.set_ylim(-0.06, 1.06); ax.set_yticks([0, 0.5, 1])
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_ylabel('Accuracy above random\nguess, normalized', linespacing=1.0)
    axs[0].set_xlim(0.004, 2e6); axs[0].set_xticks([1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])
    axs[1].set_xlim(0.4, 3e3); axs[1].set_xticks([1, 10, 1e2, 1e3])
    axs[0].set_xlabel('Detected photons per decision', labelpad=1)
    axs[1].set_xlabel('Detector readouts per decision', labelpad=1)
    for ax, t in zip(axs, ('(a)', '(b)')):
        ax.text(-0.2, 1.0, t, transform=ax.transAxes, fontsize=8, va='top')
    fig.tight_layout(pad=0.3, h_pad=0.8); _check_labels(fig); _save(fig, out)


def _check_labels(fig):
    """Warn when a text label leaves its axes or covers a plotted point or line segment."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    for ax in fig.axes:
        bb = ax.get_window_extent(r)
        segs = []
        for ln in ax.get_lines():
            if ln.get_linestyle() == '--' or len(ln.get_xdata()) == 0:
                continue
            xy = ax.transData.transform(np.column_stack([ln.get_xdata(), ln.get_ydata()]))
            if len(xy) == 2 and np.allclose(xy[:, 1], xy[0, 1]) and abs(xy[1, 0] - xy[0, 0]) > bb.width * 0.9:
                continue        # the random-guess line
            for a, b in zip(xy[:-1], xy[1:]) if len(xy) > 1 else [(xy[0], xy[0])]:
                segs.append(np.linspace(a, b, 40))
            segs.append(xy)
        pts = np.vstack(segs) if segs else np.zeros((0, 2))
        for t in ax.texts:
            if t.get_transform() == ax.transAxes:
                continue
            e = t.get_window_extent(r).expanded(1.0, 1.0)
            if e.x0 < bb.x0 - 1 or e.x1 > bb.x1 + 1 or e.y0 < bb.y0 - 1 or e.y1 > bb.y1 + 1:
                print('label out of bounds:', repr(t.get_text()))
            pad = 1.5
            hit = (pts[:, 0] > e.x0 - pad) & (pts[:, 0] < e.x1 + pad) & (pts[:, 1] > e.y0 - pad) & (pts[:, 1] < e.y1 + pad)
            if hit.any():
                print('label covers data:', repr(t.get_text()))


if __name__ == '__main__':
    fig_confusion(); fig_comparison(); fig_orders(); fig_phase_layers()
    fig_tomography(); fig_coherence(); fig_training()
    fig_photons(); fig_others()
    print('wrote', ', '.join(p.name for p in sorted(HERE.glob('fig*.pdf'))))
