# Scripts behind the numbers and figures of the paper

## Numbers

Run each script from this folder in MATLAB. Each prints the numbers below and writes its CSV files to `../results/`. The slow scripts are `object_plane_phase_layer.m` (~25 min), `fourier_plane_phase_layer.m` (~45 min) and `masked_cameras.m` (~1 h).

| Paper | Number | Script |
|---|---|---|
| Sec. III B | 0.02–720 counts/ms; ≤ 7×10⁻³ photons per pulse | `three_class_order3.m` |
| Sec. IV A | 56.2 ± 2.3 % single shot; 53.9 ± 1.4 % over 100 µs trials (median 643 per image) | `three_class_order3.m` |
| Sec. IV A, Fig. 3a | one-photon confusion matrix (0.73, 0.21, 0.67 on the diagonal; standard errors ≤ 0.03); kernel naming 4 the dimmest on every class | `three_class_order3.m` |
| Sec. IV A, App. B | training 54.9 ± 2.4 %, the same under nested 5-fold CV; same choice among all 48 masks | `three_class_order3.m` |
| Sec. IV A, App. C, Figs. 3b, 10 | several photons per decision (naive Bayes): NLONN 53.9 / 69.7 ± 1.4 / 75.3 ± 1.8 % at 1 / 10 / 100 photons; ten-photon confusion matrix (0.77, 0.57, 0.75 on the diagonal); a click of the kernel naming 4 is 4.0 times more likely from a four than from a two, and 3.3 times more likely than from a seven; ideal NLONN 75.6 → 85.4 %; exact kernel probabilities by 15 nearest training digits 78.9 % (measured), 84.2 % (ideal) | `photon_number.m` |
| Sec. V A, Fig. 5 | order ≤ 1: 46.0 / 45.2 %; order ≤ 2: 49.6 / 48.3 % | `accuracy_vs_mode_order.m` |
| Sec. V B | ten classes: 18.0 ± 2.4 % single shot, 17.3 ± 0.7 % over trials | `ten_class_order2.m` |
| Sec. IV B, Fig. 4 | paired margins over the lens (+14.9 / +12.6 pp), the π filter (+9.4 / +7.1), the object-plane layer (+4.7 / +2.4), the Fourier-plane layer (−1.5 / −3.9) and the masked cameras (+6.2, +2.4) | `comparison_margins.m` |
| Sec. III A, App. B | fiber mode of about 45 µm waist in the crystal; phase mismatch Δk ≈ 66 m⁻¹ at which HG00 converts a blank signal best | `crystal_model.m` |
| Sec. IV B, V A, Figs. 4, 5, 9b | ideal NLONN: 59.8 / 75.9 / 78.0 % for orders ≤ 1 / 2 / 3, rising to 84.0 % at order ≤ 10; 0.128 % of photons converted; 75.6 % on the 450 test images; the experiment delivers 0.51 of its gain over a random guess, (56.2 − 33.3)/(78.0 − 33.3) | `ideal_nlonn.m` |
| Sec. V B | ideal NLONN, ten classes, six modes: 31.5 % | `ideal_nlonn.m` |
| Sec. V B | choice of digits 2, 4, 7: cross-validated design on a ten-class tomography, largest margin over the simulated camera behind an absorbing mask among the leading subsets (51.8 % vs 46.5 %). The ranking of all 582 subsets of 3–5 classes was done at acquisition time and is not reproduced here | `class_subset_choice.m` |
| App. A, Eq. (A1) | pixel bound 22.96 % (ten classes), 54.4 % (2, 4, 7) | `pixel_bound.m` |
| App. A | lens + camera 41.3 % (64 × 64 crop 40.6 %, rule from all test images 41.5 %); 17 fixed filters, π plate 46.8 %, best defocus 44.3 % | `lens_camera_and_fixed_filters.m` |
| App. A, Fig. 6 | object-plane phase layer 51.5 % (6,724 phases) | `object_plane_phase_layer.m` |
| App. A, Fig. 6 | Fourier-plane phase layer 57.7 % (59.0 % on all 3,042 test images) | `fourier_plane_phase_layer.m` |
| App. A | purity-ranked mask 50.0 % (86 pixels, purity 0.53–0.58, 1.3 % passing; brightest pixel purity 0.38); trained mask 53.8 % (10 pixels, ~0.01 % passing) | `masked_cameras.m` |
| App. B, Fig. 8 | largest-eigenvalue share median 0.894 (IQR 0.873–0.908) | `tomography_and_design.m` |
| App. B, Figs. 7, 9a | class-mean response matrices; designs at floors 0 / 0.01 predict 58.1 / 50.3 % | `tomography_and_design.m` |
| App. B | designed masks' measured/predicted class-mean rates 0.21–1.35 (median 0.80); kernel naming 4 at 0.28–0.43, the others 0.54–1.35 | `tomography_and_design.m` |
| App. D | flagged acquisitions 0/3,150, 57/4,950, 13/7,200, 0/4,000; accuracy change ≤ 0.2 pp | `reference_drift_flags.m` |

Not included: the pump-modulator lookup table and the dark-mask check of Appendix D, which are bench calibrations rather than analysis of these data. The energy estimates of Appendix C (0.26 aJ per detected photon, the pump energy per decision from a calibration of the pump power, the digital cost of the three overlaps) are arithmetic on stated values, not analysis of these data, and have no script.

## Figures

| Figure | Drawn by | Data from |
|---|---|---|
| Fig. 1 | `make_network_figure.py` | `network_figure_data.m` |
| Fig. 2 | final render, `figures/fig2_setup.png` | not data-driven |
| Fig. 3 | `make_figures.py` | `three_class_order3.m` (a), `photon_number.m` (b) |
| Fig. 4 | `make_figures.py` | `comparison_margins.m` (after `ideal_nlonn.m`) |
| Fig. 5 | `make_figures.py` | `accuracy_vs_mode_order.m`, `ideal_nlonn.m` |
| Fig. 6 | `make_figures.py` | `object_plane_phase_layer.m`, `fourier_plane_phase_layer.m` |
| Figs. 7, 8, 9a | `make_figures.py` | `tomography_and_design.m` |
| Fig. 9b | `make_figures.py` | `ideal_nlonn.m` |
| Fig. 10 | `make_figures.py` | `photon_number.m` |
| Fig. 11 | `make_figures.py` | `photon_number.m`; other work from `figures/other_work.csv` (published values, each with its source) |

The response matrices rebuilt here from the tomography rates differ from those computed at acquisition in the last bit, so the no-floor design in Fig. 9a stops a few iterations earlier (368 instead of 392) at the same 58.1 %. The ideal-NLONN curves of Figs. 5 and 9b use Δk = 66.3 m⁻¹, rounded from the fitted peak used for the paper, and differ from the paper's by less than 0.01 percentage points, well below the line width. Every other figure is identical to the paper's.
