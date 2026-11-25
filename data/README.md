# Data key

Folders:

| Folder | Contents |
|---|---|
| `mnist/` | MNIST training and test sets (images, labels) |
| `three_class_order3/` | digits 2, 4, 7; kernels from the 10 HG modes of order m+n <= 3 (main result) |
| `three_class_order2/` | digits 2, 4, 7; 6 modes of order <= 2 |
| `three_class_order1/` | digits 2, 4, 7; 3 modes of order <= 1 |
| `ten_class_order2/` | all ten digits; 6 modes of order <= 2 |
| `subset_choice/` | ten-class tomography (4 modes) used to choose the class subset |

Each experiment folder holds the measured data of one run. Rates are in detected photons per ms.

**`tomography.mat`**: response tomography on training digits (Appendix B)

| Variable | Meaning |
|---|---|
| `classes` | digit classes measured |
| `mode_mn` | 2 x d: (m, n) of each HG mode; `mode_names` the same as text |
| `mask_kind` | `'single'` (one mode) or `'pair'` (two modes superposed) |
| `mask_mode_a`, `mask_mode_b` | the mode(s) of each mask (index into `mode_mn`) |
| `mask_phase_deg` | relative phase of the pair (0, 90, 180, 270) |
| `mask_scale` | s_ab, scale of the pair's field to unit modulator amplitude |
| `mask_coeffs` | d x masks: HG coefficients written on the pump |
| `pair_phases_deg`, `phase_factor`, `pair_scale` | phase steps and constants of the pair fit (Eq. B3) |
| `mnist_index`, `label` | MNIST training-set index and class of each digit |
| `rate_recorded` | masks x digits: rate as recorded (dead time 50 ns corrected, dark rate subtracted) |
| `rate` | `rate_recorded` divided by the reference rate interpolated to the acquisition, relative to its mean, and by the mean calibration rate (Appendix D) |

**`library.mat`**: the kernel library measured on further training digits

| Variable | Meaning |
|---|---|
| `mask_kind` | `'single'` or `'designed'` |
| `mask_mode` | mode index (single) or design index (designed) |
| `mask_global_phase_deg` | global phase the mask was written at |
| `mask_coeffs` | d x masks: HG coefficients |
| `mnist_index`, `label`, `rate_recorded`, `rate` | as in `tomography.mat` |

**`test.mat`**: the test

| Variable | Meaning |
|---|---|
| `test_mask` | library masks measured in the test (index into `library.mat`) |
| `selected_mask`, `selected_class` | the trained kernels (library index) and the class each names |
| `mnist_index`, `label` | MNIST test-set index and class of each test image |
| `rate_recorded`, `rate` | test masks x images, as in `tomography.mat` |
| `acq_time` | time of each acquisition (s from the run start) |
| `t_int` | acquisition time per (mask, image) pair (s) |

**`references.mat`**: drift control (Appendix D)

| Variable | Meaning |
|---|---|
| `ref_time`, `ref_rate`, `ref_ok` | reference-kernel readings (s, /ms) and whether each passed |
| `cal_time`, `cal_rate` | calibration-mask readings |
| `dark_rate` | dark rate (/ms), measured once per run |
| `flagged_tomography` | drift flag of each tomography acquisition, in acquisition order |
| `flagged_library`, `flagged_test` | drift flags, masks x digits |

**`clicks_k<j>.mat`**: recorded photon clicks of trained kernel j in the test

| Variable | Meaning |
|---|---|
| `kernel_mask`, `kernel_class` | the kernel (library index) and the class it names |
| `pat_ps` | 1 x images cell: time of every recorded click (int64 ps from the acquisition's start) |
| `papt_ps` | 1 x images cell: time tag of the laser pulse just before each click (int64 ps) |
| `t_int` | acquisition time (s) |

The time tagger records a tag for every laser pulse (100 MHz) and every click. Only the clicks and their preceding pulse tags were kept at acquisition; pulses without a click carry no information. Times are the tagger's integer picoseconds: `double(pat_ps{i}) * 1e-12` gives seconds.
