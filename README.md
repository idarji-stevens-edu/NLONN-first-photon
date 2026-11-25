# All-Optical Neural Network by First-Photon Detection — data and code

Author: Ishan Darji (Stevens Institute of Technology)

Measured data and analysis code behind the numbers and figures of the paper I. Darji, S. Kumar and Y.-P. Huang, *All-Optical Neural Network by First-Photon Detection* (the arXiv link will be added).

## Contents

| | |
|---|---|
| `data/` | measured data of the five experiments, and MNIST; the variables are described in [data/README.md](data/README.md) |
| `code/` | MATLAB analysis scripts and `functions/`; [code/README.md](code/README.md) lists the script behind each number and figure of the paper |
| `results/` | CSV files written by the scripts |
| `figures/` | Python scripts that draw the figures from `results/`, and the figures |

## Requirements

- MATLAB R2025a (R2020b or later should work) with the Optimization, Image Processing, and Statistics and Machine Learning Toolboxes.
- Python 3 with numpy, matplotlib and scipy (figures only).

## Running

1. In MATLAB, from `code/`, run `run_all.m` (about 2.5 h on a desktop PC) or any single script. `comparison_margins.m` reads the results of the other scripts, so run it after them.
2. From `figures/`, run `python make_figures.py` (Figs. 3–11) and `python make_network_figure.py` (Fig. 1).

The CSV files are committed, so step 2 works without step 1.

## Licence

Code: MIT ([LICENSE](LICENSE)). Data: CC BY 4.0 ([data/LICENSE](data/LICENSE)). MNIST (`data/mnist/`) is from Y. LeCun, C. Cortes and C. J. C. Burges, redistributed under CC BY-SA 3.0.
