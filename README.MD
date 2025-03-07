# RD-DIP: Rician Denoising - Deep Image Prior

*by HR Iglesias-Goldaracena, I. Ramírez & E. Schiavi.*

This repository contains the dependencies, code and datasets for harnessing RD-DIP, a major enhancement for the unsupervised learning Deep Image Prior (DIP) that allows it to perform Rician denoising.

## Graphical abstract

![](docs/images/grabs.png)

## Visualization of the (iterative) denoising process

|   |   |
|---|---|
| **Ground-truth**  | **Noisy** ($\sigma=0.15$)  |
| <img src="docs/images/gt.png" alt="Description" style="width: 100%; max-width: 256px;"/>  | <img src="docs/images/Std0.15.png" alt="Description" style="width: 100%; max-width: 256px;"/> 
| **Baseline DIP** | **RD-DIP** |
| <img src="docs/images/baseline_dip.gif" alt="Description" style="width: 100%; max-width: 256px;"/> | <img src="docs/images/rd_dip.gif" alt="Description" style="width: 100%; max-width: 256px;"/> |



## Getting started

Clone the repository, initialize the modules and install the requirements:

```
git clone git@github.com:heqro/rd-dip.git
cd rd-dip
git submodule init
git submodule update --recursive --remote
pip install -r requirements.txt
```

It is likely that, in the future, the file [`requirements.txt`](requirements.txt) may refer to software packages that are not obtainable anymore. Strictly speaking, only the following packages were explicitly installed by us for developing and executing the project:

- `torch` with the CUDA 11.8 version.
- `piqa`, used for in-GPU quality measurements.
- `scikit-image`, used for computing SSIM image.

Everything else that appears in the file came installed as dependencies by `pip`.
