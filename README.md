<div align="center">

# FastGS-Adaptive

### Structure-Aware View-Consistent Densification for Accelerated 3D Gaussian Splatting

<p>
<a href="#">
<img alt="Paper" src="https://img.shields.io/badge/Paper-Coming%20Soon-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white">
</a>
<a href="#">
<img alt="Code" src="https://img.shields.io/badge/Code-Available-blue?style=for-the-badge&logo=github&logoColor=white">
</a>
<a href="#">
<img alt="Python 3.7+" src="https://img.shields.io/badge/Python-3.7%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
</a>
<a href="#">
<img alt="PyTorch 1.12+" src="https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
</a>
<a href="https://hanalebeta.github.io/FastGS-Adaptive/">
<img alt="Project Page" src="https://img.shields.io/badge/Project-Page-4c9a2a?style=for-the-badge&logo=githubpages&logoColor=white">
</a>
<a href="LICENSE">
<img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge&logo=apache&logoColor=white">
</a>
</p>

**[Hana L. Goshu](mailto:hana-lebeta.goshu@connect.polyu.hk)<sup>1</sup> &nbsp;&middot;&nbsp;
[Tadesse G. Wakjira](mailto:twakjira@kennesaw.edu)<sup>2</sup> &nbsp;&middot;&nbsp;
[Kin-Man Lam](mailto:kin.man.lam@polyu.edu.hk)<sup>1</sup>**

<sup>1</sup> Department of Electrical and Electronic Engineering, The Hong Kong Polytechnic University, Hong Kong
<sup>2</sup> Department of Civil and Environmental Engineering, Kennesaw State University, Marietta, GA, USA

*Preprint, 2026 (Under Review)*

</div>

---

## Architecture

<div align="center">
<img src="static/images/architecture.png" width="95%">
<br>
<em>FastGS-Adaptive. Three additive mechanisms &mdash; an adaptive edge-structured loss (AESL), a hard example mining sampler (HEM-S), and an annealed densification schedule (ADS) &mdash; make densification structure-aware and view-consistent without modifying the CUDA rasteriser.</em>
</div>

---

<div align="center">
<img src="assets/demo_kitchen.gif" width="95%">
<br>
<em>Ground Truth (left) vs FastGS-Adaptive (Ours, right) on the <b>kitchen</b> scene.</em>
</div>

<div align="center">
<img src="assets/demo_bonsai.gif" width="95%">
<br>
<em>Ground Truth (left) vs FastGS-Adaptive (Ours, right) on the <b>bonsai</b> scene.</em>
</div>

<div align="center">
<img src="assets/demo_bicycle.gif" width="95%">
<br>
<em>Ground Truth (left) vs FastGS-Adaptive (Ours, right) on the <b>bicycle</b> outdoor scene.</em>
</div>

---

## Abstract

3D Gaussian Splatting (3DGS) has established itself as a leading representation for real-time novel
view synthesis. Recent acceleration frameworks such as FastGS have significantly reduced per-scene
training time through multi-view consistent densification and pruning. However, FastGS employs a
structure-agnostic L1 error metric to guide densification: the scoring head evaluates each pixel
solely by photometric residual magnitude, without distinguishing residuals on geometric edges from
those on textureless surfaces.

We propose **FastGS-Adaptive**, a structure-aware densification framework that introduces three
additive mechanisms:

- **AESL** (Adaptive Edge-Structured Loss) modulates the photometric penalty using Sobel-derived
  gradient priors.
- **HEM-S** (Hard Example Mining Sampler) replaces uniform camera selection with a
  loss-history-biased policy, focusing densification capacity on under-reconstructed views.
- **ADS** (Annealed Densification Schedule) decays the split frequency across three phases to
  eliminate late-stage overhead.

FastGS-Adaptive surpasses existing state-of-the-art acceleration methods on all three standard
quality metrics.

## Results

Baseline figures are those reported in the corresponding publications.

**Mip-NeRF 360**

| Method | PSNR &uarr; | SSIM &uarr; | LPIPS &darr; | Primitives |
|---|---|---|---|---|
| 3DGS | 27.53 | .812 | .221 | 2.63 M |
| Mini-Splatting | 27.32 | .821 | .217 | 0.53 M |
| Speedy-Splat | 26.91 | .781 | .295 | **0.30 M** |
| Taming-3DGS | 27.48 | .794 | .261 | 0.68 M |
| DashGaussian | 27.73 | .817 | .218 | 2.40 M |
| FastGS | 27.56 | .797 | .261 | 0.40 M |
| FastGS-Big | 27.93 | .820 | .216 | 1.15 M |
| **FastGS-Adaptive (ours)** | **28.09** | **.823** | **.211** | 1.58 M |

**Deep Blending and Tanks & Temples**

| Method | DB PSNR | DB SSIM | DB LPIPS | T&T PSNR | T&T SSIM | T&T LPIPS |
|---|---|---|---|---|---|---|
| FastGS | 30.03 | .901 | .270 | 24.15 | .839 | .210 |
| FastGS-Big | 30.12 | .907 | .243 | 24.39 | **.855** | .175 |
| **FastGS-Adaptive (ours)** | **30.31** | **.912** | .244 | **24.49** | **.855** | .180 |

Eight of the nine Mip-NeRF 360 scenes improve over FastGS-Big, for a mean gain of +0.16 dB.

### Ablation (Mip-NeRF 360)

| AESL | HEM-S | ADS | PSNR | SSIM | LPIPS |
|:--:|:--:|:--:|---|---|---|
| | | | 27.95 | .820 | .216 |
| ✓ | | | 28.08 | .822 | .211 |
| ✓ | ✓ | ✓ | **28.09** | **.823** | **.211** |

## Setup

Create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate fastgs
```

The training pipeline depends on three CUDA submodules from the upstream
FastGS project (`diff-gaussian-rasterization_fastgs`, `simple-knn`, and
`fused-ssim`). Clone the upstream repository and install them in editable
mode into the same environment:

```bash
git clone --recursive https://github.com/fastgs/FastGS.git fastgs_upstream
cd fastgs_upstream
pip install ./submodules/diff-gaussian-rasterization_fastgs
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim
cd ..
```

Download the [Mip-NeRF 360 dataset](https://jonbarron.info/mipnerf360/) and
place each scene under `~/data/fastgs/datasets/mipnerf360/`.

## Reproduce

Run all nine scenes end-to-end (training, rendering, metrics):

```bash
bash run_adaptive_benchmark.sh
```

Or train a single scene manually:

```bash
python train.py -s ~/data/fastgs/datasets/mipnerf360/kitchen \
                --eval --densification_interval 500 \
                --adaptive_densification --structural_weight 5.0 \
                --densify_until_iter 18000 \
                --highfeature_lr 0.02 --grad_abs_thresh 0.0002 \
                --model_path output/kitchen_Adaptive
python render.py  -m output/kitchen_Adaptive --skip_train --quiet
python metrics.py -m output/kitchen_Adaptive
```

## Repository Layout

```
arguments/           CLI / config dataclasses
gaussian_renderer/   FastGS rasteriser wrapper + network GUI
lpipsPyTorch/        LPIPS metric (perceptual similarity)
scene/               COLMAP loader, dataset readers, Gaussian model
utils/               loss, edge mask, hard-example mining, densification scoring
train.py             training entry point (FastGS-Adaptive)
render.py            per-scene novel-view rendering
metrics.py           PSNR / SSIM / LPIPS evaluation
compare_final.py     aggregate metrics across scenes
run_adaptive_benchmark.sh   reproduce all nine Mip-NeRF 360 scenes
environment.yml      conda environment specification
```

## License

This project is released under the [Apache 2.0 License](LICENSE).
