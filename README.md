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
<a href="LICENSE">
<img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge&logo=apache&logoColor=white">
</a>
</p>

**Anonymous Authors**

*Preprint, 2026 (Under Review)*

</div>

---

## Architecture

<div align="center">
<img src="static/images/architecture.png" width="95%">
<br>
<em>Model architecture of FastGS-Adaptive with four additive modules (AESL, HEM-S, ADS, PSA-VCD) that make densification structure-aware and view-consistent.</em>
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

3D Gaussian Splatting (3DGS) has established itself as a leading representation for real-time novel view synthesis. Recent acceleration frameworks, such as FastGS have significantly reduced per-scene training time through multi-view consistent densification and pruning. However, FastGS employs a structure-agnostic *L<sub>1</sub>* error metric to guide densification. Specifically, the scoring head evaluates each pixel solely by photometric residual magnitude, without distinguishing residuals on geometric edges and those that arise on textureless surfaces. To address this limitation, we propose **FastGS-Adaptive**, a structure-aware densification framework that introduces four additive mechanisms: (1) an *Adaptive Edge-Structured Loss* (AESL) that modulates the photometric penalty using Sobel-derived gradient priors; (2) a *Hard Example Mining Sampler* (HEM-S) that replaces uniform camera selection with a loss-history-biased policy to focus densification capacity on under-reconstructed views; (3) an *Annealed Densification Schedule* (ADS) that decays the split frequency across three phases to eliminate late-stage overhead; and (4) a *Perceptual Structure-Aware VCD score* (PSA-VCD) that augments the *L<sub>1</sub>*-based importance map with edge-weighted local SSIM error to bias split decisions toward perceptually salient regions. Experimental evaluation demonstrates that FastGS-Adaptive surpasses existing state-of-the-art acceleration methods on all three standard quality metrics while also rendering substantially faster at inference.

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
utils/               loss, edge mask, hard-example mining, PSA-VCD scoring
train.py             training entry point (FastGS-Adaptive)
render.py            per-scene novel-view rendering
metrics.py           PSNR / SSIM / LPIPS evaluation
compare_final.py     aggregate metrics across scenes
run_adaptive_benchmark.sh   reproduce all nine Mip-NeRF 360 scenes
environment.yml      conda environment specification
```

## License

This project is released under the [Apache 2.0 License](LICENSE).
