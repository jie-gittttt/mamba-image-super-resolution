# MaIR + CSBlock: a CNN-SSM fusion module

[English](README.md) | [中文](README.zh-CN.md) | [← Repository root](../README.md)

**One sentence:** MaIR fixed MambaIR's scan redundancy by removing every convolution from
the deep feature path — CSBlock puts a local operator back, running a CNN in parallel with
the VMM inside the same residual block, for **+0.27 dB** at **+126% parameters**.

## 1. Overview

[MaIR](https://github.com/csguoh/MaIR) (CVPR 2025) is the direct answer to the problem the
[previous stage](../mambair/) measured: MambaIR's four Cross-Scan directions are largely
redundant because flattening a 2D image into 1D destroys spatial continuity *before* the
scan starts, and re-flattening in another order does not restore it.

MaIR's fix is **NSS** (Nested S-shaped Scanning): instead of four independent linear
orderings, one nested S-shaped path. The image is divided into non-overlapping horizontal
strips; each strip is walked in a boustrophedon; the folds shrink the 1D distance between
vertically adjacent pixels from *W* to **2**. One scan covers the whole feature map and
preserves both directions of neighbourhood. **SSA** then replaces MambaIR's blind
element-wise sum with a learnable channel-gating fusion. Under identical training terms
that is worth **+1.61 dB**, the largest effect in this repository.

The cost is visible in the architecture: MaIR's residual block is `LayerNorm → VMM →
LayerNorm → MLP`, and with CAB gone there is **no convolution anywhere in the deep
feature path**. That is a deliberate choice — a pure-SSM vision model — but it also means
the network has to represent local, high-frequency texture using a mechanism whose
inductive bias is global.

CSBlock is the hypothesis that this is what caps MaIR's performance on regular texture,
and a minimal way to test it.

### Folder layout

```
mair-cnn-ssm/                              # stage 2 of mamba-image-super-resolution
├── README.md, README.zh-CN.md
├── report.pdf                             # 14 pages
├── code/
│   ├── mair_arch.py                       # + CSBlock (the MSS_VMM class)
│   ├── mair_arch_original.py              # upstream MaIR, kept for diffing
│   ├── train.py                           # upstream BasicSR entry point
│   └── test.py                            # upstream BasicSR entry point
├── yaml/
│   ├── train_MaIR_S_lightSR_x4_my.yml     # baseline, 25K iterations
│   ├── test_MaIR_S_lightSR_x4_my.yml
│   ├── train_MaIR_MSS_lightSR_x4.yml      # + CSBlock
│   └── test_MaIR_MSS_lightSR_x4.yml
├── figures/
│   ├── fig_mair_architecture.jpg          # MaIR paper Fig. 2
│   ├── fig_csblock_architecture.png       # CSBlock: both streams, fusion
│   ├── fig_psnr_curve.png                 # training curves, both models
│   ├── Set5_img_002_4panel.png            # LR | MaIR | +CSBlock | GT
│   ├── Set14_img_011_4panel.png
│   ├── B100_img_079_4panel.png
│   ├── Urban100_img_043_4panel.png
│   └── Urban100_img_047_4panel.png
└── logs/
    ├── train_MaIR_S_lightSR_x4_20260614_235037.log   # interrupted at ~24,400
    ├── train_MaIR_S_lightSR_x4_20260615_140900.log   # resumed from 22,500
    ├── train_MaIR_MSS_lightSR_x4_20260615_192507.log
    ├── test_MaIR_S_lightSR_x4_20260615_145356.log
    └── test_MaIR_MSS_lightSR_x4_20260616_134215.log
```

## 2. Key Features / Contributions

**1. A matched-budget MaIR reproduction.** MaIR-S-Light ×4 on DIV2K, 25,000 iterations,
1,374,156 parameters — every hyperparameter identical to the previous stage's MambaIR run,
so the +1.61 dB is a like-for-like comparison rather than two papers' numbers.

**2. CSBlock — a CNN-SSM fusion module, 46 lines.** A small CNN runs *in parallel with the
VMM inside the same residual block*. Both streams receive the block's **original input**,
not one feeding the other, so the local branch sees the unprocessed signal instead of
trying to recover it after the SSM has already smoothed it. Outputs are concatenated and
fused by a linear projection.

**3. Improvement concentrated where the design predicts.** **+0.27 dB average, 4-of-4 wins,
no regression anywhere.** The gain is 3× larger on Urban100 (+0.42) — regular architecture
and repeating structure — than on B100 (+0.13), which is natural stochastic texture. It
also converges ~2× faster in early training.

**4. An efficiency claim, corrected.** The report (and the architecture diagram in §3)
states CSBlock adds "only 5.5% parameters". It adds **+126%**. §4.4 shows the arithmetic.

**What is not a contribution.** All of `mair_arch.py` except the `MSS_VMM` class,
`train.py`, `test.py`, and the four YAML configs are upstream MaIR / BasicSR code. The
46-line diff is the entirety of the original work here.

## 3. Methodology / Architecture

### 3.1 The baseline: MaIR

![MaIR architecture](figures/fig_mair_architecture.jpg)

*(a) the overall three-stage backbone — extraction, Residual Mamba Groups, reconstruction.
(b) the RMB: `LayerNorm → VMM → LayerNorm → MLP`, no convolution. (c) the VMM, which is
where NSS lives. Source: MaIR paper, Fig. 2.*

The comparison with the previous stage's MambaIR is worth stating plainly:

| | MambaIR | MaIR |
|---|---|---|
| Scan path | Cross-Scan: 4 independent linear directions | NSS: one nested S-shaped path |
| Vertical continuity | recovered by extra scans | preserved by the folds (1D distance → 2) |
| SSM calls | 4 per block | 1 per block |
| Fusion | element-wise sum of 4 directions | single path, no fusion |
| Attention | none | SSA channel-shuffle gating |
| Feed-forward | CAB (Conv2d + channel attention) | **MLP (pure linear)** |

That last row is the opening. MambaIR's feed-forward block contained a convolution. MaIR
replaced it with a linear MLP, and in doing so removed the last local operator from the
deep path.

### 3.2 CSBlock

![CSBlock architecture](figures/fig_csblock_architecture.png)

Two streams, one block, both fed the same input:

- **VMM (SSM stream):** `in_proj → permute + Conv1d + SiLU → NSS scan + selective scan →
  chunk/sum/gate → out_norm + gate(z) + out_proj`. Unchanged from upstream — the SS2D
  kernel is not touched.
- **CNN (local stream):** `permute → BCHW → Conv3×3 + SiLU → Conv3×3 + SiLU → permute →
  BHWC`. `padding=1`, so no resolution change.
- **Fusion:** concatenate along channels (`2C`), `Linear(2C → C)`, `LayerNorm`.

In code terms the module is a class called `MSS_VMM` at `mair_arch.py:346`, with three
sub-modules: the upstream `VMM`, a two-layer `nn.Sequential` CNN, and a `Linear(2C → C)`
plus `LayerNorm` fusion head. The full source is in
[`code/mair_arch.py`](code/mair_arch.py), with
[`code/mair_arch_original.py`](code/mair_arch_original.py) next to it for diffing.

*CSBlock is the module; CNN-SSM is the fusion scheme it implements.* The class is
`MSS_VMM`, the config that enables it is `MaIR_MSS`, and the logs tag the resulting model
MaIR-CSB.

Three details carry the design:

- **The CNN branch reads `x`, the block's input, not `ssm_out`.** This is what makes it a
  parallel stream rather than a residual correction, and it is why the module is
  described as *in-layer* rather than post-hoc.
- **The convolutions are full 3×3 (`groups=1`), not depthwise.** Depthwise would have cut
  the cost by ~32×, but a depthwise convolution does not mix channels, and channel mixing
  is part of what a local operator has to provide here.
- **The VMM is untouched.** Whatever CSBlock adds, it adds alongside the SSM, so a
  regression cannot be attributed to a change in the scan.

It is swapped in for `VMM` in one line of `RMB.__init__`. Against `mair_arch_original.py`
it is **46 added lines and 1 changed line**, and the original is kept alongside so that
diff can be read directly.

## 4. Experiments and Results

### 4.1 Setup

Identical terms to [`../mambair/`](../mambair/): DIV2K train (800 images), ×4 bicubic
degradation, `gt_size=128`, `batch_size_per_gpu=96`, `total_iter=25000`, Adam
(`lr=2e-4`, `weight_decay=0`), `embed_dim=60`, seed 10, single RTX 4090, no flip/rotation
augmentation. Evaluation is Y-channel PSNR / SSIM with `crop_border=4`.

Runs in this folder:

| Model | Config | Iterations | Params |
|---|---|---|---:|
| MaIR-S-Light | `train_MaIR_S_lightSR_x4_my.yml` | 25,000 | 1,374,156 |
| MaIR-CSB | `train_MaIR_MSS_lightSR_x4.yml` | 25,000 | 3,109,356 |

The baseline was trained in two sessions — the first was interrupted near iteration 24,400
and the second resumed from the 22,500-iteration checkpoint, which is why there are two
`train_MaIR_S_*` logs and only one `test_MaIR_S_*`.

### 4.2 Results

| Test set | MambaIR (repro) | MaIR (repro) | **+ CSBlock** | MaIR (paper) | CSBlock gap to paper |
|---|:---:|:---:|:---:|:---:|:---:|
| Set5 | 29.29 / 0.837 | 31.64 / 0.886 | **31.98 / 0.891** | 32.48 / 0.899 | −0.50 |
| Set14 | 26.73 / 0.723 | 28.22 / 0.773 | **28.42 / 0.778** | 28.81 / 0.786 | −0.39 |
| B100 | 26.30 / 0.679 | 27.34 / 0.729 | **27.47 / 0.733** | 27.71 / 0.741 | −0.24 |
| Urban100 | 23.74 / 0.689 | 25.29 / 0.761 | **25.71 / 0.775** | 26.60 / 0.801 | −0.89 |
| **Average** | 26.52 | 28.12 | **28.40** | 28.90 | −0.50 |

### 4.3 Where the gain comes from

| | Set5 | Set14 | B100 | Urban100 | avg |
|---|:---:|:---:|:---:|:---:|:---:|
| MambaIR → MaIR | +2.35 | +1.49 | +1.04 | +1.55 | **+1.61** |
| MaIR → +CSBlock | +0.34 | +0.20 | +0.13 | **+0.42** | **+0.27** |

The ordering is the argument. Urban100 is a corpus of regular architecture — brickwork,
railings, window grids, facade repetition. B100 is natural stochastic texture. If CSBlock
helps by restoring a local operator, its gain should scale with how much locally periodic
structure there is to catch, and it does: **3.2× more on Urban100 than on B100**, and
4-of-4 wins with no regression, which is what rules out the gain being noise.

![PSNR curve](figures/fig_psnr_curve.png)

Convergence agrees. CSBlock breaks 30 dB around iteration 10,000 where MaIR takes ~22,000
— roughly a 2× early-training speedup — and the gap narrows to a stable ≈0.26 dB once the
SSM branch has converged. A local operator is apparently cheap to fit early and worth less
once the global model has caught up; the residual 0.26 dB is what it is worth at
equilibrium.

### 4.4 Parameter accounting

The training logs print the models as they are built:

```
INFO: Network: MaIR, with parameters: 1,374,156     # MaIR-S baseline
INFO: Network: MaIR, with parameters: 3,109,356     # + CSBlock
```

| | Params |
|---|---:|
| MaIR-S-Light | 1,374,156 |
| MaIR + CSBlock (MaIR-CSB) | 3,109,356 |
| **Difference** | **+1,735,200 (+126%)** |

Per block: 2 × 3×3 conv (2 × 32,460) + fusion `Linear(2C, C)` (7,260) + `LayerNorm` (120)
= **72,300**, × 24 blocks = 1,735,200 — exactly the difference the logs report, which
confirms both the per-block figure and the block count.

> **A note on the 5.5% figure.** The written report states that CSBlock adds "only 5.5%
> parameters", and the architecture diagram in §3.2 carries the same claim in its bottom
> corner. That number is one block's 72,300 measured against the *whole* base model
> (72,300 ÷ 1,374,156 ≈ 5.3%), and it is correct only in that per-block sense. The
> model-level increase is **+126%**. The +0.27 dB is real and the design stands on its own,
> but the efficiency framing is off by a factor of 24, and the table above is the corrected
> version.

So: **+0.27 dB for +126% parameters, gain concentrated in regular texture.** As a
paper-style contribution that is a poor trade. What it does establish is narrower and more
useful — it locates a specific deficiency in pure-SSM vision backbones, shows that a 3×3
receptive field is enough to address it, and shows the effect is largest exactly where the
theory says it should be.

### 4.5 Visual comparison

`LR (bicubic) | MaIR | MaIR + CSBlock | GT`, native resolution:

| Set5 (`img_002`) | Set14 (`img_011`) |
|---|---|
| ![Set5](figures/Set5_img_002_4panel.png) | ![Set14](figures/Set14_img_011_4panel.png) |

| B100 (`img_079`) | Urban100 (`img_043`) |
|---|---|
| ![B100](figures/B100_img_079_4panel.png) | ![Urban100](figures/Urban100_img_043_4panel.png) |

![Urban100 img_047](figures/Urban100_img_047_4panel.png)

*Urban100 `img_047`, the second of the two Urban100 crops — where the +0.42 dB is visible.*

The panels are deliberately not downscaled: they exist to show pixel-level texture
differences, and resampling them would defeat the point.

## 5. Tech Stack

| | |
|---|---|
| OS | Ubuntu 22.04 |
| GPU | RTX 4090 |
| Python | 3.10 |
| PyTorch | 2.4.0+cu121 |
| TorchVision | 0.19.0+cu121 |
| CUDA | 12.1 |
| BasicSR | 1.2.0+0051603 (bundled in the MaIR fork) |
| Kernels | `mamba-ssm` 1.1.1 + `causal-conv1d` 1.1.1, built with `nvcc` |

Note this is a **different environment** from [`../mambair/`](../mambair/), which follows
the MambaIR fork and runs PyTorch 2.1.0 + BasicSR 1.3.5. The two stages share no runtime;
each reproduces its own stage end to end.

## 6. Installation and Usage

This folder is not a complete runnable tree — it is the subset of files that differs from
upstream. Clone the upstream repository first.

```bash
git clone https://github.com/csguoh/MaIR.git
cd MaIR
pip install -r requirements.txt          # needs nvcc for the mamba-ssm build
pip install mamba-ssm==1.1.1 causal-conv1d==1.1.1
```

Then drop the architecture file in:

```bash
cp <this-repo>/mair-cnn-ssm/code/mair_arch.py basicsr/archs/mair_arch.py
```

### Train and test MaIR-CSB (with CSBlock)

```bash
python -m basicsr.train -opt <this-repo>/mair-cnn-ssm/yaml/train_MaIR_MSS_lightSR_x4.yml
python -m basicsr.test  -opt <this-repo>/mair-cnn-ssm/yaml/test_MaIR_MSS_lightSR_x4.yml
```

### Train and test the MaIR-S baseline

Swap in the unmodified architecture file first:

```bash
cp <this-repo>/mair-cnn-ssm/code/mair_arch_original.py basicsr/archs/mair_arch.py
python -m basicsr.train -opt <this-repo>/mair-cnn-ssm/yaml/train_MaIR_S_lightSR_x4_my.yml
python -m basicsr.test  -opt <this-repo>/mair-cnn-ssm/yaml/test_MaIR_S_lightSR_x4_my.yml
```

Reading the diff between the two is the whole point of keeping both files:

```bash
diff <this-repo>/mair-cnn-ssm/code/mair_arch_original.py \
     <this-repo>/mair-cnn-ssm/code/mair_arch.py
```

**The configs need editing.** All four point at the original machine's
`/root/autodl-tmp/...` dataset paths and `/root/2025-CVPR-MaIR-main/experiments/...`
checkpoint paths. Change `dataroot_gt`, `dataroot_lq`, the test-set roots, and the
`path.pretrain_network_g` / results roots before running.

No checkpoints are included — `net_g_25000.pth` is ~15 MB and the logs and figures here
are the record of the runs.

## 7. Limitations

- **One configuration.** Light variant, DIV2K, ×4, 25K iterations. Nothing here says what
  CSBlock does on a larger model, on DF2K, or on denoising and deblurring.
- **The design was chosen by hand** — two 3×3 convolutions, channel concatenation, linear
  fusion. No search over that space, and the alternatives (depthwise, single conv,
  attention fusion, sequential rather than parallel placement) were not tried.
- **+126% parameters is a large price** for +0.27 dB. The parameter accounting in §4.4 is
  the reason this is stated twice.
- **"Regular texture" is a qualitative label.** The Urban100 > Set5 > Set14 > B100 ordering
  is consistent and matches the hypothesis, but no texture-complexity metric was computed
  to predict the gain — the mechanism is argued from the ordering, not measured.
- **No ablation of the placement.** The claim that reading `x` rather than `ssm_out` matters
  is argued from the design, not tested against a variant that reads the SSM output. That
  is the most obvious next experiment.
- **Absolute PSNR is not comparable to the paper.** 25K iterations on 800 images versus
  500K on 3,450; the −0.24…−0.89 dB gap to the published MaIR numbers is a training-budget
  gap, not a reproduction failure.
- **SSIM is reported but not analysed.** The gains there track PSNR closely and no separate
  claim is made about them.

## References

- Li et al. **MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration.** CVPR 2025.
  ([arXiv:2412.20066](https://arxiv.org/abs/2412.20066))
- Guo et al. **MambaIR: A Simple Baseline for Image Restoration with State-Space Model.** ECCV 2024.
- Liu et al. **VMamba: Visual State Space Model.** NeurIPS 2024.
- Liang et al. **SwinIR: Image Restoration Using Swin Transformer.** ICCVW 2021.
- Gu & Dao. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** 2023.
