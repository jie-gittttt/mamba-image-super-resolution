# MambaIR: Reproduction and Scan-Direction Ablation

**Three of the four scan directions are echoes**

[English](README.md) | [中文](README.zh-CN.md) | [← Repository root](../README.md)

> Stage one of this repository. Full report: [`report.pdf`](report.pdf) (13 pages). Stage two — the MaIR reproduction and CSBlock — is in [`../mair-cnn-ssm/`](../mair-cnn-ssm/).

<p align="center">
  <img src="figures/fig_mambair_architecture.png" alt="MambaIR three-stage architecture">
  <br><em>MambaIR's overall structure: the stack of RSSGs (with the RSSB block inside, panel a), the VSSM that replaces self-attention (panel b), and the 2D selective scan module inside it (panel c). The reproduction lands at 924,492 parameters and 26.52 dB averaged over four benchmarks.</em>
</p>

---

## 1. Background

Single-image super-resolution reconstructs a high-resolution image from one low-resolution observation, and it is an ill-posed inverse problem: many HR images are consistent with the same LR input, so the task is really to learn a prior over natural image structure and use it to pick the right one. Sequence models carrying that prior were long dominated by attention, at O(N²); [MambaIR](https://github.com/csguoh/MambaIR) (ECCV 2024) brought the state-space model in as a direct replacement for self-attention — the architecture stays a SwinIR-style three-stage backbone, only each transformer block becomes an RSSB and self-attention becomes VSSM. What makes this work is complexity: Mamba's selective recurrence is O(N), one hidden state compresses the entire prefix, and the model gets a global receptive field at linear cost — exactly the trade high-resolution restoration wants.

<p align="center">
  <img src="figures/fig1_complexity.png" alt="Linear versus quadratic complexity">
  <br><em>Why an SSM backbone in the first place: O(N) against attention's O(N²), with the gap widening as the sequence grows — and high-resolution images are what makes it grow.</em>
</p>

It does not work cleanly, and that too is because of complexity. State-space models are defined on **1D sequences**, and an image is a 2D grid. MambaIR's answer is **Cross-Scan**: flatten the feature map in four different directions, scan each one, then sum the four results element-wise. The bet is that four 1D orders, added up, approximate the 2D neighbourhood structure the convolution it replaced used to provide for free. **This repository tests that bet directly**, by changing nothing but how many directions the forward pass sums, on frozen weights: dropping from four directions to one costs **0.10 dB on average** across four benchmarks. If the four scans really were four independent views, cutting 75% of them would cost far more than that.

Three reasons explain it, in increasing order of importance: the selective mechanism itself flattens the directional differences (`B(x)`, `C(x)` and `Δ(x)` are all functions of the input, so the same content-filtering process runs down all four orders and the relationships concentrate in whichever direction propagates them best); element-wise addition provides no way to tell them apart (`y = y1 + y2 + y3 + y4` treats the four views as equal-weight, and the sum of several near-identical sequences is just a scalar multiple); and the root cause is that **1D flattening has already destroyed 2D continuity before scanning even begins** — in a 64×64 feature map, vertically adjacent pixels sit 64 tokens apart in row-major order, and the state has to cross 64 unrelated tokens to maintain that relationship. Re-flattening in another order compensates for some of it, but compensating is all it does.

That conclusion is stage two's motivation: if "how the sequence is traversed" is not the bottleneck, then "flattening 2D structure and rebuilding it" itself is — and the local operator should go back directly rather than be compensated for with more scan orders. MaIR takes that route with a single nested S-shaped scan in place of the four straight lines, and stage two builds on it to measure what putting the convolution back is worth.

<p align="center">
  <img src="figures/fig_sr_comparison.jpg" alt="Super-resolution on a real-world photograph">
  <br><em>What this task looks like on a real photograph rather than a bicubic-degraded benchmark: three crops, LQ | MambaIR. The example figure the report opens with.</em>
</p>

---

## 2. Key Features / Contributions

1. **A matched-budget reproduction.** MambaIR-Light ×4, DIV2K, 25,000 training iterations, 924,492 parameters. Every hyperparameter is identical to stage two's runs, so the two can be compared row by row.

2. **A scan-direction ablation that isolates Cross-Scan's redundancy.** Cross-Scan's four-direction sum is that one line at `mambair_arch.py:379-381`. Reducing it to two directions, then to one, and re-running inference is a clean isolation of the scan mechanism: no retraining, and therefore no confound about what a different training regime would have learned.

3. **The finding: four directions to one costs 0.10 dB on average.** 0.005 SSIM. MambaIR bought four scans and got one direction's worth of information.

Inference-only, reusing one trained checkpoint, with the number of directions summed in `forward_core` as the only variable:

| Config | Directions summed | Set5 | Set14 | B100 | Urban100 | avg |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 4-dir (MambaIR default) | forward + backward + transpose + anti-diagonal | 29.29 | 26.73 | 26.30 | 23.74 | **26.52** |
| 2-dir | forward + backward | 29.19 | 26.71 | 26.27 | 23.70 | 26.47 |
| 1-dir | forward only | 29.13 | 26.66 | 26.23 | 23.65 | 26.42 |

| | Set5 | Set14 | B100 | Urban100 | avg |
|---|:---:|:---:|:---:|:---:|:---:|
| Δ 4 → 2 | −0.10 | −0.02 | −0.03 | −0.04 | **−0.05** |
| Δ 4 → 1 | −0.16 | −0.07 | −0.07 | −0.09 | **−0.10** |

Why this is the result is in §4.2; raw logs for every run are in [`logs/`](logs/).

**What is *not* a contribution.** `mambair_arch.py`, `train.py` and `test.py` are all upstream MambaIR and BasicSR code, byte-for-byte identical to their repositories. The config is the upstream lightSR config with a smaller `gt_size` and fewer iterations. Per-file list at the end of this document, under [Repository layout](#repository-layout).

---

## 3. Methodology / Architecture

### 3.1 Base architecture: MambaIR and Cross-Scan

MambaIR as a whole is a SwinIR-style three-stage backbone — shallow feature extraction, a stack of Residual State Space Groups (RSSGs), and a high-quality reconstruction head. Each RSSG is made of RSSBs, and inside an RSSB the path is `LayerNorm → VSSM → LayerNorm → MLP`, VSSM being the module that replaces self-attention.

Cross-Scan happens inside VSSM:

<p align="center">
  <img src="figures/fig_cross_scan_merge.png" alt="Cross-scan, SS blocks, cross-merge">
  <br><em>The VSS module takes an H×W feature map and flattens it four times — along rows, along columns, along both diagonals — runs an SSM independently on each 1D sequence, then folds the four outputs back to 2D and sums them element-wise. The design intent: row scans capture horizontal dependencies, column scans capture vertical ones, the two diagonals fill the gaps, and together the four cover every neighbourhood relationship on the grid.</em>
</p>

That last summation is this repository's entire point of entry: it is one line of code, so changing it needs no retraining, which isolates the *scan mechanism* cleanly from the *training process*. Image credit: this Cross-Scan / S6 / Cross-Merge pipeline is the 2D selective scan (SS2D) diagram from the VMamba paper; the cross-scan scheme originates there, and MambaIR adopts it as-is inside its VSSM.

### 3.2 How the ablation is done

This directory proposes no new module; the original work is a controlled inference-time change. `forward_core` sums the four directions element-wise, so it is replaced with a version that accumulates only the first *k* directions, for *k* = 4, 2, 1, with everything else untouched — the same weights, the same test script, the same `crop_border=4`.

The trade-off is explicit: it isolates the mechanism cleanly, but it also says nothing about what a single-direction model would learn if trained that way. What is compared is one training run's forward pass, not four models. That limitation is written up in §7.

<p align="center">
  <img src="figures/fig_ssm_mechanism.jpg" alt="SSM recurrence after flattening a 2D grid into 1D">
  <br><em>Why the ablation should give this result: an SSM carries h(t) = A·h(t-1) + B·x(t) and emits y(t) = C·h(t), with one hidden state carrying the whole prefix. Flatten the 2D grid, though, and vertically adjacent pixels are pulled far apart in the 1D order (the orange pair), so the recurrence can only spend state capacity representing that distance rather than the relationship.</em>
</p>

### 3.3 Experiment roadmap

| Stage | Question | Variants | Conclusion |
|---|---|---|---|
| 1 | Do Cross-Scan's four directions really each carry information? | 4-dir / 2-dir / 1-dir, forward pass only, frozen weights | dropping to one direction costs 0.10 dB; three are echoes (§4.2) |
| 2 | If traversal order is not the bottleneck, what is? | — handed to [`../mair-cnn-ssm/`](../mair-cnn-ssm/) | the flatten-and-rebuild itself; putting the local operator back is worth +0.27 dB |

---

## 4. Experiments and Results

### 4.1 Setup

**Data.** DIV2K train (800 images), ×4 bicubic degradation, `gt_size = 128`. Evaluation on the Set5, Set14, B100 and Urban100 benchmarks.

**Model.** MambaIR-Light, 924,492 parameters, `embed_dim = 60`.

**Training.** 25,000 iterations (about 30 epochs), Adam (`lr = 2e-4`, `weight_decay = 0`), `batch_size_per_gpu = 96`, seed 10, a single RTX 4090. Horizontal flips and rotations are off — this keeps the comparison with stage two cleaner but also drops a regulariser the upstream configs use.

**Metrics.** Y-channel PSNR / SSIM with `crop_border = 4`. Raw logs in [`logs/`](logs/).

### 4.2 Why the four directions are redundant

The table in §2 only tells half the story. Three reasons, in increasing order of importance:

**The selective mechanism itself flattens the directional differences.** Mamba's `B(x)`, `C(x)` and `Δ(x)` are all functions of the input, which is what separates it from a static SSM like S4. Faced with the same content, the same content-filtering process runs down all four orders: the reconstruction relationships that matter concentrate in whichever direction propagates them best, and the other directions converge to approximations of it. The four directions are not four experts — they are one direction plus three echoes.

**Element-wise addition provides no way to tell them apart.** `y = y1 + y2 + y3 + y4` treats the four views as equal-weight. No gating, no learnable weights, no attention — the model cannot learn to trust one direction more, only to add them up, and the sum of several near-identical sequences is just a scalar multiple.

**1D flattening destroyed 2D continuity long ago.** This is the root cause. In a 64×64 feature map, vertically adjacent pixels sit 64 tokens apart in row-major order, and the state has to cross 64 unrelated tokens to maintain that relationship. Scanning in another order compensates for some of that, but compensating is all it is — which stops being surprising once you see that the flattening happens **before** the scan begins.

### 4.3 Training dynamics and reproduction quality

DIV2K validation PSNR starts at 10.61 dB from random initialisation and reaches 27.19 dB within the first 2,500 iterations — 16.58 dB on 10% of the budget. This is implicit global modelling paying off early: a coarse LR→HR mapping is established almost immediately, and the remaining 22,500 iterations are worth 1.02 dB in total.

<p align="center">
  <img src="figures/training_progression.png" alt="Training progression">
  <br><em>The same crop at different points in training: LR bicubic, random initialisation, and 2.5K / 5K / 10K / 15K / 25K iterations. The steep first half and flat second half of that curve are the visual version of the numbers above.</em>
</p>

### 4.4 Parameter count

The training log prints the model's size as it builds it:

```
INFO: Network: MambaIR, with parameters: 924,492
```

The accompanying report gives MambaIR-Light as having about 1,303,401 parameters (≈1.3M) in six places. That number comes from the upstream MambaIR-S configuration, not from the Light variant actually built here; every figure in this repository follows the **924,492** the log prints.

---

## 5. Tech Stack

| | |
|---|---|
| OS | Ubuntu 22.04 |
| GPU | RTX 4090 (49 GB with VRAM expansion) |
| Python | 3.10 |
| PyTorch | 2.1.0+cu121 |
| TorchVision | 0.16.0+cu121 |
| CUDA | 12.1 |
| BasicSR | 1.3.5 (bundled in the MambaIR fork) |
| Kernel | `mamba-ssm` 1.1.1 + `causal-conv1d` 1.1.1, built with `nvcc` |

Note that this is a **different environment** from [`../mair-cnn-ssm/`](../mair-cnn-ssm/) — that one follows the MaIR fork and uses PyTorch 2.4.0 + BasicSR 1.2.0. The two stages share nothing at runtime and each reproduces its own stage end to end.

---

## 6. Installation and Usage

This directory is not a complete runnable project — it is the subset of files that differs from upstream. Start by cloning the upstream repository.

**1 — Set up the environment.**

```bash
git clone https://github.com/csguoh/MambaIR.git
cd MambaIR
pip install -r requirements.txt          # needs nvcc for the mamba-ssm build
pip install mamba-ssm==1.1.1 causal-conv1d==1.1.1
```

**2 — Drop in this repository's files.**

```bash
cp <this-repo>/mambair/code/mambair_arch.py   basicsr/archs/
cp <this-repo>/mambair/code/train.py          basicsr/train.py
cp <this-repo>/mambair/code/test.py           basicsr/test.py
```

**3 — Train.** Artifacts are written to `./experiments/`.

```bash
python -m basicsr.train -opt <this-repo>/mambair/code/train_MambaIR_lightSR_x4_my.yml
```

**4 — Test on the four benchmarks.** Results are written to `./results/`.

```bash
python -m basicsr.test -opt <this-repo>/mambair/code/test_MambaIR_lightSR_x4_my.yml
```

**5 — Edit the config.** It points at the original machine's `/root/autodl-tmp/datasets/DIV2K/...` paths. Change `dataroot_gt`, `dataroot_lq` and the four test-set roots before running. Run from the root of the cloned upstream project — the relative experiment and results directories resolve against the working directory.

No model checkpoints are included: 25K-iteration weights are ~15 MB each, and the logs plus figures here are the record of the run.

---

## 7. Limitations

- **The ablation is inference-only.** It isolates the scan mechanism cleanly, but says nothing about what a single-direction model would learn if trained that way. What is compared is one training run's forward pass, not four models.
- **One configuration only.** Light variant, DIV2K, ×4, 25K iterations. The redundancy finding is measured here; it is not shown to generalise to larger models or longer training.
- **No texture-complexity metric.** "The residual concentrates on regular high-frequency structure" is read off the figures, not measured.
- **Absolute PSNR is not comparable to the papers.** 25K iterations on 800 images versus 500K on 3,450; the paper's MambaIR-Light sits well above the numbers here.
- **Data augmentation is off.** This keeps it consistent with stage two and with the training-progression figure, but it does depart from the upstream config and gives up some generalisation.

---

## Repository layout

```
mambair/                                          # stage one of mamba-image-super-resolution
├── README.md, README.zh-CN.md
├── report.pdf                                    # 13 pages
├── code/
│   ├── mambair_arch.py                           # upstream MambaIR, unmodified
│   ├── train.py, test.py                         # upstream BasicSR entry points, unmodified
│   └── train_MambaIR_lightSR_x4_my.yml           # the config these runs used
├── figures/                                      # 6 figures, see below
│   ├── fig_mambair_architecture.png              # MambaIR paper Fig. 2
│   ├── fig_cross_scan_merge.png                  # VMamba SS2D diagram
│   ├── fig_ssm_mechanism.jpg                     # 2D→1D flattening and the SSM recurrence
│   ├── fig_sr_comparison.jpg                     # super-resolution on real-world photos
│   ├── fig1_complexity.png                       # O(N) vs O(N²)
│   └── training_progression.png                  # one crop at 7 points in training
└── logs/
    ├── train_MambaIR_lightSR_x4_20260512_133105.log
    └── test_MambaIR_lightSR_x4_benchmark_20260512_194615.log
```

Who wrote each of the files above:

| File | Status |
|---|---|
| `code/mambair_arch.py` | Upstream MambaIR, byte-for-byte identical to upstream. |
| `code/train.py`, `code/test.py` | Upstream BasicSR entry points, unmodified. |
| `code/train_MambaIR_lightSR_x4_my.yml` | Upstream lightSR config with a smaller `gt_size` and fewer iterations. |
| `report.pdf` | The written report: literature review, method, experiments, error analysis. |
| `logs/*` | Raw BasicSR output; the tables in this document are built from it. |
| `figures/fig_mambair_architecture.png`, `fig_cross_scan_merge.png` | Third-party figures reproduced from the MambaIR and VMamba papers. |
| The other 4 in `figures/` | Produced by this run. |

Upstream code comes from [MambaIR](https://github.com/csguoh/MambaIR) and [BasicSR](https://github.com/XPixelGroup/BasicSR), reproduced here under their original licences. Those projects' licences apply to their own files, not to this repository as a whole. Model checkpoints and datasets are not distributed with this repository.

---

## References

- Guo et al. **MambaIR: A Simple Baseline for Image Restoration with State-Space Model.** ECCV 2024.
  ([arXiv:2403.07764](https://arxiv.org/abs/2403.07764))
- Liu et al. **VMamba: Visual State Space Model.** NeurIPS 2024.
  ([arXiv:2401.10166](https://arxiv.org/abs/2401.10166))
- Liang et al. **SwinIR: Image Restoration Using Swin Transformer.** ICCVW 2021.
- Gu & Dao. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** 2023.
