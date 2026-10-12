# MaIR + CSBlock：一个 CNN-SSM 融合模块

[English](README.md) | [中文](README.zh-CN.md) | [← 仓库根目录](../README.zh-CN.md)

**一句话：** MaIR 用「删掉深层特征路径上全部卷积」的方式修好了 MambaIR 的扫描冗余——CSBlock 把局部算子放回来，让一个小 CNN 在同一个残差块内与 VMM 并联，换来 **+0.27 dB**，代价是 **+126% 参数**。

## 1. 项目概述

[MaIR](https://github.com/csguoh/MaIR)（CVPR 2025）正面回应了[上一程](../mambair/)测到的那个问题：MambaIR 的四个 Cross-Scan 方向之所以大体冗余，是因为把二维图像展平成一维，在扫描**开始之前**就已经破坏了空间连续性，而换一种顺序重新展平并不能把它找回来。

MaIR 的解法是 **NSS**（嵌套 S 型扫描）：不用四条独立的线性顺序，只用一条嵌套的 S 型路径。图像被切成互不重叠的水平条带，每条带内按 S 型往返访问，折返把竖直相邻像素的一维距离从 *W* 压到 **2**。一次扫描覆盖整张特征图，两个方向的邻域都保住。**SSA** 接着取代 MambaIR 那句盲目的逐元素相加，换成可学习的通道门控融合。在完全相同的训练条件下，这值 **+1.61 dB**，是本仓库里最大的效应。

代价在架构图上就能看见：MaIR 的残差块是 `LayerNorm → VMM → LayerNorm → MLP`，CAB 一去，**深层特征路径上再没有任何一个卷积**。这是一次刻意的选择——做一个纯 SSM 的视觉模型——但它同时也意味着，网络不得不靠一个归纳偏置是「全局」的机制，去表达局部的、高频的纹理。

CSBlock 就是这样一个假设：这正是 MaIR 在规则纹理上表现受限的原因；以及一个检验它的最小改动。

### 目录结构

```
mair-cnn-ssm/                              # mamba-image-super-resolution 的第二程
├── README.md、README.zh-CN.md
├── report.pdf                             # 14 页
├── code/
│   ├── mair_arch.py                       # 新增 CSBlock（MSS_VMM 类）
│   ├── mair_arch_original.py              # 上游 MaIR，留作 diff
│   ├── train.py                           # 上游 BasicSR 入口
│   └── test.py                            # 上游 BasicSR 入口
├── yaml/
│   ├── train_MaIR_S_lightSR_x4_my.yml     # 基线，25K 迭代
│   ├── test_MaIR_S_lightSR_x4_my.yml
│   ├── train_MaIR_MSS_lightSR_x4.yml      # 带 CSBlock
│   └── test_MaIR_MSS_lightSR_x4.yml
├── figures/
│   ├── fig_mair_architecture.jpg          # MaIR 论文 Fig. 2
│   ├── fig_csblock_architecture.png       # CSBlock：两条流与融合
│   ├── fig_psnr_curve.png                 # 两个模型的训练曲线
│   ├── Set5_img_002_4panel.png            # LR | MaIR | +CSBlock | GT
│   ├── Set14_img_011_4panel.png
│   ├── B100_img_079_4panel.png
│   ├── Urban100_img_043_4panel.png
│   └── Urban100_img_047_4panel.png
└── logs/
    ├── train_MaIR_S_lightSR_x4_20260614_235037.log   # 跑到约 24,400 步中断
    ├── train_MaIR_S_lightSR_x4_20260615_140900.log   # 从 22,500 恢复
    ├── train_MaIR_MSS_lightSR_x4_20260615_192507.log
    ├── test_MaIR_S_lightSR_x4_20260615_145356.log
    └── test_MaIR_MSS_lightSR_x4_20260616_134215.log
```

## 2. 核心贡献

**一、同预算下的 MaIR 复现。** MaIR-S-Light ×4，DIV2K 上训练 25,000 次迭代，1,374,156 参数——所有超参与上一程的 MambaIR 运行完全一致，所以 +1.61 dB 是同一把尺子量出来的，而不是两个论文数字相减。

**二、CSBlock——一个 CNN-SSM 融合模块，46 行。** 一个小 CNN **并联在同一个残差块内的 VMM 旁边**。两条流拿到的都是这个块的**原始输入**，不存在一条喂给另一条的关系，所以局部分支看到的是未经处理的信号，而不是等 SSM 把它抹平之后再去补救。两路输出按通道拼接、线性投影融合。

**三、增益正好落在设计预言的地方。** **平均 +0.27 dB，四比四全胜，没有一处退化。** Urban100 上（+0.42，规则建筑与重复结构）的增益是 B100（+0.13，自然随机纹理）的 3 倍。训练早期收敛也快约 2 倍。

**四、一个被纠正的效率说法。** 报告（以及 §3.2 那张架构图的右下角）说 CSBlock「仅新增 5.5% 参数」。实际是 **+126%**。§4.4 有这笔账。

**什么不算贡献。** `mair_arch.py` 里除 `MSS_VMM` 类之外的全部内容、`train.py`、`test.py`、四份 YAML 配置，都是上游 MaIR / BasicSR 代码。真正属于这里的原创工作，就是那 46 行的 diff。

## 3. 方法与架构

### 3.1 基线：MaIR

![MaIR 架构](figures/fig_mair_architecture.jpg)

*（a）三阶段整体骨干——提取、残差 Mamba 组、重建。（b）RMB：`LayerNorm → VMM → LayerNorm → MLP`，不含卷积。（c）VMM，NSS 就在这里面。来源：MaIR 论文 Fig. 2。*

与上一程 MambaIR 的对照值得摆明：

| | MambaIR | MaIR |
|---|---|---|
| 扫描路径 | Cross-Scan：4 条独立直线 | NSS：一条嵌套 S 型路径 |
| 垂直连续性 | 靠额外扫描来补偿 | 靠折返天然保留（一维距离 → 2） |
| SSM 调用 | 每块 4 次 | 每块 1 次 |
| 融合方式 | 四方向逐元素相加 | 单路径，无需融合 |
| 注意力 | 无 | SSA 通道混洗门控 |
| 前馈 | CAB（Conv2d + 通道注意力） | **MLP（纯线性层）** |

最后一行是切入点。MambaIR 的前馈块里含一个卷积。MaIR 把它换成了线性 MLP，也就顺手把深层路径上最后一个局部算子拿掉了。

### 3.2 CSBlock

![CSBlock 架构](figures/fig_csblock_architecture.png)

两条流，一个块，吃的是同一个输入：

- **VMM（SSM 流）：** `in_proj → permute + Conv1d + SiLU → NSS 扫描 + 选择性扫描 → chunk/求和/门控 → out_norm + gate(z) + out_proj`。与上游一致，SS2D 核没有改。
- **CNN（局部流）：** `permute → BCHW → Conv3×3 + SiLU → Conv3×3 + SiLU → permute → BHWC`。`padding=1`，不改变分辨率。
- **融合：** 按通道拼接（`2C`）→ `Linear(2C → C)` → `LayerNorm`。

落到代码上，这个模块是 `mair_arch.py:346` 处一个叫 `MSS_VMM` 的类，由三部分组成：上游的 `VMM`、一个两层 `nn.Sequential` 卷积流，以及 `Linear(2C → C)` 加 `LayerNorm` 的融合头。完整源码见 [`code/mair_arch.py`](code/mair_arch.py)，同目录的 [`code/mair_arch_original.py`](code/mair_arch_original.py) 可直接对读 diff。

*CSBlock 是模块名，CNN-SSM 是它实现的融合方案。* 类名叫 `MSS_VMM`，打开它的配置叫 `MaIR_MSS`，日志里把训出来的模型记作 MaIR-CSB。

三个细节撑起了这个设计：

- **CNN 分支读的是这个块的输入，不是 SSM 的输出。** 这使它成为真正并行的第二条流，而不是一个残差补丁——也正是这个模块被称作「层内」而非事后补救的原因。
- **卷积是完整 3×3（`groups=1`），不是深度可分离。** 深度可分离能把开销再砍掉约 32 倍，但它不做通道间混合，而通道混合正是这里要求局部算子承担的一部分。
- **VMM 原样不动。** CSBlock 加什么都是在 SSM 旁边加，所以任何退化都不可能归因到扫描本身被改动。

在 `RMB.__init__` 里一行替换掉 `VMM` 即可。相对 `mair_arch_original.py` 是**新增 46 行、改动 1 行**，原文件一并留着，这个 diff 可以直接读。

## 4. 实验与结果

### 4.1 设置

与 [`../mambair/`](../mambair/) 完全相同的条件：DIV2K 训练集（800 张）、×4 双三次退化、`gt_size=128`、`batch_size_per_gpu=96`、`total_iter=25000`、Adam（`lr=2e-4`、`weight_decay=0`）、`embed_dim=60`、seed 10、单张 RTX 4090，关闭翻转与旋转增强。评测为 Y 通道 PSNR / SSIM，`crop_border=4`。

本目录下的运行：

| 模型 | 配置 | 迭代数 | 参数量 |
|---|---|---|---:|
| MaIR-S-Light | `train_MaIR_S_lightSR_x4_my.yml` | 25,000 | 1,374,156 |
| MaIR-CSB | `train_MaIR_MSS_lightSR_x4.yml` | 25,000 | 3,109,356 |

基线分两次训练：第一次在约 24,400 步中断，第二次从 22,500 步的 checkpoint 恢复跑完——所以有两个 `train_MaIR_S_*` 日志，而 `test_MaIR_S_*` 只有一个。

### 4.2 结果

| 测试集 | MambaIR（复现） | MaIR（复现） | **+ CSBlock** | MaIR（原文） | CSBlock 距原文 |
|---|:---:|:---:|:---:|:---:|:---:|
| Set5 | 29.29 / 0.837 | 31.64 / 0.886 | **31.98 / 0.891** | 32.48 / 0.899 | −0.50 |
| Set14 | 26.73 / 0.723 | 28.22 / 0.773 | **28.42 / 0.778** | 28.81 / 0.786 | −0.39 |
| B100 | 26.30 / 0.679 | 27.34 / 0.729 | **27.47 / 0.733** | 27.71 / 0.741 | −0.24 |
| Urban100 | 23.74 / 0.689 | 25.29 / 0.761 | **25.71 / 0.775** | 26.60 / 0.801 | −0.89 |
| **平均** | 26.52 | 28.12 | **28.40** | 28.90 | −0.50 |

### 4.3 增益从哪来

| | Set5 | Set14 | B100 | Urban100 | 平均 |
|---|:---:|:---:|:---:|:---:|:---:|
| MambaIR → MaIR | +2.35 | +1.49 | +1.04 | +1.55 | **+1.61** |
| MaIR → +CSBlock | +0.34 | +0.20 | +0.13 | **+0.42** | **+0.27** |

排序本身就是论据。Urban100 是一批规则建筑——砖缝、栏杆、窗格、立面重复。B100 是自然随机纹理。如果 CSBlock 起作用的机制是「补回一个局部算子」，那么它的增益就该随图中局部周期结构的多少而变——事实正是如此：**Urban100 是 B100 的 3.2 倍**，而且四比四全胜、无一处退化，这排除了增益来自噪声的可能。

![PSNR 曲线](figures/fig_psnr_curve.png)

收敛曲线给出相同的答案。CSBlock 约 10000 步突破 30 dB，MaIR 要 ~22000 步——训练早期大约快 2 倍——等 SSM 分支收敛后，差距收窄到稳定的约 0.26 dB。局部算子似乎早期廉价好拟合，等全局模型追上来之后价值变小；那剩下的 0.26 dB 就是它在平衡态下的价值。

### 4.4 参数量

训练日志在构建模型时打印了它们的规模：

```
INFO: Network: MaIR, with parameters: 1,374,156     # MaIR-S 基线
INFO: Network: MaIR, with parameters: 3,109,356     # 带 CSBlock
```

| | 参数量 |
|---|---:|
| MaIR-S-Light | 1,374,156 |
| MaIR + CSBlock（MaIR-CSB） | 3,109,356 |
| **差值** | **+1,735,200（+126%）** |

单块：2 个 3×3 卷积（2 × 32,460）+ 融合层 `Linear(2C, C)`（7,260）+ `LayerNorm`（120）= **72,300**，× 24 块 = 1,735,200——与日志给出的差值完全吻合，这同时验证了单块数字与块数。

> **关于 5.5% 这个数字。** 书面报告里写 CSBlock「仅新增 5.5% 参数」，§3.2 那张架构图的右下角也印着同样的说法。这个数其实是**单块的** 72,300 除以**整个基础模型**（72,300 ÷ 1,374,156 ≈ 5.3%），只在这个「每块」的意义上成立。模型层面的真实增幅是 **+126%**。+0.27 dB 是真的，设计本身也站得住，但那套效率表述差了 24 倍，上表是纠正后的版本。

所以结论是：**+0.27 dB 换 +126% 参数，增益集中在规则纹理上。** 作为一篇论文式的贡献，这个交换很不划算。它真正立住的是一件更窄也更有用的事——定位了纯 SSM 视觉骨干的一处具体缺陷，说明一个 3×3 的感受野就足以应对它，并且证明效应最大的地方恰好是理论预期的那个地方。

### 4.5 视觉对比

`LR（双三次） | MaIR | MaIR + CSBlock | GT`，保持原始分辨率：

| Set5（`img_002`） | Set14（`img_011`） |
|---|---|
| ![Set5](figures/Set5_img_002_4panel.png) | ![Set14](figures/Set14_img_011_4panel.png) |

| B100（`img_079`） | Urban100（`img_043`） |
|---|---|
| ![B100](figures/B100_img_079_4panel.png) | ![Urban100](figures/Urban100_img_043_4panel.png) |

![Urban100 img_047](figures/Urban100_img_047_4panel.png)

*Urban100 `img_047`，两张 Urban100 裁片里的第二张——+0.42 dB 在这里最看得出来。*

这些图刻意不做压缩：它们的作用是展示像素级的纹理差异，重采样就失去意义了。

## 5. 技术栈

| | |
|---|---|
| 操作系统 | Ubuntu 22.04 |
| 显卡 | RTX 4090 |
| Python | 3.10 |
| PyTorch | 2.4.0+cu121 |
| TorchVision | 0.19.0+cu121 |
| CUDA | 12.1 |
| BasicSR | 1.2.0+0051603（随 MaIR 分支提供） |
| 算子库 | `mamba-ssm` 1.1.1 + `causal-conv1d` 1.1.1，用 `nvcc` 编译 |

注意这与 [`../mambair/`](../mambair/) 是**两套不同的环境**——那边跟着 MambaIR 分支走，用 PyTorch 2.1.0 + BasicSR 1.3.5。两个阶段运行时不共享任何东西，各自端到端复现各自的阶段。

## 6. 安装与使用

本目录不是完整的可运行工程——它是与上游有差异的那部分文件。先克隆上游仓库。

```bash
git clone https://github.com/csguoh/MaIR.git
cd MaIR
pip install -r requirements.txt          # 编译 mamba-ssm 需要 nvcc
pip install mamba-ssm==1.1.1 causal-conv1d==1.1.1
```

再放入架构文件：

```bash
cp <this-repo>/mair-cnn-ssm/code/mair_arch.py basicsr/archs/mair_arch.py
```

### 训练与测试 MaIR-CSB（带 CSBlock）

```bash
python -m basicsr.train -opt <this-repo>/mair-cnn-ssm/yaml/train_MaIR_MSS_lightSR_x4.yml
python -m basicsr.test  -opt <this-repo>/mair-cnn-ssm/yaml/test_MaIR_MSS_lightSR_x4.yml
```

### 训练与测试 MaIR-S 基线

先换回未修改的架构文件：

```bash
cp <this-repo>/mair-cnn-ssm/code/mair_arch_original.py basicsr/archs/mair_arch.py
python -m basicsr.train -opt <this-repo>/mair-cnn-ssm/yaml/train_MaIR_S_lightSR_x4_my.yml
python -m basicsr.test  -opt <this-repo>/mair-cnn-ssm/yaml/test_MaIR_S_lightSR_x4_my.yml
```

两个文件都留着的意义就是读这个 diff：

```bash
diff <this-repo>/mair-cnn-ssm/code/mair_arch_original.py \
     <this-repo>/mair-cnn-ssm/code/mair_arch.py
```

**四份配置都必须改。** 它们都指向原机器上 `/root/autodl-tmp/...` 的数据集路径和 `/root/2025-CVPR-MaIR-main/experiments/...` 的权重路径。运行前先改 `dataroot_gt`、`dataroot_lq`、四个测试集根目录，以及 `path.pretrain_network_g` 与结果输出目录。

仓库不含权重——`net_g_25000.pth` 约 15 MB，日志与配图才是这几次运行的记录。

## 7. 局限

- **只有一种配置。** Light 变体、DIV2K、×4、25K 迭代。CSBlock 在更大模型、DF2K、去噪与去模糊上是什么表现，这里一个字都说明不了。
- **结构是手工定的**——两层 3×3 卷积、通道拼接、线性融合。没有做结构搜索，也没试过替代方案（深度可分离、单层卷积、注意力融合、串行而非并行摆放）。
- **+126% 参数换 +0.27 dB 很贵。** §4.4 那笔账就是这件事要说两遍的原因。
- **「规则纹理」是定性描述。** Urban100 > Set5 > Set14 > B100 的梯次一致且符合假设，但没有计算任何纹理复杂度指标去预测增益——机制是从排序论证出来的，不是测出来的。
- **没有做摆放位置的消融。** 「读输入而不是读 SSM 输出」这一点是从设计上论证的，没有拿一个读 SSM 输出的变体做对照。这是最明显的下一个实验。
- **绝对 PSNR 不能和论文比。** 800 张跑 25K 迭代，对 3450 张跑 500K；与论文 MaIR 之间 −0.24…−0.89 dB 的差距是训练预算的差距，不是复现失败。
- **SSIM 报了但没有单独分析。** 它的走势紧跟 PSNR，这里不对它作任何额外论断。

## 参考文献

- Li et al. **MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration.** CVPR 2025.
  ([arXiv:2412.20066](https://arxiv.org/abs/2412.20066))
- Guo et al. **MambaIR: A Simple Baseline for Image Restoration with State-Space Model.** ECCV 2024.
- Liu et al. **VMamba: Visual State Space Model.** NeurIPS 2024.
- Liang et al. **SwinIR: Image Restoration Using Swin Transformer.** ICCVW 2021.
- Gu & Dao. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** 2023.
