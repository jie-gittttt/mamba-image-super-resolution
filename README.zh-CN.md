# 基于状态空间模型的图像超分辨率

[English](README.md) | [中文](README.zh-CN.md)

**一句话：** 在完全相同的 ×4 训练预算下复现 MambaIR（ECCV 2024）与 MaIR（CVPR 2025），证明 MambaIR 四个扫描方向里有三个几乎不携带信息，再把 MaIR 在改进过程中丢掉的局部算子补回来——一个叫 **CSBlock** 的 CNN-SSM 融合模块，换来 +0.27 dB，增益集中在规则纹理上。

## 1. 项目概述

单图像超分辨率（SR）从一张低分辨率观测重建高分辨率图像。它是一个病态逆问题：同一张 LR 输入可以对应很多张合理的 HR 图像，所以任务本质是学一个关于自然图像结构的先验，用它把正确的那一张挑出来。

本仓库关心的架构问题是：承载这个先验的序列模型该选哪一个。Transformer 用两两注意力建模全图，复杂度 O(N²)；循环模型是线性的，但受限于固定大小的隐状态。状态空间模型（SSM）——尤其是 Mamba——声称两者兼得：一条选择性递推，状态从前缀压缩而来，代价 O(N)。

问题恰恰出在这个「兼得」上。SSM 定义在**一维序列**上，而图像不是一维的。所以每一个基于 Mamba 的修复模型都得回答同一个问题——**怎么把二维图像序列化，而不破坏它替代掉的那个卷积本来就免费拿到的二维邻域结构**——而每个模型给出的答案，就是它与其他模型的全部区别。

本仓库沿着这条线索走了两程：

| 阶段 | 问题 | 目录 |
|---|---|---|
| 一 | MambaIR 的四方向 Cross-Scan，真的给出了四个方向的信息吗？ | [`mambair/`](mambair/) |
| 二 | MaIR 修好了扫描，但顺手删光了卷积。这个代价有多大？ | [`mair-cnn-ssm/`](mair-cnn-ssm/) |

这里所有模型的训练条件完全一致——DIV2K（800 张）、128×128 裁片、batch size 96、25000 次迭代、单张 RTX 4090——所以主表里三行之间可以直接比较，**也只限于彼此之间**。论文原结果是 DF2K + 500K 迭代，比这里高 0.24–0.89 dB。

### 目录结构

```
mamba-image-super-resolution/
├── README.md                              # 本文件
├── README.zh-CN.md
├── mambair/                               # 第一程 —— MambaIR 复现 + 扫描消融
│   ├── README.md、README.zh-CN.md
│   ├── report.pdf                         # 13 页
│   ├── code/                              # mambair_arch.py、train.py、test.py、配置
│   ├── figures/                           # 架构图、cross-scan、SSM 机制、复杂度、训练演进
│   └── logs/                              # 1 次训练 + 1 次基准测试，25K 迭代
└── mair-cnn-ssm/                          # 第二程 —— MaIR 复现 + CSBlock
    ├── README.md、README.zh-CN.md
    ├── report.pdf                         # 14 页
    ├── code/                              # mair_arch.py、原版、train.py、test.py
    ├── yaml/                              # 4 份配置：MaIR-S 与 MaIR-CSB，训练 + 测试
    ├── figures/                           # 架构图、CSBlock、曲线、5 张四栏对比
    └── logs/                              # 3 次训练 + 2 次基准测试
```

## 2. 核心贡献

**一、一个把 Cross-Scan 冗余单独拎出来的扫描方向消融。** 只推理，权重冻结，除了 `forward_core` 里求和几个方向之外什么都不改。从四个方向砍到一个，在 Set5 / Set14 / B100 / Urban100 上平均只掉 **0.10 dB**。如果四个扫描真是四个互相独立的视角，砍掉 75% 远不止掉这么点。它没有，原因见 §4.3。

**二、同预算下的 MaIR 复现。** NSS（一条嵌套 S 型扫描路径）加 SSA（序列混洗注意力），在完全相同的算力下比 MambaIR 高 **+1.61 dB**——这是本仓库测到的最大效应。收敛也更快：约 4000 步到 29 dB，而 MambaIR 跑满 25000 步都没到过。

**三、CSBlock——一个 CNN-SSM 融合模块（46 行）。** 一个小 CNN **并联在同一个块内的 VMM 旁边**，读的是这个块的原始输入而不是 SSM 的输出，让局部分支看到未经处理的信号，而不是事后再去补救。**平均 +0.27 dB，四比四全胜，Urban100 上 +0.42 dB。**

**四、一份纠正过的参数量账目。** 配套报告里写 CSBlock「仅新增 5.5% 参数」。它新增的是 **+126%**——网络规模翻了一倍还多。§4.5 说明了 5.5% 是怎么来的、以及它为什么差了 24 倍。

**什么不算贡献。** 仓库里其余的东西都是上游的。BasicSR 分支、`train.py`、`test.py`、`mambair_arch.py`，以及 `mair_arch.py` 里除 `MSS_VMM` 类之外的全部内容，都是第三方代码，按各自原有许可在此复现。逐文件清单见 §3.4。

## 3. 方法与架构

### 3.1 问题在序列化本身，不在模型

![SSM 机制](mambair/figures/fig_ssm_mechanism.jpg)

SSM 维护一个隐状态 `h(t) = A·h(t-1) + B·x(t)`，输出 `y(t) = C·h(t)`。代价 O(N)，把整个前缀压进一个状态——这正是它有能力与注意力抗衡的原因。但把二维网格展平成一维序列，恰恰是麻烦的开始：网格上竖直相邻的两个像素，在序列里被拉得很远（上图橙色那一对），而一条递推要让状态跨过它们之间的全部 token 才能建立关联，于是这些容量花在了记住距离上，而不是记住结构。

本仓库的每个模型，都是对这一个问题的不同答案。

### 3.2 MambaIR 与 Cross-Scan

![Cross-scan、SS blocks、Cross-merge](mambair/figures/fig_cross_scan_merge.png)

MambaIR 的答案是扫描**四遍**——沿行、沿列、沿两条对角线——再把四个结果逐元素相加。这个求和就是 `mambair_arch.py:379-381` 那一行，也正是 §2.1 的消融能干净地隔离扫描机制、而不是隔离训练过程的原因。

![MambaIR 架构](mambair/figures/fig_mambair_architecture.png)

整体是 SwinIR 式的三阶段骨干——浅层特征提取、若干残差状态空间组的堆叠、高质量重建——只是 transformer 块换成了 RSSB，自注意力换成了上面那个 VSSM。

### 3.3 MaIR：NSS 与 SSA

![MaIR 架构](mair-cnn-ssm/figures/fig_mair_architecture.jpg)

MaIR 不去打补丁，而是正面回答序列化问题。**NSS** 是一条嵌套的 S 型路径：图像先切成若干水平条带，每条带内按 S 型往返访问，折返把竖直相邻像素的一维距离从 *W* 缩到了 2。一次扫描覆盖整张特征图，水平和竖直两个方向的邻域都保住了。**SSA** 负责融合：把通道混洗分组、打分、门控，用可学习的方式取代 MambaIR 的盲目相加。

代价在图上就能看出来：MaIR 的残差块是 `LayerNorm → VMM → LayerNorm → MLP`。CAB 一去，深层特征路径上**一个卷积都不剩了**。

### 3.4 CSBlock：CNN-SSM 融合

![CSBlock 架构](mair-cnn-ssm/figures/fig_csblock_architecture.png)

卷积被删干净，正是 CSBlock 要处理的事。这个模块让 SSM 流和卷积流**在同一个块内并排运行**，两条流拿到的是同一个原始输入，最后按通道拼接、线性投影融合：

| 组件 | 细节 |
|---|---|
| SSM 流 | 上游 VMM 原样不动——SS2D 核没有改 |
| CNN 流 | 两层完整 3×3 卷积（`padding=1`，不改变分辨率），每层后接 SiLU |
| 融合 | 按通道拼接（2C）→ `Linear(2C → C)` → `LayerNorm` |

*CSBlock 是模块名，CNN-SSM 是它实现的融合方案。* 类名叫 `MSS_VMM`，打开它的配置叫 `MaIR_MSS`，日志里把训出来的模型记作 MaIR-CSB——一个东西四个字符串，因为模块名字是照着方法后来才定的。

两个细节撑起了这个设计。CNN 分支读的是这个块的**输入**，不是 SSM 的输出，所以它是真正并行的第二条流，而不是给 SSM 打的残差补丁。卷积是全 3×3（`groups=1`）而不是深度可分离，所以它同时混合通道和空间。完整源码见 [`mair-cnn-ssm/code/mair_arch.py`](mair-cnn-ssm/code/mair_arch.py) 的 `MSS_VMM` 类，同目录下的 [`mair_arch_original.py`](mair-cnn-ssm/code/mair_arch_original.py) 可直接对读 diff。

在 `RMB.__init__` 里一行替换掉 `VMM` 即可。相对上游文件是**新增 46 行、改动 1 行**；`mair_arch_original.py` 一并留着，方便直接读 diff。

| 文件 | 状态 |
|---|---|
| `mair-cnn-ssm/code/mair_arch.py` | **我写的。** 新增 `MSS_VMM`（即 CSBlock），并让 `RMB` 改用它。46 行。 |
| `mair-cnn-ssm/code/mair_arch_original.py` | 上游 MaIR 原文件，原样保留供 diff。 |
| `mambair/code/mambair_arch.py` | 上游 MambaIR，未修改。 |
| `*/code/train.py`、`*/code/test.py` | 上游 BasicSR 入口，未修改。 |
| `mair-cnn-ssm/yaml/*` | 上游配置，超参见 §4.1。 |
| `*/report.pdf` | 两份书面报告：文献综述、方法、实验、误差分析。 |
| `*/logs/*` | BasicSR 原始输出，表格即由此得来。 |
| `*/figures/*` | 报告配图。§3.1–3.3 的机制图与架构图是第三方图片，转载自 MambaIR、MaIR 与 VMamba 论文；训练曲线、超分对比栏与训练演进图来自这几次实际运行。 |

除 `mair_arch.py` 之外的架构文件、`train.py`、`test.py` 都来自
[BasicSR](https://github.com/XPixelGroup/BasicSR)、[MambaIR](https://github.com/csguoh/MambaIR)
和 [MaIR](https://github.com/csguoh/MaIR)；训练配置是上游 lightSR 配置，换了更小的 `gt_size` 和更少的迭代数。这些项目的许可条款适用于它们自己的文件，不适用于本仓库整体。

## 4. 实验与结果

### 4.1 设置

所有运行：DIV2K 训练集（800 张）、×4 双三次退化、`gt_size=128`、`batch_size_per_gpu=96`、`total_iter=25000`、Adam（`lr=2e-4`、`weight_decay=0`）、`embed_dim=60`、seed 10、单张 RTX 4090，关闭水平翻转与旋转增强。评测为 Y 通道 PSNR / SSIM，`crop_border=4`，测试集 Set5、Set14、B100、Urban100。

三个模型来自两套不同的环境（复现跨越了两个上游分支，它们各自锁定了不同的 BasicSR 版本）——见 §5。

### 4.2 主结果

| 测试集 | MambaIR（复现） | MaIR（复现） | + CSBlock | MaIR（原文） | CSBlock 距原文 |
|---|:---:|:---:|:---:|:---:|:---:|
| Set5 | 29.29 / 0.837 | 31.64 / 0.886 | **31.98 / 0.891** | 32.48 / 0.899 | −0.50 |
| Set14 | 26.73 / 0.723 | 28.22 / 0.773 | **28.42 / 0.778** | 28.81 / 0.786 | −0.39 |
| B100 | 26.30 / 0.679 | 27.34 / 0.729 | **27.47 / 0.733** | 27.71 / 0.741 | −0.24 |
| Urban100 | 23.74 / 0.689 | 25.29 / 0.761 | **25.71 / 0.775** | 26.60 / 0.801 | −0.89 |
| **平均** | 26.52 | 28.12 | **28.40** | 28.90 | −0.50 |

增益并不均匀，而顺序本身就是重点：

| | Set5 | Set14 | B100 | Urban100 | 平均 |
|---|:---:|:---:|:---:|:---:|:---:|
| MambaIR → MaIR | +2.35 | +1.49 | +1.04 | +1.55 | **+1.61** |
| MaIR → +CSBlock | +0.34 | +0.20 | +0.13 | **+0.42** | **+0.27** |

### 4.3 CSBlock 的增益从哪来

Urban100——规则建筑与重复结构（砖缝、栏杆、窗格）的语料——增益是 B100 自然随机纹理的 3 倍。这正是设计所预期的：3×3 卷积核的感受野，对周期只有几个像素的规则图案恰好合适，对全局统计则完全不对路。四比四全胜、没有一处退化，是这张表要说明的主要事情。

收敛曲线从另一个角度讲同一个故事：CSBlock 约 10000 步突破 30 dB，MaIR 要 ~22000 步，训练早期大约快 2 倍，等 SSM 分支收敛后收窄到稳定的约 0.26 dB。

![训练演进](mambair/figures/training_progression.png)

*同一块裁片在不同训练阶段的样子：LR 双三次、随机初始化，以及 2.5K / 5K / 10K / 15K / 25K 迭代。*

§2.1 里的消融指向同一处。Cross-Scan 四个方向里有三个彼此近似重复——这件事在你想明白「一维展平在扫描**开始之前**就已经破坏了二维连续性」之后就不奇怪了：换一种顺序重新展平，并不能把丢掉的东西找回来。MambaIR 买了四次扫描，拿到的是一个方向的信息量；MaIR 那条单独的 NSS 路径，就是这件事的直接后果。

### 4.4 视觉对比

`LR（双三次） | MaIR | MaIR + CSBlock | GT`，每个基准数据集一张裁片，保持原始分辨率——它们是 §4.2 里 +0.13…+0.42 dB 那个梯次的视觉证据。

| Set5 | Set14 |
|---|---|
| ![Set5](mair-cnn-ssm/figures/Set5_img_002_4panel.png) | ![Set14](mair-cnn-ssm/figures/Set14_img_011_4panel.png) |

| B100 | Urban100 |
|---|---|
| ![B100](mair-cnn-ssm/figures/B100_img_079_4panel.png) | ![Urban100](mair-cnn-ssm/figures/Urban100_img_043_4panel.png) |

### 4.5 参数量

数字取自训练日志里 BasicSR 打印的真实值：

| | 参数量 |
|---|---:|
| MambaIR-Light（复现） | 924,492 |
| MaIR-S-Light | 1,374,156 |
| MaIR + CSBlock（MaIR-CSB） | 3,109,356 |
| **MaIR-S → MaIR-CSB** | **+1,735,200（+126%）** |

单块：2 个 3×3 卷积（2 × 32,460）+ 融合层 `Linear(2C, C)`（7,260）+ `LayerNorm`（120）= **72,300**，× 24 块 = 1,735,200——与日志给出的差值完全吻合。

> **关于 5.5% 这个数字。** 配套报告里写 CSBlock「仅新增 5.5% 参数」，§3.4 那张架构图上也印着同样的说法。这个数其实是**单块的** 72,300 除以**整个基础模型**（72,300 ÷ 1,374,156 ≈ 5.3%），只在这个「每块」的意义上成立。模型层面的真实增幅是 **+126%**——网络规模翻了一倍还多。+0.27 dB 是真的，CSBlock 的设计本身也站得住，但报告里那套「轻量代价」的表述差了 24 倍，上表是纠正后的版本。

所以诚实的总结是：**+0.27 dB 换 +126% 参数，且增益集中在规则纹理上。** 作为一篇论文式的贡献，这个交换很不划算；作为一次「纯 SSM 骨干究竟弱在哪」的探查——弱在局部的、周期性的结构——信号是清晰的，而且分数据集的排序印证的是机制本身，不只是一个数字。

## 5. 技术栈

| | 阶段一（`mambair/`） | 阶段二（`mair-cnn-ssm/`） |
|---|---|---|
| 操作系统 | Ubuntu 22.04 | Ubuntu 22.04 |
| 显卡 | RTX 4090（显存扩展后 49 GB） | RTX 4090 |
| Python | 3.10 | 3.10 |
| PyTorch | 2.1.0+cu121 | 2.4.0+cu121 |
| TorchVision | 0.16.0+cu121 | 0.19.0+cu121 |
| CUDA | 12.1 | 12.1 |
| BasicSR | 1.3.5（随 MambaIR 分支提供） | 1.2.0+0051603（随 MaIR 分支提供） |
| 算子库 | `mamba-ssm` 1.1.1 + `causal-conv1d` 1.1.1，用 `nvcc` 编译 | 同上 |

两阶段是两套独立环境，因为它们跟着两个不同的上游分支走，而两个分支锁定的 BasicSR 版本不同。运行时不共享任何东西，各自端到端复现各自的阶段。

## 6. 安装与使用

两个阶段都没带完整的可运行工程——这里放的是与上游有差异的那部分文件。要跑起来，先克隆对应的上游仓库，再把这些文件放进去。

### 阶段一 —— MambaIR

```bash
git clone https://github.com/csguoh/MambaIR.git
cd MambaIR
pip install -r requirements.txt          # 编译 mamba-ssm 需要 nvcc
pip install mamba-ssm==1.1.1 causal-conv1d==1.1.1

# 把本仓库 mambair/ 下的文件覆盖到上游工程
cp <this-repo>/mambair/code/mambair_arch.py   basicsr/archs/
cp <this-repo>/mambair/code/train.py          basicsr/train.py
cp <this-repo>/mambair/code/test.py           basicsr/test.py

python -m basicsr.train -opt <this-repo>/mambair/code/train_MambaIR_lightSR_x4_my.yml
python -m basicsr.test  -opt <this-repo>/mambair/code/test_MambaIR_lightSR_x4_my.yml
```

### 阶段二 —— MaIR

```bash
git clone https://github.com/csguoh/MaIR.git
cd MaIR
pip install -r requirements.txt
pip install mamba-ssm==1.1.1 causal-conv1d==1.1.1

cp <this-repo>/mair-cnn-ssm/code/mair_arch.py basicsr/archs/mair_arch.py

# MaIR-CSB：带 CSBlock 的版本
python -m basicsr.train -opt <this-repo>/mair-cnn-ssm/yaml/train_MaIR_MSS_lightSR_x4.yml
python -m basicsr.test  -opt <this-repo>/mair-cnn-ssm/yaml/test_MaIR_MSS_lightSR_x4.yml

# MaIR-S 基线：先换回未修改的架构文件
cp <this-repo>/mair-cnn-ssm/code/mair_arch_original.py basicsr/archs/mair_arch.py
python -m basicsr.train -opt <this-repo>/mair-cnn-ssm/yaml/train_MaIR_S_lightSR_x4_my.yml
```

**配置必须改。** 它们指向原机器上 `/root/autodl-tmp/...` 的数据集与权重路径——两个阶段一共 12 个文件带着这些绝对路径。运行前先改 `dataroot_gt`、`dataroot_lq` 和测试集。

仓库不含模型权重：25K 迭代的权重大约 15 MB 一个，日志与配图才是这几次运行的记录。

## 7. 局限

- **只有一种配置。** Light 变体、DIV2K、×4。CSBlock 在更大模型、DF2K 数据集、去噪与去模糊任务上是什么表现，这里一个字都说明不了。
- **绝对 PSNR 不能和论文比。** 800 张跑 25K 迭代，对 3450 张跑 500K。有意义的比较只在同一张表内部。
- **CSBlock 的结构是手工定的**——两层 3×3 卷积、通道拼接、线性融合。没有做结构搜索，也没试过其他方案。
- **「规则纹理」在这里是定性描述。** Urban100 > Set5 > Set14 > B100 的梯次一致且符合假设，但没有计算任何纹理复杂度指标去预测增益，所以机制是被论证的，不是被测量的。
- **+126% 参数换 +0.27 dB 是很贵的**，如 §4.5 所述。CSBlock 首先是一件诊断工具，其次才是一个有竞争力的模型。
- **扫描消融只做了推理。** 它复用同一个训练好的权重、只改前向，所以能把机制干净地单独拎出来，但也说明不了「四方向模型如果就这么训练」会学到什么。
- **两套环境、两个分支。** 见 §5。单环境复现会更干净，但这里是实际跑出来的样子。

## 参考文献

- Guo et al. **MambaIR: A Simple Baseline for Image Restoration with State-Space Model.** ECCV 2024.
- Li et al. **MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration.** CVPR 2025.
  ([arXiv:2412.20066](https://arxiv.org/abs/2412.20066))
- Liu et al. **VMamba: Visual State Space Model.** NeurIPS 2024.
- Liang et al. **SwinIR: Image Restoration Using Swin Transformer.** ICCVW 2021.
- Gu & Dao. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** 2023.
