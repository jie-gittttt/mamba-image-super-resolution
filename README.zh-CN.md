# 基于状态空间模型的图像超分辨率

**扫描冗余与局部算子的取舍**

[English](README.md) | [中文](README.zh-CN.md)

> 课程研究报告项目，分两程。完整报告见 [`mambair/report.pdf`](mambair/report.pdf)（13 页）与 [`mair-cnn-ssm/report.pdf`](mair-cnn-ssm/report.pdf)（14 页）。

<p align="center">
  <img src="framework.jpg" alt="MaIR + CSBlock 整体框架">
  <br><em>整个系统一张图。左半边：三阶段骨干、RMB 内部结构，以及 CSBlock 插在哪个位置。右半边：CSBlock 内部的两条并行流（CNN 分支与 VMM 分支）与融合方式。左下角：两程的最终指标。下面各节是这张图的展开。</em>
</p>

---

## 1. 项目背景

单图像超分辨率从一张低分辨率观测重建高分辨率图像，是一个病态逆问题：同一张 LR 输入对应很多张合理的 HR 图像。任务因此不是「还原」而是「挑选」——学一个关于自然图像结构的先验，用它把正确的那一张挑出来。承载这个先验的序列模型有三条路线：CNN 靠堆深度把感受野撑大，代价是全局关系只能被间接地逼近；Transformer 用两两注意力建模全图，代价是 O(N²)；状态空间模型（SSM）——尤其是 Mamba——声称两者兼得，用一条选择性递推把整个前缀压进隐状态，代价 O(N)。MambaIR 据此把 SwinIR 的注意力块换成 VSSM 搬进图像修复，MaIR 再往前推一步。问题恰恰出在这个「兼得」上：SSM 定义在**一维序列**上，而图像是二维网格。所以每一个基于 Mamba 的修复模型都得回答同一个问题——**怎么把二维图像序列化，而不破坏它替代掉的那个卷积本来就免费拿到的二维邻域结构**——而每个模型给出的答案，就是它与其他模型的全部区别。

MambaIR 的答案是扫描**四遍**：沿行、沿列、沿两条对角线各展平一次，再把四路输出逐元素相加。这个赌注是「四条一维顺序加起来足以近似二维邻域」。本仓库第一程直接检验它：在冻结权重上、只改前向里求和几个方向，把四个方向砍到一个，四个基准平均只掉 **0.10 dB**。四个方向里有三个是回声，不是四个独立视角。根因也不难想明白——一维展平在扫描**开始之前**就已经破坏了二维连续性：64×64 的特征图里，竖直相邻的像素在行主序下相隔 64 个 token，而换一种顺序重新展平并不能把丢掉的东西找回来。MambaIR 买了四次扫描，拿到的是一个方向的信息量。

诊断清楚了，设计原则也就清楚了：与其用更多扫描顺序去补偿，不如把局部算子直接放回来。MaIR 顺着这条路给出了正面解法——**NSS** 用一条嵌套的 S 型路径替代四条直线，条带内折返把竖直相邻像素的一维距离从 *W* 压到 **2**，一次扫描同时保住两个方向的邻域；**SSA** 用可学习的通道门控取代 MambaIR 的盲目相加。在完全相同的训练条件下，这值 **+1.61 dB**，是本仓库测到的最大效应。但代价在架构图上就能看见：MaIR 的残差块是 `LayerNorm → VMM → LayerNorm → MLP`，CAB 一去，深层特征路径上**一个卷积都不剩了**。一个归纳偏置是「全局」的机制，要去表达局部的、高频的纹理——这就是 **CSBlock** 要检验的假设。

CSBlock 让 SSM 流和卷积流**在同一个块内并排运行**，两条流拿到的是同一个原始输入，最后按通道拼接、线性投影融合。平均 **+0.27 dB，四比四全胜，没有一处退化**；Urban100 上（+0.42，规则建筑与重复结构）的增益是 B100（+0.13，自然随机纹理）的 3 倍——增益的排序印证的是机制本身，不只是一个数字。代价是 **+126% 参数**，网络规模翻了一倍还多。作为一篇论文式的贡献，这个交换很不划算；作为一次「纯 SSM 视觉骨干究竟弱在哪」的探查，信号是清晰的：弱在局部的、周期性的结构，而一个 3×3 的感受野就足以应对它。

<p align="center">
  <img src="mambair/figures/fig_ssm_mechanism.jpg" alt="二维网格展平为一维序列后的 SSM 递推">
  <br><em>SSM 维护 h(t) = A·h(t-1) + B·x(t)，输出 y(t) = C·h(t)：代价 O(N)，一个隐状态承载整个前缀。但把二维网格展平，竖直相邻的两个像素在一维顺序里被拉得很远（橙色那一对），状态只能拿容量去表达这段距离，而不是表达那个关系。</em>
</p>

---

## 2. 主要贡献

1. **一个把 Cross-Scan 冗余单独拎出来的扫描方向消融。** 只推理，权重冻结，除了 `forward_core` 里求和几个方向之外什么都不改。从四个方向砍到一个，四个基准平均只掉 **0.10 dB**。如果四个扫描真是四个互相独立的视角，砍掉 75% 远不止掉这么点——它没有。

2. **同预算下复现 MambaIR 与 MaIR。** 两个上游模型在完全相同的算力下跑完，MaIR 比 MambaIR 高 **+1.61 dB**，是本仓库最大的效应；收敛也快得多，约 4,000 步到 29 dB，而 MambaIR 跑满 25,000 步都没到过。有了这把统一的尺子，下面所有对比才是可比的。

3. **CSBlock——一个 CNN-SSM 融合模块（46 行）。** 一个小 CNN **并联在同一个块内的 VMM 旁边**，读的是这个块的原始输入而不是 SSM 的输出，让局部分支看到未经处理的信号，而不是事后再去补救。**平均 +0.27 dB，四比四全胜，Urban100 上 +0.42 dB。**

4. **一份纠正过的参数量账目。** CSBlock 的模型级开销是 **+126%**，不是可以忽略的零头。这条被单独列出来，是因为它决定了这个模块该怎么被定位——一件诊断工具，而不是一个划算的改进。

最终结果（×4，Y 通道 PSNR / SSIM）：

| 测试集 | MambaIR（复现） | MaIR（复现） | MaIR + CSBlock | MaIR（原文） | CSBlock 距原文 |
|---|:---:|:---:|:---:|:---:|:---:|
| Set5 | 29.29 / 0.837 | 31.64 / 0.886 | **31.98 / 0.891** | 32.48 / 0.899 | −0.50 |
| Set14 | 26.73 / 0.723 | 28.22 / 0.773 | **28.42 / 0.778** | 28.81 / 0.786 | −0.39 |
| B100 | 26.30 / 0.679 | 27.34 / 0.729 | **27.47 / 0.733** | 27.71 / 0.741 | −0.24 |
| Urban100 | 23.74 / 0.689 | 25.29 / 0.761 | **25.71 / 0.775** | 26.60 / 0.801 | −0.89 |
| **平均** | 26.52 | 28.12 | **28.40** | 28.90 | −0.50 |

两程的增益分解：

| | Set5 | Set14 | B100 | Urban100 | 平均 |
|---|:---:|:---:|:---:|:---:|:---:|
| 扫描路径 MambaIR → MaIR | +2.35 | +1.49 | +1.04 | +1.55 | **+1.61** |
| 局部算子 MaIR → +CSBlock | +0.34 | +0.20 | +0.13 | **+0.42** | **+0.27** |

原文那一列用的是 DF2K（3,450 张）跑 500K 迭代，本仓库是 DIV2K（800 张）跑 25K——那 0.24–0.89 dB 的差距是训练预算的差距。有意义的比较只在同一张表内部。原始日志见 [`mambair/logs/`](mambair/logs/) 与 [`mair-cnn-ssm/logs/`](mair-cnn-ssm/logs/)。

**什么不算贡献。** 仓库里其余的东西都是上游的。BasicSR 分支、`train.py`、`test.py`、`mambair_arch.py`，以及 `mair_arch.py` 里除 `MSS_VMM` 类之外的全部内容，都是第三方代码，按各自原有许可在此复现；逐文件清单见文末的[仓库结构](#仓库结构)。

---

## 3. 方法与结构

### 3.1 基础架构：MambaIR 与 MaIR

两程都建立在现成的上游模型上，本仓库没有改动它们的骨干。第一程的 [MambaIR](https://github.com/csguoh/MambaIR)（ECCV 2024）是 SwinIR 式的三阶段骨干——浅层特征提取、残差状态空间组的堆叠、重建头——只是 transformer 块换成了 RSSB，自注意力换成了 VSSM。第二程的 [MaIR](https://github.com/csguoh/MaIR)（CVPR 2025）保留这个骨架，重写了残差块内部；顶部框架图的左半边画的就是它，包含 RMB 的 `LayerNorm → VMM → LayerNorm → MLP` 这条纯 SSM 路径。MaIR 论文原图（含 VMM 内部结构）见 [`mair-cnn-ssm/`](mair-cnn-ssm/README.zh-CN.md)。

两者的差异就是这条线索本身：

| | MambaIR | MaIR |
|---|---|---|
| 扫描路径 | Cross-Scan：4 条独立直线 | NSS：一条嵌套 S 型路径 |
| 垂直连续性 | 靠额外扫描来补偿 | 靠折返天然保留（一维距离 → 2） |
| SSM 调用 | 每块 4 次 | 每块 1 次 |
| 融合方式 | 四方向逐元素相加 | 单路径，无需融合 |
| 注意力 | 无 | SSA 通道混洗门控 |
| 前馈 | CAB（Conv2d + 通道注意力） | **MLP（纯线性层）** |

最后一行是切入点。MambaIR 的前馈块里含一个卷积，MaIR 把它换成线性 MLP，也就顺手把深层路径上最后一个局部算子拿掉了。

### 3.2 CSBlock：CNN-SSM 融合

顶部框架图的右半边是它，下面是同一张图的全分辨率版本。卷积被删干净正是 CSBlock 要处理的事——这个模块让 SSM 流和卷积流在同一个块内并排运行，两条流拿到的是同一个原始输入：

<p align="center">
  <img src="mair-cnn-ssm/figures/fig_csblock_architecture.png" alt="CSBlock：CNN 流与 SSM 流并联">
  <br><em>CSBlock 内部。注意 CNN 分支的箭头直接来自最左侧的输入，不经过 VMM——这是「并联」这个词的全部含义。图右下角的 +72K 是单块开销，24 块累计 +1.735M，即模型层面的 +126%，见 §4.4。</em>
</p>

| 组件 | 细节 |
|---|---|
| SSM 流 | 上游 VMM 原样不动——`in_proj → permute + Conv1d + SiLU → NSS 扫描 + 选择性扫描 → chunk/求和/门控 → out_norm + gate(z) + out_proj`，SS2D 核没有改 |
| CNN 流 | `permute → BCHW → Conv3×3 + SiLU → Conv3×3 + SiLU → permute → BHWC`，`padding=1`，不改变分辨率 |
| 融合 | 按通道拼接（`2C`）→ `Linear(2C → C)` → `LayerNorm` |

落到代码上，这是 `mair_arch.py:346` 处一个叫 `MSS_VMM` 的类，由三部分组成：上游的 `VMM`、一个两层 `nn.Sequential` 卷积流，以及 `Linear(2C → C)` 加 `LayerNorm` 的融合头。在 `RMB.__init__` 里一行替换掉 `VMM` 即可，相对上游是**新增 46 行、改动 1 行**。完整源码见 [`mair-cnn-ssm/code/mair_arch.py`](mair-cnn-ssm/code/mair_arch.py)，同目录的 [`mair_arch_original.py`](mair-cnn-ssm/code/mair_arch_original.py) 原样保留，可直接对读 diff：

```bash
diff mair-cnn-ssm/code/mair_arch_original.py mair-cnn-ssm/code/mair_arch.py
```

*CSBlock 是模块名，CNN-SSM 是它实现的融合方案。* 类名叫 `MSS_VMM`，打开它的配置叫 `MaIR_MSS`，日志里把训出来的模型记作 MaIR-CSB——一个东西四个字符串，因为模块名字是照着方法后来才定的。

三个细节撑起了这个设计：

- **CNN 分支读的是这个块的输入，不是 SSM 的输出。** 这使它成为真正并行的第二条流，而不是一个残差补丁——也正是这个模块被称作「层内」而非事后补救的原因。
- **卷积是完整 3×3（`groups=1`），不是深度可分离。** 深度可分离能把开销再砍掉约 32 倍，但它不做通道间混合，而通道混合正是这里要求局部算子承担的一部分。
- **VMM 原样不动。** CSBlock 加什么都是在 SSM 旁边加，所以任何退化都不可能归因到扫描本身被改动。

### 3.3 实验路线

两程的问题不是并排的两个课题，而是同一条线索的两步——第二步的动机正是第一步的结论：

| 阶段 | 问题 | 变体 | 结论 |
|---|---|---|---|
| 一 | Cross-Scan 的四个方向真的各自携带信息吗？ | 4-dir / 2-dir / 1-dir，只改前向，权重冻结 | 砍到单方向只掉 0.10 dB，三个方向是回声（§4.3） |
| 二 | MaIR 修好扫描的代价是删光了卷积，这个代价有多大？ | MaIR-S / MaIR-CSB | +0.27 dB 换 +126% 参数，增益集中在规则纹理（§4.2、§4.4） |

---

## 4. 实验与结果

### 4.1 实验设置

**数据。** DIV2K 训练集（800 张），×4 双三次退化，`gt_size = 128`。评测用 Set5、Set14、B100、Urban100 四个标准基准。

**模型。** MambaIR-Light（924,492 参数）、MaIR-S-Light（1,374,156）、MaIR-CSB（3,109,356），`embed_dim = 60`。三者除骨干与是否插入 CSBlock 之外配置逐字相同。

**训练。** 25,000 次迭代，Adam（`lr = 2e-4`、`weight_decay = 0`），`batch_size_per_gpu = 96`，seed 10，单张 RTX 4090。水平翻转与旋转增强关闭——这让对比更干净，但也去掉了上游配置里的一个正则项。

**指标。** Y 通道 PSNR / SSIM，`crop_border = 4`。原始日志见 [`mambair/logs/`](mambair/logs/) 与 [`mair-cnn-ssm/logs/`](mair-cnn-ssm/logs/)。

三个模型来自两套不同的环境——复现跨越了两个上游分支，它们各自锁定了不同的 BasicSR 版本，见 §5。

### 4.2 增益从哪来

§2 那张增益分解表里，排序本身就是论据。Urban100 是一批规则建筑——砖缝、栏杆、窗格、立面重复；B100 是自然随机纹理。如果 CSBlock 起作用的机制是「补回一个局部算子」，那么它的增益就该随图中局部周期结构的多少而变——事实正是如此：**Urban100 是 B100 的 3.2 倍**。四比四全胜、无一处退化，则排除了增益来自噪声的可能。

<p align="center">
  <img src="mair-cnn-ssm/figures/fig_psnr_curve.png" alt="MaIR 与 MaIR-CSB 的训练曲线">
  <br><em>收敛曲线给出相同的答案：CSBlock 约 10,000 步突破 30 dB，MaIR 要 ~22,000 步——训练早期大约快 2 倍——等 SSM 分支收敛后差距收窄到稳定的约 0.26 dB。</em>
</p>

局部算子似乎早期廉价好拟合，等全局模型追上来之后价值变小；那剩下的 0.26 dB 就是它在平衡态下的价值。

### 4.3 扫描方向消融

只推理，用同一个训练好的权重，唯一变量是 `forward_core` 求和几个方向：

| 配置 | 参与方向 | Set5 | Set14 | B100 | Urban100 | 平均 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 4-dir（MambaIR 默认） | 前行 + 反行 + 转置 + 反对角 | 29.29 | 26.73 | 26.30 | 23.74 | 26.52 |
| 2-dir | 前行 + 反行 | 29.19 | 26.71 | 26.27 | 23.70 | 26.47 |
| 1-dir | 仅前行 | 29.13 | 26.66 | 26.23 | 23.65 | 26.42 |

| | Set5 | Set14 | B100 | Urban100 | 平均 |
|---|:---:|:---:|:---:|:---:|:---:|
| Δ 4 → 2 | −0.10 | −0.02 | −0.03 | −0.04 | **−0.05** |
| Δ 4 → 1 | −0.16 | −0.07 | −0.07 | −0.09 | **−0.10** |

丢掉 75% 的扫描方向，代价是 0.10 dB 和 0.005 SSIM。三个原因，按重要性递增：

**选择性机制本身抹平了方向差异。** Mamba 的 `B(x)`、`C(x)`、`Δ(x)` 都是输入的函数，这正是它区别于 S4 这类静态 SSM 的地方。面对相同的内容，同样的内容过滤过程会在四条顺序上各跑一遍：关键的重量关系会集中到传播得最好的那个方向，其余方向收敛成它的近似。**逐元素相加没有任何区分手段。** `y = y1 + y2 + y3 + y4` 把四个视角当成等权，没有门控、没有可学习权重，模型学不会更信某个方向，而几个近似相同序列的和不过是一个标量倍数。**一维展平早就破坏了二维连续性**——这是根因，也是第二程的出发点。

### 4.4 参数量

数字取自训练日志里 BasicSR 打印的真实值：

| | 参数量 |
|---|---:|
| MambaIR-Light（复现） | 924,492 |
| MaIR-S-Light | 1,374,156 |
| MaIR + CSBlock（MaIR-CSB） | 3,109,356 |
| **MaIR-S → MaIR-CSB** | **+1,735,200（+126%）** |

单块：2 个 3×3 卷积（2 × 32,460）+ 融合层 `Linear(2C, C)`（7,260）+ `LayerNorm`（120）= **72,300**，× 24 块 = 1,735,200——与日志给出的差值完全吻合，这同时验证了单块数字与块数。

报告里出现的「5.5%」是**单块的** 72,300 除以**整个基础模型**（72,300 ÷ 1,374,156 ≈ 5.3%），描述的是「每块只加这么点」，不是模型规模的整体涨幅。模型级的口径是 **+126%**，两者差 24 倍。

所以诚实的总结是：**+0.27 dB 换 +126% 参数，且增益集中在规则纹理上。** CSBlock 首先是一件诊断工具，其次才是一个有竞争力的模型——它定位了纯 SSM 视觉骨干的一处具体缺陷，说明一个 3×3 的感受野就足以应对它，并且证明效应最大的地方恰好是理论预期的那个地方。

### 4.5 视觉对比

`LR（双三次） | MaIR | MaIR + CSBlock | GT`，保持原始分辨率——它们是 §2 里 +0.13…+0.42 dB 那个梯次的视觉证据。这些图刻意不做压缩：它们的作用是展示像素级的纹理差异，重采样就失去意义了。

<p align="center">
  <img src="mair-cnn-ssm/figures/Set5_img_002_4panel.png" alt="Set5 img_002 四栏对比">
  <br><em>Set5 img_002。自然平滑区域占主导，是四个基准里增益最小的一档。</em>
</p>

<p align="center">
  <img src="mair-cnn-ssm/figures/Urban100_img_043_4panel.png" alt="Urban100 img_043 四栏对比">
  <br><em>Urban100 img_043。规则立面的重复结构，+0.42 dB 最看得出来。这个数据集上还有第二张裁片（img_047）见 <a href="mair-cnn-ssm/README.zh-CN.md">mair-cnn-ssm/</a>。</em>
</p>

---

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

---

## 6. 安装与使用

两个阶段都没带完整的可运行工程——这里放的是与上游有差异的那部分文件。要跑起来，先克隆对应的上游仓库，再把这些文件放进去。

**1 —— 阶段一：MambaIR 复现与扫描消融。**

```bash
git clone https://github.com/csguoh/MambaIR.git
cd MambaIR
pip install -r requirements.txt          # 编译 mamba-ssm 需要 nvcc
pip install mamba-ssm==1.1.1 causal-conv1d==1.1.1

# 把本仓库 mambair/ 下的文件覆盖到上游工程
cp <this-repo>/mambair/code/mambair_arch.py   basicsr/archs/
cp <this-repo>/mambair/code/train.py          basicsr/train.py
cp <this-repo>/mambair/code/test.py           basicsr/test.py

# 训练 -> ./experiments/ ；测试四个基准 -> ./results/
python -m basicsr.train -opt <this-repo>/mambair/code/train_MambaIR_lightSR_x4_my.yml
python -m basicsr.test  -opt <this-repo>/mambair/code/test_MambaIR_lightSR_x4_my.yml
```

**2 —— 阶段二：MaIR 复现与 CSBlock。**

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
python -m basicsr.test  -opt <this-repo>/mair-cnn-ssm/yaml/test_MaIR_S_lightSR_x4_my.yml
```

**3 —— 改配置。** 所有配置都指向原机器上的绝对路径：阶段一的 `/root/autodl-tmp/datasets/DIV2K/...`，阶段二的 `/root/2025-CVPR-MaIR-main/experiments/...`，两个阶段一共 12 个文件带着这些路径。运行前先改 `dataroot_gt`、`dataroot_lq`、四个测试集根目录，以及阶段二的 `path.pretrain_network_g` 与结果输出目录。请在克隆下来的上游工程根目录下运行——相对的实验目录与结果目录都基于当前工作目录解析。

仓库不含模型权重：25K 迭代的权重大约 15 MB 一个，日志与配图才是这几次运行的记录。

---

## 7. 局限

- **只有一种配置。** Light 变体、DIV2K、×4。CSBlock 在更大模型、DF2K 数据集、去噪与去模糊任务上是什么表现，这里一个字都说明不了。
- **绝对 PSNR 不能和论文比。** 800 张跑 25K 迭代，对 3,450 张跑 500K。有意义的比较只在同一张表内部。
- **CSBlock 的结构是手工定的**——两层 3×3 卷积、通道拼接、线性融合。没有做结构搜索，也没试过深度可分离、单层卷积、注意力融合、串行而非并行摆放这些替代方案。
- **没有做摆放位置的消融。** 「读输入而不是读 SSM 输出」这一点是从设计上论证的，没有拿一个读 SSM 输出的变体做对照。这是最明显的下一个实验。
- **「规则纹理」在这里是定性描述。** Urban100 > Set5 > Set14 > B100 的梯次一致且符合假设，但没有计算任何纹理复杂度指标去预测增益，所以机制是被论证的，不是被测量的。
- **+126% 参数换 +0.27 dB 是很贵的**，如 §4.4 所述。CSBlock 首先是一件诊断工具，其次才是一个有竞争力的模型。
- **扫描消融只做了推理。** 它复用同一个训练好的权重、只改前向，所以能把机制干净地单独拎出来，但也说明不了「四方向模型如果就这么训练」会学到什么。
- **两套环境、两个分支。** 见 §5。单环境复现会更干净，但这里是实际跑出来的样子。

---

## 仓库结构

```
mamba-image-super-resolution/
├── .gitignore
├── README.md、README.zh-CN.md              # 本文件
├── framework.jpg                           # 整体框架图，覆盖两程与 CSBlock
├── mambair/                                # 第一程 —— MambaIR 复现 + 扫描方向消融
│   ├── README.md、README.zh-CN.md
│   ├── report.pdf                          # 13 页
│   ├── code/
│   │   ├── mambair_arch.py                 # 上游 MambaIR，未修改
│   │   ├── train.py、test.py               # 上游 BasicSR 入口，未修改
│   │   └── train_MambaIR_lightSR_x4_my.yml # 本次运行使用的配置
│   ├── figures/                            # 6 张：架构、cross-scan、SSM 机制、真实场景、复杂度、训练演进
│   └── logs/                               # 1 次训练 + 1 次基准测试
└── mair-cnn-ssm/                           # 第二程 —— MaIR 复现 + CSBlock
    ├── README.md、README.zh-CN.md
    ├── report.pdf                          # 14 页
    ├── code/
    │   ├── mair_arch.py                    # 新增 CSBlock（MSS_VMM 类）
    │   ├── mair_arch_original.py           # 上游 MaIR，留作 diff
    │   └── train.py、test.py               # 上游 BasicSR 入口，未修改
    ├── yaml/                               # 4 份配置：MaIR-S 与 MaIR-CSB，各含训练 + 测试
    ├── figures/                            # 8 张：架构、CSBlock、PSNR 曲线、5 张四栏对比
    └── logs/                               # 3 次训练 + 2 次基准测试
```

上表里每一个文件是谁写的：

| 文件 | 状态 |
|---|---|
| `mair-cnn-ssm/code/mair_arch.py` | **我写的。** 新增 `MSS_VMM`（即 CSBlock），并让 `RMB` 改用它。46 行。 |
| `mair-cnn-ssm/code/mair_arch_original.py` | 上游 MaIR 原文件，原样保留供 diff。 |
| `mambair/code/mambair_arch.py` | 上游 MambaIR，未修改。 |
| `*/code/train.py`、`*/code/test.py` | 上游 BasicSR 入口，未修改。 |
| `*/code/*.yml`、`mair-cnn-ssm/yaml/*` | 上游 lightSR 配置，换了更小的 `gt_size` 和更少的迭代数。 |
| `*/report.pdf` | 两份书面报告：文献综述、方法、实验、误差分析。 |
| `*/logs/*` | BasicSR 原始输出，正文表格即由此得来。 |
| `*/figures/*` | 报告配图。机制图与架构图是第三方图片，转载自 MambaIR、MaIR 与 VMamba 论文；训练曲线、超分对比栏与训练演进图来自这几次实际运行。 |

除 `mair_arch.py` 之外的架构文件、`train.py`、`test.py` 都来自
[BasicSR](https://github.com/XPixelGroup/BasicSR)、[MambaIR](https://github.com/csguoh/MambaIR)
和 [MaIR](https://github.com/csguoh/MaIR)；训练配置是上游 lightSR 配置的改小版本。这些项目的许可条款适用于它们自己的文件，不适用于本仓库整体。模型权重与数据集都不随仓库分发。

---

## 参考文献

- Guo et al. **MambaIR: A Simple Baseline for Image Restoration with State-Space Model.** ECCV 2024.
  ([arXiv:2403.07764](https://arxiv.org/abs/2403.07764))
- Li et al. **MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration.** CVPR 2025.
  ([arXiv:2412.20066](https://arxiv.org/abs/2412.20066))
- Liu et al. **VMamba: Visual State Space Model.** NeurIPS 2024.
  ([arXiv:2401.10166](https://arxiv.org/abs/2401.10166))
- Liang et al. **SwinIR: Image Restoration Using Swin Transformer.** ICCVW 2021.
- Gu & Dao. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** 2023.
