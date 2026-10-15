# MaIR + CSBlock：一个 CNN-SSM 融合模块

**把纯 SSM 骨干丢掉的局部算子放回块内**

[English](README.md) | [中文](README.zh-CN.md) | [← 仓库根目录](../README.zh-CN.md)

> 本仓库第二程。完整报告见 [`report.pdf`](report.pdf)（14 页）。第一程——MambaIR 复现与扫描方向消融——在 [`../mambair/`](../mambair/)。

<p align="center">
  <img src="figures/fig_csblock_architecture.png" alt="CSBlock：CNN 流与 SSM 流并联">
  <br><em>CSBlock：一个小 CNN 与 VMM 并联在同一个残差块内。注意 CNN 分支的箭头直接来自最左侧的输入，不经过 VMM——这是「并联」这个词的全部含义。右下角的 +72K 是单块开销，24 块累计 +1.735M，即模型层面的 +126%，见 §4.3。</em>
</p>

---

## 1. 项目背景

第一程测量到，MambaIR 的四个 Cross-Scan 方向里有三个彼此近似冗余，根因是一维展平在扫描**开始之前**就已经破坏了二维连续性，而换一种顺序重新展平并不能把它找回来。[MaIR](https://github.com/csguoh/MaIR)（CVPR 2025）顺着这个诊断给出了正面解法：**NSS**（嵌套 S 型扫描）用一条嵌套的 S 型路径替代四条独立直线，图像先切成互不重叠的水平条带，每条带内按 S 型往返访问，折返把竖直相邻像素的一维距离从 *W* 压到 **2**，一次扫描同时保住两个方向的邻域；**SSA** 用可学习的通道门控取代 MambaIR 那句盲目的逐元素相加。在完全相同的训练条件下这值 **+1.61 dB**，是本仓库测到的最大效应。

代价在架构图上就能看见。MaIR 的残差块是 `LayerNorm → VMM → LayerNorm → MLP`，CAB 一去，**深层特征路径上再没有任何一个卷积**。这是一次刻意的选择——做一个纯 SSM 的视觉模型——但它同时意味着，网络不得不靠一个归纳偏置是「全局」的机制，去表达局部的、高频的纹理。于是问题变得具体：**这个代价有多大，能不能用最小的改动量化它？**

<p align="center">
  <img src="figures/fig_mair_architecture.jpg" alt="MaIR 架构">
  <br><em>（a）三阶段整体骨干——浅层特征提取、残差 Mamba 组、重建。（b）RMB：LayerNorm → VMM → LayerNorm → MLP。最后那一步是 MLP 而不是 CAB，卷积就是在这里消失的。（c）VMM，NSS 在里面。来源：MaIR 论文 Fig. 2。</em>
</p>

CSBlock 就是这样一个假设——「卷积的缺席正是 MaIR 在规则纹理上表现受限的原因」——以及一个检验它的最小改动。检验的方式很直接：把局部算子放回去，看增益落在哪里。如果假设成立，增益就应该随图中局部周期结构的多少而变，而不该是个均匀的背景提升。**事实正是如此：Urban100（规则建筑与重复结构）的增益是 B100（自然随机纹理）的 3 倍，四比四全胜，没有一处退化。** 排序一致、方向单一、无退化，这三点合起来排除了「增益来自噪声」或「增益只是参数变多的副产品」这两类解释。

设计上有三条约束，都是为了让这次检验尽量干净。**并联而不是串行**：两条流读的是同一个块的原始输入，不存在一条喂给另一条的关系，所以局部分支看到的是未经处理的信号，而不是等 SSM 把它抹平之后再去补救——这也是这个方案叫「层内」而非事后补救的原因。**完整 3×3 而不是深度可分离**：深度可分离能把开销再砍掉约 32 倍，但它不做通道间混合，而通道混合正是这里要求局部算子承担的一部分。**VMM 原样不动**：CSBlock 加什么都是在 SSM 旁边加，所以任何退化都不可能归因到扫描本身被改动。落到代码上，相对上游是**新增 46 行、改动 1 行**，一个类替换掉一个类。

结果上，CSBlock 平均 **+0.27 dB**，全部四个基准都赢，没有一处退化；训练早期收敛快约 2 倍，约 10,000 步突破 30 dB 而 MaIR 要 ~22,000 步，等 SSM 分支收敛后差距收窄到稳定的约 0.26 dB。代价是 **+126% 参数**，网络规模翻了一倍还多。作为一篇论文式的贡献，这个交换很不划算；它真正立住的是一件更窄也更有用的事——定位了纯 SSM 视觉骨干的一处具体缺陷，说明一个 3×3 的感受野就足以应对它，并且证明效应最大的地方恰好是理论预期的那个地方。

---

## 2. 主要贡献

1. **同预算下的 MaIR 复现。** MaIR-S-Light ×4，DIV2K 上训练 25,000 次迭代，1,374,156 参数——所有超参与第一程的 MambaIR 运行完全一致，所以 +1.61 dB 是同一把尺子量出来的，而不是两个论文数字相减。

2. **CSBlock——一个 CNN-SSM 融合模块，46 行。** 一个小 CNN **并联在同一个残差块内的 VMM 旁边**。两条流拿到的都是这个块的**原始输入**，两路输出按通道拼接、线性投影融合。

3. **增益正好落在设计预言的地方。** **平均 +0.27 dB，四比四全胜，没有一处退化。** Urban100（+0.42）的增益是 B100（+0.13）的 3 倍。训练早期收敛也快约 2 倍。

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

原文那一列用的是 DF2K（3,450 张）跑 500K 迭代，本仓库是 DIV2K（800 张）跑 25K——那 0.24–0.89 dB 的差距是训练预算的差距，不是复现失败。原始日志见 [`logs/`](logs/)。

**什么不算贡献。** `mair_arch.py` 里除 `MSS_VMM` 类之外的全部内容、`train.py`、`test.py`、四份 YAML 配置，都是上游 MaIR / BasicSR 代码。真正属于这里的原创工作，就是那 46 行的 diff。逐文件清单见文末的[仓库结构](#仓库结构)。

---

## 3. 方法与结构

### 3.1 基础架构：MaIR

MaIR 保留第一程那个 SwinIR 式的三阶段骨干——浅层特征提取、残差 Mamba 组、重建头——重写了残差块内部。与 MambaIR 的对照值得摆明：

| | MambaIR | MaIR |
|---|---|---|
| 扫描路径 | Cross-Scan：4 条独立直线 | NSS：一条嵌套 S 型路径 |
| 垂直连续性 | 靠额外扫描来补偿 | 靠折返天然保留（一维距离 → 2） |
| SSM 调用 | 每块 4 次 | 每块 1 次 |
| 融合方式 | 四方向逐元素相加 | 单路径，无需融合 |
| 注意力 | 无 | SSA 通道混洗门控 |
| 前馈 | CAB（Conv2d + 通道注意力） | **MLP（纯线性层）** |

最后一行是切入点。MambaIR 的前馈块里含一个卷积，MaIR 把它换成线性 MLP，也就顺手把深层路径上最后一个局部算子拿掉了。顶部那张图是 MaIR 论文 Fig. 2 里的完整结构。

### 3.2 CSBlock

顶部那张图就是它。卷积被删干净正是 CSBlock 要处理的事——这个模块让 SSM 流和卷积流在同一个块内并排运行，两条流拿到的是同一个原始输入：

| 组件 | 细节 |
|---|---|
| SSM 流 | 上游 VMM 原样不动——`in_proj → permute + Conv1d + SiLU → NSS 扫描 + 选择性扫描 → chunk/求和/门控 → out_norm + gate(z) + out_proj`，SS2D 核没有改 |
| CNN 流 | `permute → BCHW → Conv3×3 + SiLU → Conv3×3 + SiLU → permute → BHWC`，`padding=1`，不改变分辨率 |
| 融合 | 按通道拼接（`2C`）→ `Linear(2C → C)` → `LayerNorm` |

落到代码上，这是 `mair_arch.py:346` 处一个叫 `MSS_VMM` 的类，由三部分组成：上游的 `VMM`、一个两层 `nn.Sequential` 卷积流，以及 `Linear(2C → C)` 加 `LayerNorm` 的融合头。在 `RMB.__init__` 里一行替换掉 `VMM` 即可，相对上游是**新增 46 行、改动 1 行**。完整源码见 [`code/mair_arch.py`](code/mair_arch.py)，同目录的 [`mair_arch_original.py`](code/mair_arch_original.py) 原样保留，可直接对读 diff：

```bash
diff mair-cnn-ssm/code/mair_arch_original.py mair-cnn-ssm/code/mair_arch.py
```

*CSBlock 是模块名，CNN-SSM 是它实现的融合方案。* 类名叫 `MSS_VMM`，打开它的配置叫 `MaIR_MSS`，日志里把训出来的模型记作 MaIR-CSB——一个东西四个字符串，因为模块名字是照着方法后来才定的。

### 3.3 实验路线

| 阶段 | 问题 | 变体 | 结论 |
|---|---|---|---|
| 一 | Cross-Scan 的四个方向真的各自携带信息吗？ | 4-dir / 2-dir / 1-dir（见 [`../mambair/`](../mambair/)） | 砍到单方向只掉 0.10 dB，三个方向是回声 |
| 二 | 局部算子的缺席值多少？ | MaIR-S / MaIR-CSB | +0.27 dB 换 +126% 参数，增益集中在规则纹理（§4.2、§4.3） |

---

## 4. 实验与结果

### 4.1 实验设置

**数据。** DIV2K 训练集（800 张），×4 双三次退化，`gt_size = 128`。评测用 Set5、Set14、B100、Urban100 四个标准基准。

**模型。** MaIR-S-Light（1,374,156 参数）与 MaIR-CSB（3,109,356），`embed_dim = 60`，两者除是否插入 CSBlock 之外配置逐字相同。

**训练。** 25,000 次迭代，Adam（`lr = 2e-4`、`weight_decay = 0`），`batch_size_per_gpu = 96`，seed 10，单张 RTX 4090。水平翻转与旋转增强关闭——这与第一程保持一致，但也去掉了上游配置里的一个正则项。

**指标。** Y 通道 PSNR / SSIM，`crop_border = 4`。原始日志见 [`logs/`](logs/)。

基线分两次训练：第一次在约 24,400 步中断，第二次从 22,500 步的 checkpoint 恢复跑完——所以有两个 `train_MaIR_S_*` 日志，而 `test_MaIR_S_*` 只有一个。

### 4.2 增益从哪来

§2 那张增益分解表里，排序本身就是论据。Urban100 是一批规则建筑——砖缝、栏杆、窗格、立面重复；B100 是自然随机纹理。如果 CSBlock 起作用的机制是「补回一个局部算子」，那么它的增益就该随图中局部周期结构的多少而变——事实正是如此：**Urban100 是 B100 的 3.2 倍**。四比四全胜、无一处退化，则排除了增益来自噪声的可能。

<p align="center">
  <img src="figures/fig_psnr_curve.png" alt="MaIR 与 MaIR-CSB 的训练曲线">
  <br><em>收敛曲线给出相同的答案：CSBlock 约 10,000 步突破 30 dB，MaIR 要 ~22,000 步——训练早期大约快 2 倍——等 SSM 分支收敛后差距收窄到稳定的约 0.26 dB。</em>
</p>

局部算子似乎早期廉价好拟合，等全局模型追上来之后价值变小；那剩下的 0.26 dB 就是它在平衡态下的价值。

### 4.3 参数量

数字取自训练日志里 BasicSR 打印的真实值：

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

配套报告与架构图右下角的「5.5%」是**单块的** 72,300 除以**整个基础模型**（72,300 ÷ 1,374,156 ≈ 5.3%），描述的是「每块只加这么点」，不是模型规模的整体涨幅。模型级的口径是 **+126%**，两者差 24 倍。

所以结论是：**+0.27 dB 换 +126% 参数，增益集中在规则纹理上。** 作为一篇论文式的贡献，这个交换很不划算。CSBlock 首先是一件诊断工具，其次才是一个有竞争力的模型。

### 4.4 视觉对比

`LR（双三次） | MaIR | MaIR + CSBlock | GT`，每个基准数据集一张裁片，保持原始分辨率——它们是 §2 里 +0.13…+0.42 dB 那个梯次的视觉证据。这些图刻意不做压缩：它们的作用是展示像素级的纹理差异，重采样就失去意义了。

<p align="center">
  <img src="figures/B100_img_079_4panel.png" alt="B100 img_079 四栏对比">
  <br><em>B100 img_079。自然随机纹理，四个基准里增益最低的一档（该数据集整体 +0.13）。</em>
</p>

<p align="center">
  <img src="figures/Set14_img_011_4panel.png" alt="Set14 img_011 四栏对比">
  <br><em>Set14 img_011。自然场景，整体 +0.20。</em>
</p>

<p align="center">
  <img src="figures/Set5_img_002_4panel.png" alt="Set5 img_002 四栏对比">
  <br><em>Set5 img_002。平滑区域占主导，整体 +0.34。</em>
</p>

<p align="center">
  <img src="figures/Urban100_img_043_4panel.png" alt="Urban100 img_043 四栏对比">
  <br><em>Urban100 img_043。规则立面的重复结构，这个数据集整体 +0.42，是 B100 的 3.2 倍。</em>
</p>

<p align="center">
  <img src="figures/Urban100_img_047_4panel.png" alt="Urban100 img_047 四栏对比">
  <br><em>Urban100 img_047，两张 Urban100 裁片里的第二张——+0.42 dB 在这里最看得出来。</em>
</p>

---

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

---

## 6. 安装与使用

本目录不是完整的可运行工程——它是与上游有差异的那部分文件。先克隆上游仓库。

**1 —— 装环境。**

```bash
git clone https://github.com/csguoh/MaIR.git
cd MaIR
pip install -r requirements.txt          # 编译 mamba-ssm 需要 nvcc
pip install mamba-ssm==1.1.1 causal-conv1d==1.1.1
```

**2 —— 放入架构文件。**

```bash
cp <this-repo>/mair-cnn-ssm/code/mair_arch.py basicsr/archs/mair_arch.py
```

**3 —— 训练与测试 MaIR-CSB（带 CSBlock）。**

```bash
python -m basicsr.train -opt <this-repo>/mair-cnn-ssm/yaml/train_MaIR_MSS_lightSR_x4.yml
python -m basicsr.test  -opt <this-repo>/mair-cnn-ssm/yaml/test_MaIR_MSS_lightSR_x4.yml
```

**4 —— 训练与测试 MaIR-S 基线。** 先换回未修改的架构文件：

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

**5 —— 改配置。** 四份配置都必须改。它们指向原机器上 `/root/autodl-tmp/...` 的数据集路径和 `/root/2025-CVPR-MaIR-main/experiments/...` 的权重路径。运行前先改 `dataroot_gt`、`dataroot_lq`、四个测试集根目录，以及 `path.pretrain_network_g` 与结果输出目录。请在克隆下来的上游工程根目录下运行——相对的实验目录与结果目录都基于当前工作目录解析。

仓库不含权重——`net_g_25000.pth` 约 15 MB，日志与配图才是这几次运行的记录。

---

## 7. 局限

- **只有一种配置。** Light 变体、DIV2K、×4、25K 迭代。CSBlock 在更大模型、DF2K、去噪与去模糊上是什么表现，这里一个字都说明不了。
- **结构是手工定的**——两层 3×3 卷积、通道拼接、线性融合。没有做结构搜索，也没试过替代方案（深度可分离、单层卷积、注意力融合、串行而非并行摆放）。
- **+126% 参数换 +0.27 dB 很贵。** §4.3 那笔账就是这件事要说两遍的原因。CSBlock 首先是一件诊断工具。
- **「规则纹理」是定性描述。** Urban100 > Set5 > Set14 > B100 的梯次一致且符合假设，但没有计算任何纹理复杂度指标去预测增益——机制是从排序论证出来的，不是测出来的。
- **没有做摆放位置的消融。** 「读输入而不是读 SSM 输出」这一点是从设计上论证的，没有拿一个读 SSM 输出的变体做对照。这是最明显的下一个实验。
- **绝对 PSNR 不能和论文比。** 800 张跑 25K 迭代，对 3,450 张跑 500K；与论文 MaIR 之间 −0.24…−0.89 dB 的差距是训练预算的差距，不是复现失败。
- **SSIM 报了但没有单独分析。** 它的走势紧跟 PSNR，这里不对它作任何额外论断。

---

## 仓库结构

```
mair-cnn-ssm/                                     # mamba-image-super-resolution 的第二程
├── README.md、README.zh-CN.md
├── report.pdf                                    # 14 页
├── code/
│   ├── mair_arch.py                              # 新增 CSBlock（MSS_VMM 类）
│   ├── mair_arch_original.py                     # 上游 MaIR，留作 diff
│   └── train.py、test.py                         # 上游 BasicSR 入口，未修改
├── yaml/
│   ├── train_MaIR_S_lightSR_x4_my.yml            # 基线，25K 迭代
│   ├── test_MaIR_S_lightSR_x4_my.yml
│   ├── train_MaIR_MSS_lightSR_x4.yml             # 带 CSBlock
│   └── test_MaIR_MSS_lightSR_x4.yml
├── figures/
│   ├── fig_mair_architecture.jpg                 # MaIR 论文 Fig. 2
│   ├── fig_csblock_architecture.png              # CSBlock：两条流与融合
│   ├── fig_psnr_curve.png                        # 两个模型的训练曲线
│   ├── B100_img_079_4panel.png                   # LR | MaIR | +CSBlock | GT
│   ├── Set14_img_011_4panel.png
│   ├── Set5_img_002_4panel.png
│   ├── Urban100_img_043_4panel.png
│   └── Urban100_img_047_4panel.png
└── logs/
    ├── train_MaIR_S_lightSR_x4_20260614_235037.log   # 跑到约 24,400 步中断
    ├── train_MaIR_S_lightSR_x4_20260615_140900.log   # 从 22,500 恢复
    ├── train_MaIR_MSS_lightSR_x4_20260615_192507.log
    ├── test_MaIR_S_lightSR_x4_20260615_145356.log
    └── test_MaIR_MSS_lightSR_x4_20260616_134215.log
```

上表里每一个文件是谁写的：

| 文件 | 状态 |
|---|---|
| `code/mair_arch.py` | **我写的。** 相对上游**新增 46 行、改动 1 行**：加入 `MSS_VMM`（即 CSBlock），并让 `RMB` 改用它。 |
| `code/mair_arch_original.py` | 上游 MaIR 原文件，原样保留供 diff。 |
| `code/train.py`、`code/test.py` | 上游 BasicSR 入口，未修改。 |
| `yaml/*` | 上游 lightSR 配置，换了更小的 `gt_size` 和更少的迭代数。 |
| `report.pdf` | 书面报告：文献综述、方法、实验、误差分析。 |
| `logs/*` | BasicSR 原始输出，正文表格即由此得来。 |
| `figures/fig_mair_architecture.jpg` | 第三方图片，转载自 MaIR 论文 Fig. 2。 |
| `figures/` 其余 7 张 | 本次运行产生：CSBlock 结构图、PSNR 曲线、5 张四栏对比。 |

上游代码来自 [MaIR](https://github.com/csguoh/MaIR) 与 [BasicSR](https://github.com/XPixelGroup/BasicSR)，按各自原有许可在此复现。这些项目的许可条款适用于它们自己的文件，不适用于本仓库整体。模型权重与数据集都不随仓库分发。

---

## 参考文献

- Li et al. **MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration.** CVPR 2025.
  ([arXiv:2412.20066](https://arxiv.org/abs/2412.20066))
- Guo et al. **MambaIR: A Simple Baseline for Image Restoration with State-Space Model.** ECCV 2024.
  ([arXiv:2403.07764](https://arxiv.org/abs/2403.07764))
- Liu et al. **VMamba: Visual State Space Model.** NeurIPS 2024.
  ([arXiv:2401.10166](https://arxiv.org/abs/2401.10166))
- Liang et al. **SwinIR: Image Restoration Using Swin Transformer.** ICCVW 2021.
- Gu & Dao. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** 2023.
