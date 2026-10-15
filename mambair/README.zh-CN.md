# MambaIR：复现与扫描方向消融

**四个扫描方向里有三个是回声**

[English](README.md) | [中文](README.zh-CN.md) | [← 仓库根目录](../README.zh-CN.md)

> 本仓库第一程。完整报告见 [`report.pdf`](report.pdf)（13 页）。第二程——MaIR 复现与 CSBlock——在 [`../mair-cnn-ssm/`](../mair-cnn-ssm/)。

<p align="center">
  <img src="figures/fig_mambair_architecture.png" alt="MambaIR 三阶段架构">
  <br><em>MambaIR 的整体结构：RSSG 堆叠（内含 RSSB 块，图 a）、替代自注意力的 VSSM（图 b），以及它内部的二维选择性扫描模块（图 c）。复现的结果是 924,492 参数、四个基准平均 26.52 dB。</em>
</p>

---

## 1. 项目背景

单图像超分辨率从一张低分辨率观测重建高分辨率图像，是一个病态逆问题：同一张 LR 输入对应很多张合理的 HR 图像，所以任务本质是学一个关于自然图像结构的先验，用它把正确的那一张挑出来。承载这个先验的序列模型长期由注意力主导，代价是 O(N²)；[MambaIR](https://github.com/csguoh/MambaIR)（ECCV 2024）把状态空间模型作为自注意力的直接替代品搬进来——架构仍是 SwinIR 式的三阶段骨干，只是每个 transformer 块换成了 RSSB，自注意力换成了 VSSM。这套东西能成立靠的是复杂度：Mamba 的选择性递推是 O(N)，一个隐状态把整个前缀压缩进来，模型以线性代价拿到全局感受野，这正是高分辨率修复任务想要的交换。

<p align="center">
  <img src="figures/fig1_complexity.png" alt="线性复杂度对二次复杂度">
  <br><em>当初为什么选 SSM 骨干：O(N) 对注意力的 O(N²)，而且差距随序列长度——也就是高分辨率图像实际产生的长度——不断拉大。</em>
</p>

它成立得不够干净，也还是因为复杂度。状态空间模型定义在**一维序列**上，而图像是二维网格。MambaIR 的答案是 **Cross-Scan**：沿四个不同方向把特征图展平，各扫一遍，再把四个结果逐元素相加。这个赌注是——四条一维顺序加起来，足以近似它替代掉的那个卷积本来就免费提供的二维邻域结构。**本仓库直接检验这个赌注**，方法是在冻结的权重上、只改前向里求和几个方向：从四个方向砍到一个，四个基准平均只掉 **0.10 dB**。如果四个扫描真是四个互相独立的视角，砍掉 75% 远不止掉这么点。

三个原因解释了这一点，按重要性递增：选择性机制本身抹平了方向差异（`B(x)`、`C(x)`、`Δ(x)` 都是输入的函数，同样的内容过滤过程会在四条顺序上各跑一遍，关键关系集中到传播得最好的那个方向）；逐元素相加没有任何区分手段（`y = y1 + y2 + y3 + y4` 把四个视角当成等权，而几个近似相同序列的和不过是一个标量倍数）；根因是**一维展平在扫描开始之前就已经破坏了二维连续性**——64×64 的特征图里，竖直相邻的像素在行主序下相隔 64 个 token，状态必须跨过 64 个无关 token 才能维持这个关系，换一种顺序扫描能部分补偿，但补偿就是补偿而已。

这个结论就是第二程的动机：既然「怎么遍历序列」不是瓶颈，「把二维结构展平再重建」这件事本身才是——那就应该把局部算子直接放回来，而不是靠更多扫描顺序去补偿。MaIR 顺着这条路用一条嵌套的 S 型扫描替代了四条直线，第二程在它的基础上验证了补回卷积的价值。

<p align="center">
  <img src="figures/fig_sr_comparison.jpg" alt="真实场景超分示例">
  <br><em>这个任务在真实拍摄图片上（而不是双三次退化的基准集上）长什么样：三组裁片的 LQ | MambaIR。配套报告开篇用的示例图。</em>
</p>

---

## 2. 主要贡献

1. **同预算下的复现。** MambaIR-Light ×4，DIV2K 训练 25,000 次迭代，924,492 参数。所有超参与第二程的运行完全一致，所以两者可以逐行对比。

2. **一个把 Cross-Scan 冗余单独拎出来的扫描方向消融。** Cross-Scan 四个方向的求和是 `mambair_arch.py:379-381` 那一行。把它降到两个方向、再降到一方向，重新跑推理——这是对扫描机制的一次干净隔离：不重训，也就没有「换个方式训练会学到什么」这个混淆项。

3. **结论：四个方向砍到一个，四个基准平均只掉 0.10 dB。** 代价只有 0.005 SSIM。MambaIR 买了四次扫描，拿到的是一个方向的信息量。

只推理，用同一个训练好的权重，唯一变量是 `forward_core` 求和几个方向：

| 配置 | 参与方向 | Set5 | Set14 | B100 | Urban100 | 平均 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 4-dir（MambaIR 默认） | 前行 + 反行 + 转置 + 反对角 | 29.29 | 26.73 | 26.30 | 23.74 | **26.52** |
| 2-dir | 前行 + 反行 | 29.19 | 26.71 | 26.27 | 23.70 | 26.47 |
| 1-dir | 仅前行 | 29.13 | 26.66 | 26.23 | 23.65 | 26.42 |

| | Set5 | Set14 | B100 | Urban100 | 平均 |
|---|:---:|:---:|:---:|:---:|:---:|
| Δ 4 → 2 | −0.10 | −0.02 | −0.03 | −0.04 | **−0.05** |
| Δ 4 → 1 | −0.16 | −0.07 | −0.07 | −0.09 | **−0.10** |

为什么是这个结果见 §4.2，逐次运行的原始日志见 [`logs/`](logs/)。

**什么不算贡献。** `mambair_arch.py`、`train.py`、`test.py` 都是上游 MambaIR 与 BasicSR 代码，与各自仓库逐字节一致。配置是上游 lightSR 配置，换了更小的 `gt_size` 和更少的迭代数。逐文件清单见文末的[仓库结构](#仓库结构)。

---

## 3. 方法与结构

### 3.1 基础架构：MambaIR 与 Cross-Scan

MambaIR 的整体是 SwinIR 式的三阶段骨干——浅层特征提取、若干残差状态空间组（RSSG）的堆叠、高质量重建头。每个 RSSG 由若干 RSSB 组成，RSSB 内部是 `LayerNorm → VSSM → LayerNorm → MLP`，VSSM 就是替换掉自注意力的那个模块。

Cross-Scan 发生在 VSSM 内部：

<p align="center">
  <img src="figures/fig_cross_scan_merge.png" alt="Cross-scan、SS blocks、Cross-merge">
  <br><em>VSS 模块拿到 H×W 的特征图，展平四次——沿行、沿列、沿两条对角线——对每条一维序列独立跑一个 SSM，再把四路输出折回二维逐元素相加。设计意图是：行扫描捕获水平依赖，列扫描捕获垂直依赖，两条对角线补上缝隙，四个合起来覆盖网格上全部邻域关系。</em>
</p>

就是最后那一步求和，成了本仓库的全部切入点：它只有一行代码，所以改它不需要重训，也就把「扫描机制」从「训练过程」里干净地隔离了出来。图片来源：这张 Cross-Scan / S6 / Cross-Merge 流水线是 VMamba 论文里的二维选择性扫描（SS2D）示意图，cross-scan 这个方案出自该文，MambaIR 在它的 VSSM 里照搬了它。

### 3.2 消融怎么做

本目录没有提出新模块，原创工作是一次受控的推理期改造。`forward_core` 里四个方向的输出本来是逐元素相加，把它换成只累加前 *k* 个方向，*k* 取 4、2、1，其余一切不动——同一份权重、同一套测试脚本、同一个 `crop_border=4`。

这样做的取舍是明确的：它把机制干净地单独拎了出来，但也说明不了「单方向模型如果就这么训练」会学到什么。比较的是一次训练的前向，不是四个模型。这条限制写在 §7。

<p align="center">
  <img src="figures/fig_ssm_mechanism.jpg" alt="二维网格展平为一维序列后的 SSM 递推">
  <br><em>消融为什么应该给出这个结果：SSM 维护 h(t) = A·h(t-1) + B·x(t)，输出 y(t) = C·h(t)，一个隐状态承载整个前缀。但把二维网格展平，竖直相邻的两个像素在一维顺序里被拉得很远（橙色那一对），递推只能拿状态容量去表达这段距离，而不是表达那个关系。</em>
</p>

### 3.3 实验路线

| 阶段 | 问题 | 变体 | 结论 |
|---|---|---|---|
| 一 | Cross-Scan 的四个方向真的各自携带信息吗？ | 4-dir / 2-dir / 1-dir，只改前向，权重冻结 | 砍到单方向只掉 0.10 dB，三个方向是回声（§4.2） |
| 二 | 既然瓶颈不在遍历顺序，那在哪？ | —— 交给 [`../mair-cnn-ssm/`](../mair-cnn-ssm/) | 在「展平重建」本身，补回局部算子值 +0.27 dB |

---

## 4. 实验与结果

### 4.1 实验设置

**数据。** DIV2K 训练集（800 张），×4 双三次退化，`gt_size = 128`。评测用 Set5、Set14、B100、Urban100 四个标准基准。

**模型。** MambaIR-Light，924,492 参数，`embed_dim = 60`。

**训练。** 25,000 次迭代（约 30 个 epoch），Adam（`lr = 2e-4`、`weight_decay = 0`），`batch_size_per_gpu = 96`，seed 10，单张 RTX 4090。水平翻转与旋转增强关闭——这让与第二程的对比更简单，但也去掉了上游配置里的一个正则项。

**指标。** Y 通道 PSNR / SSIM，`crop_border = 4`。原始日志见 [`logs/`](logs/)。

### 4.2 为什么四个方向是冗余的

§2 的表只说了一半。三个原因，按重要性递增：

**选择性机制本身抹平了方向差异。** Mamba 的 `B(x)`、`C(x)`、`Δ(x)` 都是输入的函数，这正是它区别于 S4 这类静态 SSM 的地方。面对相同的内容，同样的内容过滤过程会在四条顺序上各跑一遍：关键的重建关系会集中到传播得最好的那个方向，其余方向收敛成它的近似。四个方向不是四个专家，而是一个方向加三个回声。

**逐元素相加没有任何区分手段。** `y = y1 + y2 + y3 + y4` 把四个视角当成等权。没有门控、没有可学习权重、没有注意力——模型没法学会更信某个方向，只能把它们加起来，而几个近似相同序列的和，不过是一个标量倍数。

**一维展平早就破坏了二维连续性。** 这是根因。在 64×64 的特征图里，竖直相邻的像素在行主序下相隔 64 个 token，状态必须跨过 64 个无关 token 才能维持这个关系。换一种顺序扫描能部分补偿，但补偿就是补偿而已——这件事在你想明白「展平发生在扫描**开始之前**」之后就不奇怪了。

### 4.3 训练动态与复现质量

DIV2K 验证集 PSNR 从随机初始化的 10.61 dB 起步，前 2,500 次迭代就到 27.19 dB——用整轮 10% 的预算走了 16.58 dB。这是隐式全局建模在早期发力：粗糙的 LR→HR 映射几乎立刻就建立了，剩下 22,500 次迭代一共值 1.02 dB。

<p align="center">
  <img src="figures/training_progression.png" alt="训练演进">
  <br><em>同一块裁片在不同训练阶段的样子：LR 双三次、随机初始化，以及 2.5K / 5K / 10K / 15K / 25K 迭代。这条曲线前半段的陡峭和后半段的平坦，就是上面那组数字的视觉版本。</em>
</p>

### 4.4 参数量

训练日志在构建模型时打印了它的规模：

```
INFO: Network: MambaIR, with parameters: 924,492
```

配套报告在六处写到 MambaIR-Light 约有 1,303,401 个参数（≈1.3M）。那个数来自上游的 MambaIR-S 配置，不是本次实际构建的 Light 变体；本仓库的所有数字以日志打印的 **924,492** 为准。

---

## 5. 技术栈

| | |
|---|---|
| 操作系统 | Ubuntu 22.04 |
| 显卡 | RTX 4090（显存扩展后 49 GB） |
| Python | 3.10 |
| PyTorch | 2.1.0+cu121 |
| TorchVision | 0.16.0+cu121 |
| CUDA | 12.1 |
| BasicSR | 1.3.5（随 MambaIR 分支提供） |
| 算子库 | `mamba-ssm` 1.1.1 + `causal-conv1d` 1.1.1，用 `nvcc` 编译 |

注意这与 [`../mair-cnn-ssm/`](../mair-cnn-ssm/) 是**两套不同的环境**——那边跟着 MaIR 分支走，用 PyTorch 2.4.0 + BasicSR 1.2.0。两个阶段运行时不共享任何东西，各自端到端复现各自的阶段。

---

## 6. 安装与使用

本目录不是完整的可运行工程——它是与上游有差异的那部分文件。先克隆上游仓库。

**1 —— 装环境。**

```bash
git clone https://github.com/csguoh/MambaIR.git
cd MambaIR
pip install -r requirements.txt          # 编译 mamba-ssm 需要 nvcc
pip install mamba-ssm==1.1.1 causal-conv1d==1.1.1
```

**2 —— 放入本仓库的文件。**

```bash
cp <this-repo>/mambair/code/mambair_arch.py   basicsr/archs/
cp <this-repo>/mambair/code/train.py          basicsr/train.py
cp <this-repo>/mambair/code/test.py           basicsr/test.py
```

**3 —— 训练。** 产物写到 `./experiments/`。

```bash
python -m basicsr.train -opt <this-repo>/mambair/code/train_MambaIR_lightSR_x4_my.yml
```

**4 —— 在四个基准上测试。** 结果写到 `./results/`。

```bash
python -m basicsr.test -opt <this-repo>/mambair/code/test_MambaIR_lightSR_x4_my.yml
```

**5 —— 改配置。** 配置指向原机器上 `/root/autodl-tmp/datasets/DIV2K/...` 的路径。运行前先改 `dataroot_gt`、`dataroot_lq` 和四个测试集根目录。请在克隆下来的上游工程根目录下运行——相对的实验目录与结果目录都基于当前工作目录解析。

仓库不含权重——25K 迭代的权重大约 15 MB，日志与配图才是这次运行的记录。

---

## 7. 局限

- **消融只做了推理。** 它把扫描机制干净地单独拎了出来，但也说明不了「单方向模型如果就这么训练」会学到什么。比较的是一次训练的前向，不是四个模型。
- **只有一种配置。** Light 变体、DIV2K、×4、25K 迭代。冗余这个发现在此处被测到，并未证明能推广到更大的模型或更长的训练。
- **没有纹理复杂度指标。** 「残差集中在规则高频结构上」这个说法是从图上读出来的，不是测出来的。
- **绝对 PSNR 不能和论文比。** 800 张跑 25K 迭代，对 3,450 张跑 500K；论文的 MambaIR-Light 远高于这里的数字。
- **关闭了数据增强。** 这与第二程、以及训练演进图保持一致，但确实偏离了上游配置，牺牲了一部分泛化性。

---

## 仓库结构

```
mambair/                                          # mamba-image-super-resolution 的第一程
├── README.md、README.zh-CN.md
├── report.pdf                                    # 13 页
├── code/
│   ├── mambair_arch.py                           # 上游 MambaIR，未修改
│   ├── train.py、test.py                         # 上游 BasicSR 入口，未修改
│   └── train_MambaIR_lightSR_x4_my.yml           # 本次运行使用的配置
├── figures/                                      # 6 张配图，见下
│   ├── fig_mambair_architecture.png              # MambaIR 论文 Fig. 2
│   ├── fig_cross_scan_merge.png                  # VMamba SS2D 示意图
│   ├── fig_ssm_mechanism.jpg                     # 2D→1D 展平与 SSM 递推
│   ├── fig_sr_comparison.jpg                     # 真实场景超分示例
│   ├── fig1_complexity.png                       # O(N) 对 O(N²)
│   └── training_progression.png                  # 同一块裁片的 7 个训练阶段
└── logs/
    ├── train_MambaIR_lightSR_x4_20260512_133105.log
    └── test_MambaIR_lightSR_x4_benchmark_20260512_194615.log
```

上表里每一个文件是谁写的：

| 文件 | 状态 |
|---|---|
| `code/mambair_arch.py` | 上游 MambaIR，与上游仓库逐字节一致。 |
| `code/train.py`、`code/test.py` | 上游 BasicSR 入口，未修改。 |
| `code/train_MambaIR_lightSR_x4_my.yml` | 上游 lightSR 配置，换了更小的 `gt_size` 和更少的迭代数。 |
| `report.pdf` | 书面报告：文献综述、方法、实验、误差分析。 |
| `logs/*` | BasicSR 原始输出，正文表格即由此得来。 |
| `figures/fig_mambair_architecture.png`、`fig_cross_scan_merge.png` | 第三方图片，转载自 MambaIR 与 VMamba 论文。 |
| `figures/` 其余 4 张 | 本次运行产生。 |

上游代码来自 [MambaIR](https://github.com/csguoh/MambaIR) 与 [BasicSR](https://github.com/XPixelGroup/BasicSR)，按各自原有许可在此复现。这些项目的许可条款适用于它们自己的文件，不适用于本仓库整体。模型权重与数据集都不随仓库分发。

---

## 参考文献

- Guo et al. **MambaIR: A Simple Baseline for Image Restoration with State-Space Model.** ECCV 2024.
  ([arXiv:2403.07764](https://arxiv.org/abs/2403.07764))
- Liu et al. **VMamba: Visual State Space Model.** NeurIPS 2024.
  ([arXiv:2401.10166](https://arxiv.org/abs/2401.10166))
- Liang et al. **SwinIR: Image Restoration Using Swin Transformer.** ICCVW 2021.
- Gu & Dao. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** 2023.
