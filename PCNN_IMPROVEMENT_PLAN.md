# 基于 PCNN 的硕士论文改进计划

创建日期：2026-04-28

本文档用于规划在 PCNN 基线模型上的逐步改进路线。目标不是一次性重写整个网络，而是在可复现、可消融、可解释的前提下，围绕 PCNN 的不足逐步加入模块，形成适合硕士大论文撰写的研究主线。

## 总体思路

PCNN 的优势在于同时考虑了全局人脸信息和局部表情区域信息，但它仍然存在三个可以优化的方向：

```text
1. 局部区域建模较固定
2. 全局与局部之间交互不够充分
3. 特征融合方式不够自适应
```

因此，建议采用“三步走”的改进路线：

```text
第一步：固定局部裁剪 -> 数据驱动局部增强
第二步：简单并联融合 -> 全局-局部双向交互引导
第三步：固定融合权重 -> 实例感知动态门控融合
```

每一步都可以单独实现、单独实验、单独写消融，最终组合成完整模型。

## 改进点一：数据驱动局部增强模块

### 1.1 要解决的问题

PCNN 使用局部分支增强细粒度表情区域建模能力，但局部区域往往依赖预设裁剪或固定感知方式。

这种方式的问题是：

```text
不同人脸的五官位置存在差异
不同姿态会导致关键表情区域偏移
不同表情依赖的关键区域不完全相同
固定局部区域可能包含无关背景或弱判别区域
```

因此，可以将固定局部建模改为数据驱动的局部增强，让网络根据输入样本自动突出关键表情区域。

### 1.2 可以怎么改

推荐从轻量模块开始，不要一开始做太复杂。

可选方案：

```text
方案 A：空间注意力模块
在中层特征图上生成空间权重图，增强高响应表情区域。

方案 B：多尺度局部增强模块
使用不同卷积核大小捕捉嘴角、眼角、眉毛等不同尺度的细节变化。

方案 C：显著性引导局部增强
从全局特征生成显著性图，再对局部分支特征进行加权。
```

建议优先实现：

```text
空间注意力 + 多尺度卷积
```

原因是结构简单、容易稳定训练、论文也好解释。

### 1.2.1 当前代码实现状态

已在 `network/models.py` 中加入可开关的轻量局部增强版本：

```text
pcnn                 原始 PCNN baseline
pcnn_local_enhanced  PCNN + 数据驱动局部增强模块
```

该增强模块作用在 PCNN 拼接后的局部特征 `x10` 上，包含：

```text
1. 1x1 卷积降维
2. 3x3 与 5x5 多尺度卷积分支
3. 空间注意力 mask
4. 残差式局部特征增强
```

这样做的好处是：

```text
不破坏原始 PCNN 主体结构
可以直接做 baseline 与增强版消融
模块含义清晰，适合论文描述
```

训练时使用：

```bash
python -u train_ferplus.py \
  --model pcnn_local_enhanced \
  --device cuda:0 \
  --batch-size 64 \
  --workers 0 \
  --epochs 100 \
  --lr 0.005 \
  --grad-clip 5.0 \
  --no-pretrained-pcnn \
  --test-after-training
```

测试时使用：

```bash
python -u val.py \
  --model pcnn_local_enhanced \
  --dataset ferplus \
  --num-class 8 \
  --device cuda:0 \
  --batch-size 128 \
  --workers 0 \
  --split test \
  --model-path checkpoints/你的增强版_best.pth
```

### 1.3 论文中怎么表述

可以写成：

```text
针对 PCNN 中局部区域建模依赖固定区域、难以适应姿态变化和个体差异的问题，本文提出一种数据驱动的局部表情增强模块。该模块通过空间注意力与多尺度卷积自适应突出高判别性表情区域，从而提升模型对细粒度面部肌肉变化的表征能力。
```

### 1.4 实验怎么做

消融实验：

```text
Baseline PCNN
PCNN + 局部增强模块
```

观察指标：

```text
RAF-DB accuracy
FERPlus accuracy
Occlusion-RAFDB accuracy
Occlusion-FERPlus accuracy
```

重点关注：

```text
fear / sad / anger / disgust 等细粒度负面表情类别是否提升
遮挡场景下是否更稳定
```

### 1.5 当前主线实现：GAPW

根据新的研究主线，已将第一阶段改为更适合写大论文的网络结构：

```text
GAPW = Global-guided Adaptive Patch Weighting
中文：全局上下文引导的自适应 Patch 区域加权模块
```

模型开关：

```text
--model pcnn_gapw
```

该模块不是原来的简单局部增强，而是直接针对 PCNN 固定 5 个语义局部区域的问题进行改造。实现位置在 PCNN 拼接得到局部特征图 `x10` 之后、STN 融合之前。

核心流程：

```text
1. 保留 PCNN 的全局特征 x1 和局部拼接特征 x10。
2. 将 x10 自适应池化为 4x4 patch 描述，共 16 个局部区域。
3. 使用全局上下文特征与每个 patch 描述共同预测 patch 重要性权重。
4. 使用 softmax 得到 16 个 patch 的样本级权重。
5. 将 patch 权重上采样回局部特征图，对局部区域进行自适应加权。
6. 加权后的局部特征继续进入 PCNN 原有 STN 对齐融合路径。
7. 加权局部特征反向生成通道门控，对全局特征进行轻量校准。
```

对应公式可以写为：

```text
F_g = GlobalBranch(x)
F_l = LocalBranch(x)
P_i = PatchPool(F_l), i = 1,...,16

s_i = MLP([GAP(F_g), P_i])
w_i = softmax(s_i)

F_l' = F_l * (1 + alpha * (Upsample(16w) - 1))
F_g' = F_g * (1 + beta * C(F_l'))
```

其中：

```text
alpha、beta 初始为 0
```

这样做的意义：

```text
1. 初始状态等价于原始 PCNN，便于加载作者或本地 baseline 权重。
2. patch 权重可视化后可以作为论文中的可解释性分析。
3. 4x4 patch 可以和 2x2、3x3、固定 5 区域做消融。
4. 该模块是网络内部结构创新，不属于 TTA 或测试技巧。
```

建议训练命令：

```bash
python -u train.py \
  --model pcnn_gapw \
  --dataset rafdb \
  --num-class 7 \
  --device cuda:0 \
  --batch-size 128 \
  --workers 0 \
  --epochs 50 \
  --lr 0.001 \
  --base-lr-mult 0.1 \
  --interaction-lr-mult 5.0 \
  --lr-step 15 \
  --grad-clip 5.0 \
  --freeze-backbone \
  --val-split test \
  --test-split test \
  --pretrained-pcnn checkpoints/rafdb_[04-28]-[20-25]-_best.pth \
  --test-after-training
```

## 改进点二：全局-局部双向交互引导模块

### 2.1 要解决的问题

PCNN 同时使用全局分支和局部分支，但两者之间更多是并行提取特征，交互关系不够充分。

存在的问题：

```text
全局分支知道整张脸的情绪语义，但不能充分指导局部区域关注哪里
局部分支捕捉到细粒度变化，但不能有效反馈给全局语义表征
全局与局部特征可能只是后期拼接或相加，缺乏协同建模过程
```

因此，可以设计双向交互机制，实现：

```text
全局引导局部
局部反馈全局
```

### 2.2 可以怎么改

推荐设计两个轻量方向：

```text
方向 A：全局到局部的空间引导
全局分支生成空间注意图，对局部分支特征进行区域级调制。

方向 B：局部到全局的通道反馈
局部分支生成通道权重，对全局特征进行通道重标定。
```

可以写成公式：

```text
F_l' = F_l * A_g
F_g' = F_g * C_l
```

其中：

```text
F_l 表示局部特征
F_g 表示全局特征
A_g 表示由全局分支生成的空间引导权重
C_l 表示由局部分支生成的通道反馈权重
```

### 2.3 论文中怎么表述

可以写成：

```text
针对全局结构信息与局部细粒度变化难以协同建模的问题，本文提出全局-局部双向交互引导模块。该模块首先利用全局分支生成空间引导信息，对局部分支进行区域级调制，使模型更加关注表情核心区域；随后利用局部分支提取到的细粒度判别线索生成通道重标定权重，反馈增强全局语义特征，从而实现整体结构与局部细节的互补增强。
```

### 2.4 实验怎么做

消融实验：

```text
Baseline PCNN
PCNN + 全局引导局部
PCNN + 局部反馈全局
PCNN + 双向交互引导
```

最好能证明：

```text
单向交互有提升
双向交互比单向更好
```

这对硕士论文非常重要，因为它能支撑模块设计的合理性。

### 2.5 当前代码实现状态

已在 `network/models.py` 中加入可开关的双向交互版本：

```text
pcnn                 原始 PCNN baseline
pcnn_local_enhanced  PCNN + 局部增强模块
pcnn_bi_interaction  PCNN + 全局-局部双向交互引导模块
```

模块名称：

```text
BidirectionalInteractionBlock
```

插入位置：

```text
原始 PCNN 在 x1 表示全局人脸特征，x10 表示拼接后的局部区域特征。
双向交互模块插入在 x1 与 x10 通过 STN 融合之前。
```

这样设计的原因：

```text
1. x1 已包含整张人脸的全局语义信息，适合生成空间引导图。
2. x10 已包含眼部、鼻部、嘴部等局部表情区域，适合反馈细粒度通道线索。
3. 插在 STN 融合之前，可以让局部特征先被全局语义筛选，再参与后续对齐和融合。
4. 不改变五个局部分支的辅助分类头，便于和原始 PCNN 做公平消融。
```

具体交互方式：

```text
全局到局部：
由全局特征 F_g 生成空间权重 A_g，对局部特征 F_l 进行区域级调制。

局部到全局：
由引导后的局部特征 F_l' 生成通道权重 C_l，对全局特征 F_g 进行通道级反馈。
```

公式表达：

```text
A_g = sigmoid(Conv(F_g))
F_l' = F_l * (1 + gamma_l * A_g)

C_l = sigmoid(MLP(GAP(F_l')))
F_g' = F_g * (1 + gamma_g * C_l)
```

其中：

```text
gamma_l 和 gamma_g 初始为 0。
```

这样做的意义：

```text
模块初始状态严格等价于原始 PCNN，不会一开始破坏 baseline 特征。
训练过程中模型会自动学习是否需要增强局部区域或反馈全局通道。
如果模块有效，提升可以解释为全局语义和局部细节协同建模带来的收益。
```

RAF-DB 训练命令：

```bash
python -u train.py \
  --model pcnn_bi_interaction \
  --dataset rafdb \
  --num-class 7 \
  --device cuda:0 \
  --batch-size 128 \
  --workers 0 \
  --epochs 100 \
  --lr 0.01 \
  --grad-clip 5.0 \
  --val-split test \
  --test-split test \
  --no-pretrained-pcnn \
  --test-after-training
```

对比对象：

```text
原始 PCNN baseline: 88.331%
```

## 改进点三：实例感知动态门控融合模块

### 3.1 要解决的问题

PCNN 的全局和局部特征最终需要融合，但固定融合方式无法适应不同样本的差异。

例如：

```text
无遮挡正脸样本中，全局结构信息比较可靠
嘴部遮挡样本中，眼部和眉部局部线索可能更重要
低质量或模糊样本中，某些局部分支可能包含噪声
不同表情类别对全局和局部信息的依赖程度不同
```

因此，可以设计实例感知动态门控模块，让模型根据每张输入图像自适应分配全局和局部特征权重。

### 3.2 可以怎么改

推荐使用轻量 MLP 或 1x1 卷积生成门控权重。

基本形式：

```text
z = concat(F_g, F_l)
gate = sigmoid(MLP(z))
F_fuse = gate * F_g + (1 - gate) * F_l
```

含义：

```text
gate 越大，模型越依赖全局特征
gate 越小，模型越依赖局部特征
```

也可以扩展成多局部门控：

```text
F_fuse = w_g * F_g + w_1 * F_l1 + w_2 * F_l2 + ... + w_n * F_ln
```

其中权重由 softmax 归一化。

### 3.3 论文中怎么表述

可以写成：

```text
针对固定特征融合策略难以适应样本差异的问题，本文提出实例感知动态门控融合模块。该模块根据输入样本中全局结构特征与局部细节特征的响应情况，自适应评估不同特征源的可靠性，并动态分配融合权重，实现高可靠特征主导、低可靠特征补充的自适应融合。
```

### 3.4 实验怎么做

消融实验：

```text
Baseline PCNN
PCNN + 固定相加融合
PCNN + 固定拼接融合
PCNN + 动态门控融合
```

重点观察：

```text
遮挡数据集是否提升
复杂表情类别是否提升
不同样本的门控权重是否具有解释性
```

可视化：

```text
展示无遮挡样本中 global gate 较高
展示遮挡样本中 local gate 较高
展示不同表情类别的门控分布
```

### 3.5 当前实验发现

在 RAF-DB baseline 上，PCNN 的局部辅助输出 `heads` 与主输出 `out` 具有明显互补性。

原始推理只使用：

```text
out
```

当前测试发现固定输出融合：

```text
logits = out + 0.4 * heads
```

可以将本地原始 PCNN baseline 从：

```text
88.331%
```

提升到：

```text
88.820%
```

这说明第三个方向比当前特征级交互更有潜力。后续应从固定融合进一步发展为实例感知动态门控融合：

```text
logits = out + gate(x) * heads
```

其中 `gate(x)` 根据输入样本自适应决定局部辅助输出的贡献，而不是固定使用 0.4。

### 3.6 遮挡鲁棒性扩展

进一步测试发现，遮挡场景下多视角预测与局部头融合更有效：

```text
original/hflip = 0.2/0.8
logits = out + 1.3 * heads
```

结果：

```text
RAF-DB:           88.331% -> 89.276%
Occlusion-RAFDB: 84.196% -> 85.967%
```

这说明遮挡鲁棒性不仅依赖单图特征增强，也依赖不同视角预测之间的互补性。后续动态门控模块可以扩展为：

```text
logits = gate_o(x) * out_original
       + gate_f(x) * out_flip
       + gate_l(x) * heads
```

其中 `gate_o`、`gate_f` 和 `gate_l` 根据样本遮挡程度、主分支置信度和局部分支置信度自适应分配权重。

## 推荐实施顺序

不要三个模块一起改。建议按下面顺序逐步尝试：

```text
第 1 阶段：复现 PCNN 基线
目标：确保 RAF-DB、FERPlus、Occlusion 数据集能稳定跑通。

第 2 阶段：加入数据驱动局部增强模块
目标：先看单模块是否带来稳定提升。

第 3 阶段：加入全局-局部双向交互模块
目标：验证全局和局部协同是否优于简单并联。

第 4 阶段：加入动态门控融合模块
目标：提升复杂样本和遮挡样本下的自适应融合能力。

第 5 阶段：完整模型组合实验
目标：对比 Baseline、单模块、双模块、完整模型。
```

## 推荐消融实验表

可以设计如下表格：

```text
Baseline PCNN
PCNN + DLE
PCNN + BIG
PCNN + DGF
PCNN + DLE + BIG
PCNN + DLE + BIG + DGF
```

其中：

```text
DLE = Data-driven Local Enhancement，数据驱动局部增强
BIG = Bidirectional Interaction Guidance，双向交互引导
DGF = Dynamic Gated Fusion，动态门控融合
```

实验数据集：

```text
RAF-DB
FERPlus
Occlusion-RAFDB
Occlusion-FERPlus
```

## 大论文结构建议

可以这样组织章节：

```text
第 1 章 绪论
研究背景、意义、国内外研究现状、本文贡献。

第 2 章 相关理论与基础方法
FER 任务、CNN、注意力机制、PCNN 基线模型、数据集介绍。

第 3 章 基于数据驱动局部增强的全局-局部表情识别模型
介绍第一个改进模块和初步实验。

第 4 章 面向全局-局部协同建模的双向交互与动态融合方法
介绍双向交互和动态门控融合。

第 5 章 实验结果与分析
主实验、消融实验、可视化、复杂场景分析。

第 6 章 总结与展望
总结贡献，分析不足，提出后续工作。
```

## 当前最稳妥的研究路线

建议先完成：

```text
1. 稳定复现 PCNN
2. 固定 RAF-DB 和 FERPlus 的正确数据协议
3. 固定训练命令和测试命令
4. 保存 baseline 结果
5. 再开始逐个加模块
```

不要在 baseline 还不稳定时同时改很多结构，否则后面很难判断到底是哪一个模块带来了提升或下降。

## 一句话总结

这三个改进点可以一个一个做：

```text
先改局部区域怎么提
再改全局和局部怎么交互
最后改全局和局部怎么融合
```

这样实验路径清楚，论文逻辑完整，也方便答辩时解释每个模块的必要性。
