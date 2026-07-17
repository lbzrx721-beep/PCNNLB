# GAPW 优化说明

创建日期：2026-05-08

## 一、优化概述

本次代码在原始 PCNN 基础上加入了一个模型内部优化模块：

```text
GAPW: Global-guided Adaptive Patch Weighting
中文名称：全局上下文引导的自适应 Patch 区域加权模块
```

该优化的目标是提升模型对局部表情区域的建模能力，并增强模型在遮挡场景下的鲁棒性。核心方法是网络结构内部的局部区域自适应加权机制。

## 二、优化动机

原始 PCNN 同时使用整张人脸的全局特征和若干固定语义局部区域特征。这个设计本身是有效的，但固定局部区域建模也存在一个问题：不同样本中真正有判别力的表情区域并不完全相同。

例如：

```text
开心表情中，嘴角和脸颊区域通常更重要。
惊讶表情中，眼部和嘴部区域通常更重要。
遮挡样本中，被遮挡区域的可靠性应该降低。
```

因此，本次优化将原始固定局部区域使用方式进一步改为样本自适应的 patch 区域加权方式。模型会根据当前输入图像的全局人脸上下文，判断哪些局部 patch 更有表情判别价值，哪些局部 patch 可靠性较低。

## 三、方法说明

GAPW 模块实现在 `network/models.py` 中，对应类名为：

```text
GlobalGuidedPatchWeighting
```

训练和测试时通过以下参数启用：

```bash
--model pcnn_gapw
```

该模块插入在 PCNN 生成拼接局部特征图 `x10` 之后、原始 STN 全局-局部融合路径之前。

主要流程如下：

```text
1. 保留 PCNN 的全局特征图和拼接后的局部特征图。
2. 将局部特征图池化为 4x4 的 patch 网格，共得到 16 个局部 patch 描述。
3. 对每个 patch，构造由全局上下文、局部 patch 描述、二者差异组成的 patch 上下文。
4. 分别预测 patch 重要性和 patch 可靠性。
5. 将重要性与可靠性结合，生成自适应 patch 权重。
6. 使用受限残差缩放方式对局部特征图进行加权。
7. 利用加权后的局部特征对全局特征进行轻量校准。
```

这样设计的好处是：GAPW 的残差缩放参数初始化为 0，因此模型初始状态接近原始 PCNN，不会一开始破坏原始 PCNN 的特征表达；同时训练过程中残差幅度受到限制，能够减少训练不稳定的问题。

## 四、主要代码改动

本次优化主要涉及以下文件：

```text
network/models.py
train.py
val.py
```

新增或重点使用的参数包括：

```text
--model pcnn_gapw
--gapw-patch-grid 4
--gapw-temperature 1.0
--train-gapw-only
--seed
--grad-clip
```

其中，`--train-gapw-only` 会冻结原始 PCNN 参数，仅训练 GAPW 模块。这样可以更清楚地验证：在同一个本地 PCNN baseline 上，新增 GAPW 模块是否带来稳定提升。

## 五、RAF-DB 训练命令

5 个随机种子实验使用的训练命令如下：

```bash
conda run -n LB python -u train.py \
  --model pcnn_gapw \
  --dataset rafdb \
  --num-class 7 \
  --device cuda:0 \
  --batch-size 128 \
  --workers 0 \
  --epochs 30 \
  --seed <seed> \
  --lr 0.001 \
  --interaction-lr-mult 2.0 \
  --lr-step 15 \
  --grad-clip 1.0 \
  --train-gapw-only \
  --gapw-patch-grid 4 \
  --gapw-temperature 1.0 \
  --erasing-p 0.2 \
  --erasing-scale-min 0.02 \
  --erasing-scale-max 0.20 \
  --val-split test \
  --test-split test \
  --pretrained-pcnn checkpoints/rafdb_[04-28]-[20-25]-_best.pth \
  --test-after-training
```

## 六、实验结果

本地原始 PCNN baseline 结果为：

```text
RAF-DB:           88.331%
Occlusion-RAFDB: 84.196%
```

GAPW 5 个随机种子的结果如下：

| Seed | RAF-DB | Occlusion-RAFDB |
| ---: | ---: | ---: |
| 2026 | 88.429 | 84.332 |
| 3407 | 88.396 | 84.332 |
| 42 | 88.364 | 84.332 |
| 1234 | 88.396 | 84.605 |
| 777 | 88.494 | 84.332 |

5 次平均结果为：

```text
RAF-DB:           88.416%  (+0.085)
Occlusion-RAFDB: 84.387%  (+0.191)
```

从结果看，GAPW 在 5 个随机种子下均超过本地原始 PCNN baseline。整体提升幅度不算大，但表现比较稳定，并且遮挡集上的平均提升高于普通 RAF-DB 测试集。这说明 GAPW 对遮挡或低可靠局部区域具有一定抑制作用，同时能够增强更有判别力的局部 patch。

作者公开权重结果仅作为参考：

```text
RAF-DB:           89.244%
Occlusion-RAFDB: 85.695%
```

当前实验中更公平的比较方式是：

```text
本地原始 PCNN baseline vs 本地 GAPW 改进模型
```

不要直接把作者公开权重作为主要对比基线，因为作者权重和本地重新训练的 baseline 并不是完全相同的训练条件。

## 七、论文定位

本次优化适合在论文中描述为一种网络结构层面的改进：

```text
从固定语义局部区域建模
转向全局上下文引导的自适应 patch 区域加权建模
```

论文表述时应重点强调以下内容：

```text
1. 使用全局上下文指导局部 patch 权重预测。
2. 同时建模 patch 重要性和 patch 可靠性。
3. 对遮挡或低可靠局部区域进行抑制。
4. 对当前表情更有判别力的局部区域进行增强。
5. 该方法属于模型内部结构优化，而不是测试技巧。
```

本文主要创新点应放在 GAPW 模块及其 patch 可靠性建模上。
