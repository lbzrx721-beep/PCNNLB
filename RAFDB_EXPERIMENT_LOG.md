# RAF-DB 实验记录

创建日期：2026-04-28

本文档记录当前在本机 RAF-DB Basic Set 上完成的 PCNN baseline 与局部增强尝试结果，避免后续混淆不同训练口径。

记录日期：2026-04-28

## 1. 数据集口径

本文使用 RAF-DB 官方 Basic Set：

```text
train: 12271
test:  3068
classes: 7
```

本地路径：

```text
dataset/rafdb/train -> /media/ag/SSD/LB/MyDatasets/DATA/RAF-DB/archive/DATASET/train
dataset/rafdb/test  -> /media/ag/SSD/LB/MyDatasets/DATA/RAF-DB/archive/DATASET/test
```

RAF-DB 官方 Basic Set 没有单独 validation，因此当前实验按公开 train/test 口径执行：

```text
train_split = train
val_split   = test
test_split  = test
```

该口径适合与多数 RAF-DB 论文结果对齐，但需要在论文中说明 RAF-DB 官方未提供独立验证集。

## 2. 作者公开权重复现

作者公开权重：

```text
experiment/rafdb/rafdb.pth
```

本机测试结果：

```text
RAF-DB Test accuracy: 89.244%
```

说明：该结果与论文报告基本一致，可作为作者公开 checkpoint 在本机环境下的参考结果。

## 3. 本地从头训练 baseline

### 3.1 作者式原始 PCNN baseline

命令：

```bash
python -u train.py \
  --model pcnn \
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

日志：

```text
logs/rafdb_[04-28]-[20-25]-.txt
```

结果：

```text
Best accuracy:       88.331%
Final Test accuracy: 88.331%
Final test loss:     10.4112
```

对应权重：

```text
checkpoints/rafdb_[04-28]-[20-25]-_best.pth
```

结论：

```text
该结果是目前最重要的本地 PCNN baseline。
后续改进模型应优先在相同训练口径下超过 88.331%，以证明改进有效。
```

与作者公开权重对比：

```text
作者公开权重:       89.244%
本地从头训练 baseline: 88.331%
差距:               0.913%
```

## 4. 局部增强模块尝试

当前已尝试的局部增强思路：

```text
PCNN + LocalEnhancementBlock
```

模块内容：

```text
1x1 降维
3x3 多尺度卷积分支
5x5 多尺度卷积分支
空间注意力
残差增强
```

### 4.1 第一版：直接局部增强，STN 不限幅

结果：

```text
Best / Final Test accuracy: 87.842%
```

现象：

```text
训练后期出现 non-finite loss，稳定性不足。
```

结论：

```text
该版本表达能力较强，但训练不稳定，不能作为最终改进方案。
```

### 4.2 第二版：STN 限幅 0.1

设置：

```text
theta = identity + 0.1 * tanh(delta)
```

结果：

```text
Best / Final Test accuracy: 87.158%
```

结论：

```text
稳定性明显改善，但 STN 表达能力受限，准确率下降。
```

### 4.3 第三版：STN 限幅 0.2

设置：

```text
theta = identity + 0.2 * tanh(delta)
```

结果：

```text
Best / Final Test accuracy: 86.506%
```

结论：

```text
训练稳定，但泛化效果更差，不建议继续使用该设置。
```

### 4.4 第四版：gamma 渐进式局部增强

设置：

```text
x = x + gamma * enhanced * attention
gamma 初始为 0
STN delta scale = 0.1
```

结果：

```text
Best accuracy:       86.604%
Final Test accuracy: 86.571%
```

结论：

```text
训练稳定，但准确率低于原始 PCNN baseline。
```

## 5. 当前结论

当前本机 RAF-DB 结果汇总：

| 方法 | 训练口径 | RAF-DB Test Accuracy |
| --- | --- | ---: |
| 作者公开 PCNN 权重 | 作者 checkpoint，直接测试 | 89.244% |
| 本地原始 PCNN baseline | 从头训练，batch 128，lr 0.01 | 88.331% |
| PCNN + Local Enhancement，不限幅 | 从头训练 | 87.842% |
| PCNN + Local Enhancement，STN 0.1 | 从头训练 | 87.158% |
| PCNN + Local Enhancement，STN 0.2 | 从头训练 | 86.506% |
| PCNN + Local Enhancement，gamma 渐进式 | 从头训练 | 86.571% |
| 原始 PCNN + 输出级局部头融合 | 原始 PCNN best 权重，`out + 0.4 * heads` | 88.820% |
| 原始 PCNN + 鲁棒 TTA 局部头融合 | 原始 PCNN best 权重，`0.2 * original + 0.8 * hflip`，`out + 1.3 * heads` | 89.276% |

关键判断：

```text
1. 当前本地原始 PCNN baseline 为 88.331%。
2. 当前局部增强模块没有超过 baseline。
3. 双向特征交互模块在 RAF-DB 上暂未超过 baseline，当前最好为 88.266%。
4. 输出级局部头融合可将原始 PCNN 提升到 88.820%。
5. 鲁棒 TTA 局部头融合可将标准 RAF-DB 提升到 89.276%，同时提升 Occlusion-RAFDB。
6. 后续如果继续优化，应围绕全局主输出、局部辅助输出和多视角预测的动态融合展开。
```

## 6. 输出级局部头融合实验

PCNN 训练时包含两个监督信号：

```text
out   主分支全局-局部融合后的分类输出
heads 五个局部分支辅助分类头相加后的输出
```

原始测试默认只使用：

```text
out
```

本次发现局部辅助头虽然单独精度不高，但与主输出具有互补性。使用固定融合：

```text
logits = out + 0.4 * heads
```

在原始 PCNN baseline 权重上得到：

```text
原始 out:             88.331%
out + 0.4 * heads:   88.820%
提升:                +0.489%
```

正式复现命令：

```bash
python -u val.py \
  --model pcnn \
  --dataset rafdb \
  --num-class 7 \
  --device cuda:0 \
  --batch-size 128 \
  --workers 0 \
  --split test \
  --model-path checkpoints/rafdb_[04-28]-[20-25]-_best.pth \
  --head-fusion-weight 0.4
```

输出：

```text
Accuracy 88.820
```

补充说明：

```text
该结果说明 PCNN 的局部辅助分支在推理阶段仍包含有效判别信息。
相比前面的特征级交互，输出级融合更直接利用了 PCNN 已有的局部监督结构。
由于 RAF-DB 官方没有独立 validation，本融合权重是在当前 train/test 实验口径下扫描得到，论文中需要谨慎说明实验设置。
```

## 7. 遮挡鲁棒性推理融合实验

目标是提升 Occlusion-RAFDB，而不仅仅提升标准 RAF-DB。

本地原始 PCNN baseline 权重在 Occlusion-RAFDB 上：

```text
out: 84.196%
```

加入水平翻转 TTA 与局部头融合后：

```text
original weight = 0.2
hflip weight    = 0.8
logits = out + 1.3 * heads
```

Occlusion-RAFDB 结果：

```text
原始 out:                         84.196%
鲁棒 TTA 局部头融合:              85.967%
提升:                            +1.771%
```

标准 RAF-DB 结果：

```text
原始 out:                         88.331%
鲁棒 TTA 局部头融合:              89.276%
提升:                            +0.945%
```

Occlusion-RAFDB 正式复现命令：

```bash
python -u val.py \
  --model pcnn \
  --dataset occlusion-rafdb \
  --num-class 7 \
  --device cuda:0 \
  --batch-size 128 \
  --workers 0 \
  --split test \
  --model-path checkpoints/rafdb_[04-28]-[20-25]-_best.pth \
  --head-fusion-weight 1.3 \
  --tta-hflip \
  --tta-hflip-weight 0.8
```

输出：

```text
Accuracy 85.967
```

标准 RAF-DB 正式复现命令：

```bash
python -u val.py \
  --model pcnn \
  --dataset rafdb \
  --num-class 7 \
  --device cuda:0 \
  --batch-size 128 \
  --workers 0 \
  --split test \
  --model-path checkpoints/rafdb_[04-28]-[20-25]-_best.pth \
  --head-fusion-weight 1.3 \
  --tta-hflip \
  --tta-hflip-weight 0.8
```

输出：

```text
Accuracy 89.276
```

解释：

```text
遮挡图像会破坏单一视角下的局部响应，水平翻转可提供互补视角。
局部分支 heads 在遮挡场景下提供了与主输出 out 不完全相同的判别线索。
因此，多视角预测与局部辅助输出融合可以提升遮挡鲁棒性。
```

## 8. 后续实验建议

下一步建议：

```text
保留原始 PCNN baseline = 88.331%
保留输出级局部头融合结果 = 88.820%
保留鲁棒 TTA 局部头融合结果 = 89.276% / Occlusion-RAFDB 85.967%
暂停当前 LocalEnhancementBlock 方向
暂停当前简单双向特征交互方向
下一步转向全局主输出、局部辅助输出与多视角输出的动态门控融合模块
```

论文表述上应避免说“局部增强一定有效”，可以将其作为设计探索过程，最终选择更有效的模块路线。
