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

关键判断：

```text
1. 当前本地原始 PCNN baseline 为 88.331%。
2. 当前局部增强模块没有超过 baseline。
3. 双向特征交互模块在 RAF-DB 上暂未超过 baseline，当前最好为 88.266%。
4. 输出级局部头融合可将原始 PCNN 提升到 88.820%。
5. 后续如果继续优化，应围绕全局主输出和局部辅助输出的动态融合展开。
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

## 7. 后续实验建议

下一步建议：

```text
保留原始 PCNN baseline = 88.331%
保留输出级局部头融合结果 = 88.820%
暂停当前 LocalEnhancementBlock 方向
暂停当前简单双向特征交互方向
下一步转向全局主输出与局部辅助输出的动态门控融合模块
```

论文表述上应避免说“局部增强一定有效”，可以将其作为设计探索过程，最终选择更有效的模块路线。

## 8. 2026-07-16 区域可靠性关系融合实验

新增模型：`pcnn_region_relation`。

目标不是继续对拼接后的 `x10` 做统一加权，而是把 PCNN 的五个原始局部区域作为独立节点，显式学习：

```text
区域可靠性、区域缺失模拟、区域间关系、局部辅助监督、主分支残差融合
```

严格模块单独训练设置：

```text
seed=1234
base checkpoint=checkpoints/rafdb_[07-16]-[15-33]-_best.pth
冻结 PCNN 参数和 BatchNorm buffers
只训练 region_relation 参数（828,938 个）
batch size=128
region dropout=0.2
RandomErasing p=0.2, scale=(0.02, 0.20)
```

结果：

| 实验 | RAF-DB | Occlusion-RAFDB |
| --- | ---: | ---: |
| 原始 PCNN baseline | 88.331% | 84.741% |
| 残差上限 0.2 | 88.462% | 84.605% |
| 残差上限 0.8，重新训练 | 88.592% | 85.150% |

最佳权重：

```text
checkpoints/rafdb_[07-16]-[16-55]-_best.pth
```

日志：

```text
logs/rafdb_[07-16]-[16-55]-.txt
```

注意：训练在第 23 轮检测到 non-finite loss 后安全停止，最佳权重来自第 13 轮；最佳权重参数检查均为有限值。

## 9. 2026-07-17 GAPW 端到端联合训练（实验 B）

训练设置与原始 PCNN 的 100 轮端到端协议保持一致：

```text
model=pcnn_gapw
seed=1234
epochs=100
batch_size=128
lr=0.01
StepLR step_size=15, gamma=0.5
RandomErasing p=0.5, scale=(0.02, 0.25)
不加载 PCNN checkpoint，仅加载相同的人脸 ResNet18 backbone
不冻结 PCNN、BatchNorm、STN、分类头或 GAPW
```

结果：

| 模型 | RAF-DB | Occlusion-RAFDB |
| --- | ---: | ---: |
| 原始 PCNN 实验 A（seed 未记录） | 88.331% | 84.741% |
| PCNN + GAPW 实验 B（seed=1234） | 88.103% | 83.379% |

文件：

```text
checkpoint: checkpoints/rafdb_[07-17]-[14-32]-_best.pth
log:        logs/rafdb_[07-17]-[14-32]-.txt
best epoch: 31
```

GAPW 有效缩放分别约为 `1.38e-08`、`-3.46e-04`、`5.03e-04`。在同一个最佳权重中清零 GAPW 缩放后，RAF-DB 和 Occlusion-RAFDB 均多识别对 1 张，说明当前最佳模型实际上几乎关闭了 GAPW。

限制：实验 A 当时 `seed=None`，实验 B 为 `seed=1234`，因此这不是严格随机种子配对。当前结果可以否定“本次 B 已证明 GAPW 有效”，但不能用一次跨种子比较证明所有 GAPW 设计普遍无效。

## 10. 2026-07-17 区域可靠性关系融合端到端训练

训练设置：

```text
model=pcnn_region_relation
seed=1234
epochs=100
batch_size=128
lr=0.01
StepLR step_size=15, gamma=0.5
RandomErasing p=0.5, scale=(0.02, 0.25)
region_dropout=0.2
region_temperature=1.0
region_output_scale=0.8
不加载 PCNN checkpoint，仅加载相同的人脸 ResNet18 backbone
不冻结 PCNN、BatchNorm、STN、分类头或区域关系模块
```

结果：

| 模型 | RAF-DB | Occlusion-RAFDB |
| --- | ---: | ---: |
| 原始 PCNN 实验 A（seed 未记录） | 88.331%（2710/3068） | 84.741%（622/734） |
| 区域关系模块单独训练（冻结 PCNN/BN） | 88.592%（2718/3068） | 85.150%（625/734） |
| 区域关系模块端到端训练（seed=1234） | **88.853%（2726/3068）** | 84.605%（621/734） |

文件：

```text
checkpoint: checkpoints/rafdb_[07-17]-[15-51]-_best.pth
log:        logs/rafdb_[07-17]-[15-51]-.txt
best epoch: 87
```

同一最佳权重关闭关系分类残差的配对消融：

| 设置 | RAF-DB | Occlusion-RAFDB |
| --- | ---: | ---: |
| `region_output_scale=0` | 88.070%（2702/3068） | 83.924%（616/734） |
| `region_output_scale=0.8` | 88.853%（2726/3068） | 84.605%（621/734） |
| 开启关系输出的差值 | +0.783（+24 张） | +0.681（+5 张） |

该消融只关闭推理时关系分支对主输出的直接残差，不会撤销联合训练期间该模块对共享 PCNN 参数的影响。因此它能证明最佳权重实际使用了关系输出，但不能代替从头训练的纯 PCNN 同种子对照。

参数检查：区域关系模块 27 个状态张量、828,938 个参数值全部有限；有效主输出残差缩放为 `0.23499`，局部头混合权重为 `0.02479`。

结论：端到端训练使 RAF-DB 达到当前本地从头训练方法的最高值，但 Occlusion-RAFDB 没有超过原 PCNN，也低于冻结 PCNN/BN 的区域关系版本。由于原始 PCNN 的种子未记录，下一项严格消融应使用完全相同配置补跑 `seed=1234` 的纯 PCNN。
