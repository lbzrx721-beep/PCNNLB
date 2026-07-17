# PCNN 当前实验状态总记录

创建日期：2026-04-28

记录日期：2026-04-28

本文档用于记录当前 PCNN 复现、数据集口径、作者权重测试、本地训练 baseline、已尝试优化模块和后续优化方向。后续继续实验时，优先以本文档作为当前状态依据。

## 2026-07-16 更新：区域可靠性关系融合

在调研近期 FER 工作后，新增独立模型开关：

```text
--model pcnn_region_relation
```

该版本复用 PCNN 已有的全局人脸特征和五个语义局部特征，不引入额外的人脸关键点、分割模型或视觉语言模型。主要借鉴方向如下：

```text
ORSANet (ACM MM 2025): 遮挡语义、跨区域交互、困难类别区分
ARPGNet (TAFFC 2025): 以图/注意力建模面部区域关系
CMNet (TIP 2026): 左右半脸互补和对称信息
NLA (AAAI 2025): 噪声感知与一致性训练思路
```

当前模块流程：

```text
全局特征 + 五个局部特征 -> 六个区域 token
每个局部区域与全局区域的一致程度 -> 样本级可靠性权重
训练时随机丢弃局部区域 -> 模拟区域缺失/遮挡
多头自注意力 -> 学习区域间补偿关系
可靠性加权局部 logits -> 直接进入局部辅助损失
关系分类残差 -> 进入最终主输出
```

固定种子 1234、冻结原 PCNN 参数、只训练新模块的结果：

| 方法 | RAF-DB | Occlusion-RAFDB | 说明 |
| --- | ---: | ---: | --- |
| 2026-07-16 本地 PCNN baseline | 88.331% | 84.741% | `rafdb_[07-16]-[15-33]-_best.pth` |
| 区域关系模块，残差上限 0.2 | 88.462% | 84.605% | 遮挡集未提升 |
| 旧权重测试时尺度诊断，上限 0.8 | 88.657% | 85.150% | 仅作诊断，不作为正式训练结果 |
| 区域关系模块，上限 0.8，重新训练 | **88.592%** | **85.150%** | `rafdb_[07-16]-[16-55]-_best.pth` |
| 作者 PCNN 权重 | 89.244% | 85.695% | 参考结果 |

相对同一本地 baseline，重新训练的新模块提升：

```text
RAF-DB:           +0.261 个百分点（多识别对 8 张）
Occlusion-RAFDB:  +0.409 个百分点（多识别对 3 张）
```

当前结论：

```text
1. 这是目前具有正向遮挡增益的候选结构，但提升仍小，不能直接宣称创新成立。
2. 残差上限 0.2 使新分支平均幅度仅约为原主分支的 2.7%，作用过弱。
3. 上限 0.8 能同时改善正常集和遮挡集，但本轮第 23 轮出现一次非有限损失；最佳第 13 轮权重本身全部有限。
4. RAF-DB 官方无独立验证集，当前选择最佳 epoch 使用 test，必须在论文中披露。
5. 下一步优先做 3 个种子、FERPlus/Occlusion-FERPlus 和模块消融，再决定是否作为论文主创新。
```

## 2026-07-17 更新：GAPW 端到端联合训练（实验 B）

为验证 GAPW 是否只是受“冻结 PCNN”限制，已从人脸 ResNet18 预训练权重开始，对 `PCNN + GAPW` 完成 100 轮端到端训练。PCNN 主干、STN、分类头、BatchNorm 和 GAPW 均参与更新。

| 实验 | 种子 | RAF-DB | Occlusion-RAFDB |
| --- | ---: | ---: | ---: |
| A：原始 PCNN 端到端训练 | 未记录（`None`） | 88.331% | 84.741% |
| B：PCNN + GAPW 端到端训练 | 1234 | 88.103% | 83.379% |

实验 B 最佳权重来自第 31 轮：

```text
checkpoints/rafdb_[07-17]-[14-32]-_best.pth
logs/rafdb_[07-17]-[14-32]-.txt
```

最佳权重中 GAPW 的实际有效缩放：

```text
局部增强：   1.38e-08
Patch 加权：-3.46e-04
全局校准：   5.03e-04
```

在同一个 B 权重上将三个 GAPW 缩放清零：RAF-DB 从 2703/3068 变为 2704/3068，Occlusion-RAFDB 从 611/734 变为 612/734。GAPW 只改变了各 1 个预测，并且两次都使正确预测变错。

当前可支持的结论：

```text
1. 当前 GAPW 既没有即插即用增益，也没有在本次端到端训练中超过原 PCNN。
2. 网络在端到端训练中把 GAPW 有效缩放压到接近 0，说明当前结构/监督没有被有效利用。
3. A 当时没有固定种子，而 B 使用 1234，因此 A/B 不是严格种子配对；下降幅度不能全部归因于 GAPW。
4. 即使考虑种子限制，B 内部的“启用/清零 GAPW”对照仍表明最佳模型几乎没有使用 GAPW。
```

## 2026-07-17 更新：区域可靠性关系融合端到端训练

已完成 `pcnn_region_relation` 的 100 轮端到端训练。该实验没有使用 `--train-region-only` 或 `--freeze-backbone`，因此 PCNN 主干、BatchNorm、STN、五个局部分类头和区域关系模块均参与更新。

训练口径：

```text
seed=1234
epochs=100
batch_size=128
lr=0.01
StepLR step_size=15, gamma=0.5
RandomErasing p=0.5, scale=(0.02, 0.25)
region_dropout=0.2
region_output_scale=0.8
不加载 PCNN checkpoint，仅加载相同的人脸 ResNet18 backbone
```

结果：

| 模型 | RAF-DB | Occlusion-RAFDB |
| --- | ---: | ---: |
| 原始 PCNN baseline（seed 未记录） | 88.331%（2710/3068） | 84.741%（622/734） |
| 区域关系模块单独训练，冻结 PCNN/BN | 88.592%（2718/3068） | 85.150%（625/734） |
| 区域关系模块端到端训练，seed=1234 | **88.853%（2726/3068）** | 84.605%（621/734） |

最佳权重来自第 87 轮：

```text
checkpoint: checkpoints/rafdb_[07-17]-[15-51]-_best.pth
log:        logs/rafdb_[07-17]-[15-51]-.txt
```

同一端到端权重仅把 `region_output_scale` 设为 0，关闭关系分类残差后的配对对照：

| 设置 | RAF-DB | Occlusion-RAFDB |
| --- | ---: | ---: |
| 关闭关系输出 | 88.070%（2702/3068） | 83.924%（616/734） |
| 开启关系输出 | 88.853%（2726/3068） | 84.605%（621/734） |
| 关系输出直接贡献 | +0.783（+24 张） | +0.681（+5 张） |

检查点中的关系模块共 828,938 个参数，全部为有限值。有效主输出残差缩放约为 `0.2350`，局部头混合权重约为 `0.0248`。

当前判断：关系模块本身确实被使用，并非 GAPW 那种近零失效状态；但端到端联合训练削弱了遮挡泛化，使最终 Occlusion-RAFDB 仍比原 PCNN 少识别对 1 张。由于原 PCNN baseline 当时没有固定种子，端到端结果与 baseline 不是严格同种子配对；正式论文结论前仍需补跑 `seed=1234` 的纯 PCNN 对照。

## 1. 当前目标

当前研究目标不是单纯追作者公开权重的最高数值，而是在同一机器、同一数据集、同一训练和测试口径下，证明改进模型优于本地原始 PCNN baseline。

当前最重要的本地 baseline：

```text
RAF-DB Basic Set
原始 PCNN 从头训练
Best / Final Test accuracy: 88.331%
```

当前已确认有效的轻量优化：

```text
原始 PCNN + 输出级局部头融合
logits = out + 0.4 * heads
RAF-DB Test accuracy: 88.820%
相对原始 out 提升: +0.489%
```

当前已完成端到端验证、但尚未观察到有效增益的网络模块：

```text
pcnn_gapw
Global-guided Adaptive Patch Weighting
全局上下文引导的自适应 Patch 区域加权模块
```

该模块用于替代“固定 5 个语义局部区域直接拼接”的局部建模方式，在 PCNN 局部特征 `x10` 上构造 4x4 patch，并由全局上下文预测 16 个 patch 的重要性权重。

后续改进模型如果要证明有效，应优先在相同口径下超过：

```text
88.331%
```

作者公开权重在本机 RAF-DB test 上的结果：

```text
89.244%
```

这个结果可作为作者 checkpoint 的参考上界，但论文实验中更公平的比较方式是：

```text
本地原始 PCNN baseline vs 本地改进 PCNN
```

## 2. 数据集口径

### 2.1 RAF-DB

当前使用 RAF-DB 官方 Basic Set，也是多数论文常用的 RAF-DB 实验口径。

本地路径：

```text
dataset/rafdb/train -> /media/ag/SSD/LB/MyDatasets/DATA/RAF-DB/archive/DATASET/train
dataset/rafdb/test  -> /media/ag/SSD/LB/MyDatasets/DATA/RAF-DB/archive/DATASET/test
```

样本统计：

```text
train: 12271
test:  3068
class: 7
```

重要说明：

```text
RAF-DB Basic Set 官方没有单独 validation。
当前训练时使用 train 训练，使用 test 做 validation/test。
论文中需要说明 RAF-DB 按官方 train/test 划分进行实验。
```

当前命令中应显式传入：

```text
--val-split test
--test-split test
```

### 2.2 Occlusion-RAF-DB

本地路径：

```text
dataset/occlusion-rafdb/test -> /media/ag/SSD/LB/MyDatasets/DATA/Occlusion-RAF-DB/images_by_label
```

样本统计：

```text
734 images
7 classes
```

说明：

```text
原始 CSV 约 735 条，其中 test_2464_aligned 重复。
ImageFolder 实际读取唯一图片 734 张。
```

### 2.3 FERPlus

当前标准版本使用 Microsoft FERPlus 标注，并由 FER2013 官方 CSV/图像生成 ImageFolder。

当前推荐使用 argmax/voting 版本：

```text
/media/ag/SSD/LB/MyDatasets/FERPlus_ImageFolder_argmax
```

本地软链接：

```text
dataset/ferplus/train      -> /media/ag/SSD/LB/MyDatasets/FERPlus_ImageFolder_argmax/train
dataset/ferplus/validation -> /media/ag/SSD/LB/MyDatasets/FERPlus_ImageFolder_argmax/validation
dataset/ferplus/test       -> /media/ag/SSD/LB/MyDatasets/FERPlus_ImageFolder_argmax/test
```

样本统计：

```text
train:      28385
validation: 3555
test:       3545
class:      8
```

不要把 strict majority 版本作为主实验口径：

```text
/media/ag/SSD/LB/MyDatasets/FERPlus_ImageFolder_majority
```

原因：

```text
strict majority 版本样本更少、更干净，准确率容易虚高，不适合作为主论文口径。
```

### 2.4 Occlusion-FERPlus

本地路径：

```text
dataset/occlusion-ferplus/test -> /media/ag/SSD/LB/MyDatasets/DATA/Occlusion-FERPlus/images_by_label
```

样本统计：

```text
605 images
8 classes
```

类别：

```text
angry, contempt, disgust, fear, happy, neutral, sad, suprise
```

## 3. 作者公开权重测试结果

| 作者公开权重 | 正常测试集 | 正常准确率 | 遮挡测试集 | 遮挡准确率 |
| --- | --- | ---: | --- | ---: |
| `experiment/rafdb/rafdb.pth` | RAF-DB Test | 89.244% | Occlusion-RAF-DB | 85.695% |
| `experiment/ferplus/ferplus.pth` | FERPlus Test | 84.429% | Occlusion-FERPlus | 82.149% |

解释：

```text
RAF-DB 作者权重结果与论文基本一致，说明 RAF-DB 数据路径和测试流程可信。
FERPlus 作者权重在本地 argmax/voting 口径下为 84.429%，低于论文表格可疑数值。
FERPlus 论文正文提到 85.59%，作者 checkpoint 中 best_acc 也约为 85.5863。
论文表格中的 87.85 大概率是表格错误或不同实验口径。
```

## 4. 本地原始 PCNN baseline

当前最重要的本地 baseline 命令：

```bash
cd /home/ag/LB/PCNN
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

权重：

```text
checkpoints/rafdb_[04-28]-[20-25]-_best.pth
```

结果：

```text
Best accuracy:       88.331%
Final Test accuracy: 88.331%
Final test loss:     10.4112
```

结论：

```text
这是当前本地原始 PCNN 的主 baseline。
后续优化必须优先和这个结果比较。
```

## 5. 已尝试的局部增强优化

当前已实现可切换模型：

```text
--model pcnn                 原始 PCNN
--model pcnn_local_enhanced  PCNN + 局部增强模块
```

局部增强模块内容：

```text
1x1 降维
3x3 多尺度卷积分支
5x5 多尺度卷积分支
空间注意力
残差式增强
gamma 初始为 0 的渐进增强
```

实验结果：

| 方法 | RAF-DB Test Accuracy | 结论 |
| --- | ---: | --- |
| 原始 PCNN baseline | 88.331% | 当前主 baseline |
| Local Enhancement，不限幅 | 87.842% | 后期出现 non-finite loss |
| Local Enhancement，STN 0.1 | 87.158% | 稳定但下降 |
| Local Enhancement，STN 0.2 | 86.506% | 稳定但更差 |
| Local Enhancement，gamma 渐进式 | 86.571% | 稳定但低于 baseline |

当前判断：

```text
局部增强模块暂时没有超过 baseline。
它可以保留为失败尝试或消融记录，但不适合作为最终主创新点。
后续不要继续盲目堆这个模块。
```

## 6. 当前代码状态

关键文件：

```text
network/models.py
train.py
val.py
PCNN_IMPROVEMENT_PLAN.md
RAFDB_EXPERIMENT_LOG.md
AUTHOR_WEIGHTS_EVAL_RESULTS.md
```

当前功能：

```text
train.py 支持 --model pcnn / pcnn_local_enhanced
train.py 支持 --model pcnn_bi_interaction
train.py 支持 --test-after-training
train.py 支持 --grad-clip
train.py 支持 --stop-on-nonfinite
train.py 支持 --val-split 和 --test-split
train.py 通过 --dataset ferplus 使用 FERPlus argmax/voting 数据
val.py 可测试 RAF-DB、FERPlus、Occlusion-RAF-DB、Occlusion-FERPlus
```

重要提醒：

```text
如果要跑作者式原始网络，必须使用 --model pcnn。
如果使用 --model pcnn_local_enhanced，就是带局部增强的实验网络。
如果使用 --model pcnn_bi_interaction，就是带全局-局部双向交互引导的实验网络。
```

## 7. 当前已实现的第二方案

第二方案已经实现为：

```text
BidirectionalInteractionBlock
```

模型开关：

```text
--model pcnn_bi_interaction
```

理论对应关系：

```text
全局到局部：
使用全局人脸特征 x1 生成空间注意图，引导局部拼接特征 x10 关注更关键的表情区域。

局部到全局：
使用引导后的局部特征生成通道权重，反馈增强全局特征中与局部表情细节一致的语义通道。
```

插入位置：

```text
x1 和 x10 均为 256 通道、同空间分辨率的中层特征。
模块插入在 PCNN 原始 STN 对齐融合之前。
```

合理性：

```text
1. 全局特征负责判断整张脸的语义结构，适合做空间引导。
2. 局部特征包含眼、鼻、嘴等细粒度变化，适合做通道反馈。
3. 模块使用残差式调制，不直接替换原始特征。
4. 两个残差缩放参数初始为 0，因此初始状态等价于原始 PCNN。
5. 原始 --model pcnn 路径不受影响，可以做严格消融对比。
```

已完成冒烟测试：

```text
input:  (2, 3, 224, 224)
out:    (2, 7)
heads:  (2, 7)
local_scale initial:  0.0
global_scale initial: 0.0
```

训练启动状态：

```text
已尝试启动 RAF-DB 完整训练，但当前 /media/ag/SSD 未挂载。
dataset/rafdb/train 软链接存在，但指向的真实数据目录暂时不可访问。
需要先在文件管理器中挂载标签为 SSD 的磁盘，或手动挂载 /dev/nvme2n1p2。
```

挂载后可直接运行：

```bash
cd /home/ag/LB/PCNN
conda run -n LB python -u train.py \
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

## 8. 当前 Git 状态

已推送到 GitHub 的提交：

```text
5e4be57 Initial PCNN reproduction setup
88dc95d 支持官方FERPlus与遮挡测试集
120a354 完善FERPlus训练协议与遮挡测试记录
```

当前可能还有未提交改动：

```text
PCNN_IMPROVEMENT_PLAN.md
network/models.py
train.py
val.py
RAFDB_EXPERIMENT_LOG.md
CURRENT_EXPERIMENT_STATUS.md
```

如果要保存当前状态到 GitHub，建议下一次提交信息：

```text
实现全局局部双向交互模块并记录实验状态
```

## 9. 后续优化路线

当前不建议继续把重点放在局部增强模块上。

当前特征级双向交互模块在 RAF-DB 上暂未超过原始 baseline：

```text
pcnn_bi_interaction 从头训练最好: 88.005%
pcnn_bi_interaction 微调最好:     88.266%
原始 PCNN baseline:                88.331%
```

当前已确认有效的是输出级局部头融合：

```text
out + 0.4 * heads = 88.820%
```

当前新的论文主线实现是：

```text
--model pcnn_gapw
```

核心创新：

```text
固定语义局部区域 -> 全局上下文引导的 4x4 patch 自适应加权
```

当前代码检查：

```text
Python 编译通过
前向测试通过
加载原始 PCNN baseline 权重时仅缺少 gapw.* 新参数，符合预期
由于 /media/ag/SSD 当前未挂载，RAF-DB 正式验证暂时无法运行
```

推荐顺序：

```text
1. 保留原始 PCNN baseline = 88.331%
2. 保留输出级局部头融合结果 = 88.820%
3. 暂停当前 pcnn_bi_interaction 方向
4. 优先训练和验证 pcnn_gapw，作为论文主创新模块
5. GAPW 需要和原始 PCNN、固定 5 区域、固定融合进行消融比较
```

论文中最重要的实验逻辑：

```text
同数据集
同训练命令
同测试集
同 baseline
只改变网络模块
```

这样才能说明提升来自我们的优化，而不是数据集、训练轮数或测试口径变化。
