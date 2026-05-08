# 作者公开权重测试结果记录

创建日期：2026-04-28

本文档记录 PCNN 作者公开权重在本地标准测试集与遮挡测试集上的验证结果。

测试日期：2026-04-28

## 1. 测试结论

| 作者公开权重 | 正常测试集 | 正常准确率 | 遮挡测试集 | 遮挡准确率 |
| --- | --- | ---: | --- | ---: |
| `experiment/rafdb/rafdb.pth` | RAF-DB Test | 89.244% | Occlusion-RAF-DB | 85.695% |
| `experiment/ferplus/ferplus.pth` | FERPlus Test | 84.429% | Occlusion-FERPlus | 82.149% |

其中 RAF-DB 正常测试结果与论文结果基本一致，可以认为 RAF-DB 部分复现成功。

FERPlus 与 Occlusion-FERPlus 的结果低于论文表格结果，主要可能来自 FERPlus 数据整理协议、标签筛选方式、遮挡集版本或作者公开 checkpoint 与论文实验 checkpoint 不完全一致。

## 2. 测试集信息

| 测试集 | 本地路径 | 样本数 | 类别数 | 说明 |
| --- | --- | ---: | ---: | --- |
| RAF-DB Test | `dataset/rafdb/test` | 3068 | 7 | 标准 RAF-DB Basic test |
| Occlusion-RAF-DB | `dataset/occlusion-rafdb/test` | 734 | 7 | 遮挡 RAF-DB 测试集，CSV 原始标注 735 条，其中 1 条重复，实际唯一图片 734 张 |
| FERPlus Test | `dataset/ferplus/test` | 3545 | 8 | 当前使用 argmax/voting 方式整理的 FERPlus test |
| Occlusion-FERPlus | `dataset/occlusion-ferplus/test` | 605 | 8 | 遮挡 FERPlus 测试集 |

## 3. 数据集软链接

RAF-DB：

```text
dataset/rafdb/train -> /media/ag/SSD/LB/MyDatasets/DATA/RAF-DB/archive/DATASET/train
dataset/rafdb/test  -> /media/ag/SSD/LB/MyDatasets/DATA/RAF-DB/archive/DATASET/test
```

Occlusion-RAF-DB：

```text
dataset/occlusion-rafdb/train      -> /media/ag/SSD/LB/MyDatasets/DATA/RAF-DB/archive/DATASET/train
dataset/occlusion-rafdb/validation -> /media/ag/SSD/LB/MyDatasets/DATA/RAF-DB/archive/DATASET/test
dataset/occlusion-rafdb/test       -> /media/ag/SSD/LB/MyDatasets/DATA/Occlusion-RAF-DB/images_by_label
```

FERPlus：

```text
dataset/ferplus/train      -> /media/ag/SSD/LB/MyDatasets/FERPlus_ImageFolder_argmax/train
dataset/ferplus/validation -> /media/ag/SSD/LB/MyDatasets/FERPlus_ImageFolder_argmax/validation
dataset/ferplus/test       -> /media/ag/SSD/LB/MyDatasets/FERPlus_ImageFolder_argmax/test
```

Occlusion-FERPlus：

```text
dataset/occlusion-ferplus/test -> /media/ag/SSD/LB/MyDatasets/DATA/Occlusion-FERPlus/images_by_label
```

## 4. 运行命令

### 4.1 RAF-DB Test

```bash
cd /home/ag/LB/PCNN
conda run -n LB python -u val.py \
  --dataset rafdb \
  --num-class 7 \
  --device cuda:0 \
  --batch-size 128 \
  --workers 0 \
  --split test \
  --model-path experiment/rafdb/rafdb.pth
```

结果：

```text
Accuracy 89.244
```

### 4.2 Occlusion-RAF-DB

```bash
cd /home/ag/LB/PCNN
conda run -n LB python -u val.py \
  --dataset occlusion-rafdb \
  --num-class 7 \
  --device cuda:0 \
  --batch-size 128 \
  --workers 0 \
  --split test \
  --model-path experiment/rafdb/rafdb.pth
```

结果：

```text
Accuracy 85.695
```

### 4.3 FERPlus Test

```bash
cd /home/ag/LB/PCNN
conda run -n LB python -u val.py \
  --dataset ferplus \
  --num-class 8 \
  --device cuda:0 \
  --batch-size 128 \
  --workers 0 \
  --split test \
  --model-path experiment/ferplus/ferplus.pth
```

结果：

```text
Accuracy 84.429
```

### 4.4 Occlusion-FERPlus

```bash
cd /home/ag/LB/PCNN
conda run -n LB python -u val.py \
  --dataset occlusion-ferplus \
  --num-class 8 \
  --device cuda:0 \
  --batch-size 128 \
  --workers 0 \
  --split test \
  --model-path experiment/ferplus/ferplus.pth
```

结果：

```text
Accuracy 82.149
```

补充检查：如果 Occlusion-FERPlus 推理时使用 `out + heads` 融合预测，准确率为 82.975%。当前主表采用 `val.py` 默认的主分支 `out` 预测结果。

## 5. 结果解释

RAF-DB 部分：

```text
作者公开权重 + 本地标准 RAF-DB test = 89.244%
```

该结果与论文报告基本一致，因此 RAF-DB 数据路径、标签映射和测试流程可以认为是正确的。

Occlusion-RAF-DB 部分：

```text
作者公开权重 + 本地 Occlusion-RAF-DB = 85.695%
```

该结果可作为 RAF-DB 遮挡鲁棒性 baseline。

FERPlus 部分：

```text
作者公开权重 + 本地 FERPlus argmax/voting test = 84.429%
```

该结果低于作者 checkpoint 记录的 `best_acc 85.5863`，也低于论文表格中可疑的 FERPlus 数值。因此 FERPlus 部分需要在论文中谨慎表述为：

```text
使用作者公开发布的 FERPlus 预训练权重，在本地整理的 FERPlus 测试协议下得到 84.429%。
```

Occlusion-FERPlus 部分：

```text
作者公开权重 + 本地 Occlusion-FERPlus = 82.149%
```

该结果低于论文遮挡表格中的 84.63%，可能原因包括遮挡数据集版本、图像生成方式、FERPlus 标签协议或公开 checkpoint 与论文实验 checkpoint 不完全一致。

## 6. 建议写法

在论文或实验记录中可以写：

```text
为了验证本地实验环境与数据预处理流程的正确性，本文首先采用 PCNN 作者公开发布的预训练权重进行复现测试。实验结果显示，在标准 RAF-DB 测试集上，作者公开权重取得 89.244% 的准确率，与论文报告结果基本一致，说明 RAF-DB 数据路径、标签映射和测试流程正确。在遮挡测试集上，作者 RAF-DB 权重在 Occlusion-RAF-DB 上取得 85.695% 的准确率。

对于 FERPlus，作者公开权重在本文本地整理的 FERPlus argmax/voting 测试集上取得 84.429% 的准确率，在 Occlusion-FERPlus 上取得 82.149% 的准确率。该结果与论文表格中 FERPlus 及遮挡 FERPlus 结果存在一定差异，推测与 FERPlus 标签整理协议、遮挡数据版本或公开 checkpoint 与论文实验 checkpoint 不完全一致有关。因此，后续实验中 FERPlus 结果均基于本文统一整理的数据协议进行公平比较。
```
