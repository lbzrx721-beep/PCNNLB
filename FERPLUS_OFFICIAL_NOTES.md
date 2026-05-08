# FERPlus 官方协议适配说明

创建日期：2026-04-28

本文档记录本次为了复现 PCNN 在 FERPlus / Occlusion-FERPlus 上的实验，对代码和数据入口做出的调整。

## 1. 为什么要改

之前使用的 FERPlus 数据是别人整理好的 `train/validation/test + 类别文件夹` 版本，虽然可以直接被 `torchvision.datasets.ImageFolder` 读取，但它不一定等价于 Microsoft 官方 FERPlus 协议。

官方 FERPlus 的来源是：

```text
FER2013 官方 fer2013.csv
+ Microsoft FERPlus 官方 fer2013new.csv
```

其中 FER2013 提供原始 48x48 灰度图像像素，FERPlus 提供 10 人投票标签。为了让 PCNN 更接近论文评估口径，我们先使用 Microsoft 官方脚本生成官方图片目录，再把官方投票标签转换成 PCNN 当前交叉熵训练可用的 majority 单标签 ImageFolder 目录。

## 2. 新增的数据转换脚本

新增脚本：

```text
scripts/prepare_ferplus_imagefolder.py
```

功能：

```text
读取 FERPlus_official/FER2013Train/label.csv
读取 FERPlus_official/FER2013Valid/label.csv
读取 FERPlus_official/FER2013Test/label.csv
```

然后按 FERPlus majority 规则生成：

```text
FERPlus_ImageFolder_majority/train
FERPlus_ImageFolder_majority/validation
FERPlus_ImageFolder_majority/test
```

转换规则：

```text
只保留 8 个表情类：
neutral, happiness, surprise, sadness, anger, disgust, fear, contempt

跳过 unknown / NF 样本
跳过没有超过半数投票的歧义样本
```

这样做的原因是 PCNN 当前训练代码使用 `CrossEntropyLoss`，需要每张图对应一个硬标签。

## 3. train.py 的改动

`train.py` 增加了：

```text
--train-split
--val-split
```

以前训练脚本固定读取：

```text
dataset/<dataset>/train
dataset/<dataset>/test
```

现在可以显式指定：

```bash
--train-split train --val-split validation
```

这样 FERPlus 训练时可以用 validation 选 best checkpoint，最终再单独用 test 报告结果。

另外，训练和验证 transform 增加：

```python
image.convert("RGB")
```

原因是官方 FERPlus 从 FER2013 生成的是 48x48 灰度 PNG，而 PCNN 的 ResNet backbone 和 ImageNet 均值方差归一化按 3 通道图像设计。显式转 RGB 可以避免灰度图通道数不匹配或底层张量异常。

## 4. train_ferplus.py 的改动

`train_ferplus.py` 默认数据目录改为：

```text
/media/ag/SSD/LB/MyDatasets/FERPlus_ImageFolder_majority
```

默认验证集改为：

```text
validation
```

同时支持两套 FERPlus 类别命名：

```text
旧整理版：angry, happy, sad, suprise
官方 majority 版：anger, happiness, sadness, surprise
```

两者都会映射到 PCNN 使用的 8 类输出顺序。

## 5. val.py 的改动

`val.py` 增加：

```text
--split
```

现在可以评估不同划分：

```bash
--split validation
--split test
```

同时，`val.py` 也增加了灰度图转 RGB，以及 FERPlus / Occlusion-FERPlus 的标签顺序适配。

## 6. Occlusion-FERPlus 支持

新增数据集入口：

```text
dataset/occlusion-ferplus/train
dataset/occlusion-ferplus/validation
dataset/occlusion-ferplus/test
```

其中：

```text
train      -> 官方 FERPlus majority train
validation -> 官方 FERPlus majority validation
test       -> Occlusion-FERPlus/images_by_label
```

这样可以先在正常 FERPlus 上训练，再在遮挡 FERPlus 上测试鲁棒性。

## 7. 推荐训练命令

官方 FERPlus majority 训练：

```bash
cd /home/ag/LB/PCNN
conda activate LB

python -u train_ferplus.py \
  --device cuda:0 \
  --batch-size 64 \
  --workers 0 \
  --epochs 100 \
  --no-pretrained-pcnn
```

如果运行稳定，可以提高数据加载速度：

```bash
python -u train_ferplus.py \
  --device cuda:0 \
  --batch-size 64 \
  --workers 4 \
  --epochs 100 \
  --no-pretrained-pcnn
```

最终测试：

```bash
python -u val.py \
  --dataset ferplus \
  --num-class 8 \
  --device cuda:0 \
  --batch-size 128 \
  --workers 4 \
  --split test \
  --model-path checkpoints/<your_best_checkpoint>.pth
```

Occlusion-FERPlus 测试：

```bash
python -u val.py \
  --dataset occlusion-ferplus \
  --num-class 8 \
  --device cuda:0 \
  --batch-size 128 \
  --workers 4 \
  --split test \
  --model-path checkpoints/<your_best_checkpoint>.pth
```

## 8. 关于旧版本

本次修改会通过 Git commit 保存。推送到 GitHub 后，旧版本不会消失，可以通过 GitHub 的提交历史、`git log`、`git checkout <commit>` 或 `git diff <old> <new>` 找回和比较。
