# PCNN 复现说明

创建日期：2026-04-28

本文档用于说明当前项目如何复现原始 PCNN 代码与实验流程。

原始项目地址：

```text
https://github.com/hellloxiaotian/PCNN
```

## 一、目录结构

项目主要目录如下：

```text
PCNN/
├── dataset/rafdb/train
├── dataset/rafdb/test
├── models/resnet18_msceleb.pth
├── experiment/rafdb/rafdb.pth
├── checkpoints
├── logs
└── network
```

其中，`dataset` 用于存放数据集软链接或实际数据，`models` 用于存放 ResNet-18 预训练权重，`experiment` 用于存放作者公开的 PCNN 权重，`checkpoints` 和 `logs` 分别用于保存本地训练权重和日志。

## 二、需要准备的外部文件

原作者在 Google Drive 中提供了 ResNet-18 预训练权重、PCNN 权重以及一个小规模 RAF-DB 示例数据集：

```text
https://drive.google.com/drive/folders/1st0sETk5Jw0Qs6o4qcAKPn5EWAJJR_vc?usp=sharing
```

文件建议放置位置如下：

```text
models/resnet18_msceleb.pth
experiment/rafdb/rafdb.pth
dataset/rafdb/train/<class_name>/*.jpg
dataset/rafdb/test/<class_name>/*.jpg
```

`train.py` 和 `val.py` 使用 `torchvision.datasets.ImageFolder` 读取数据，因此每个类别都必须是一个单独的子文件夹。

## 三、环境配置

基础环境配置命令如下：

```bash
cd /home/ag/LB/PCNN
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果需要使用 GPU，请根据本机 CUDA 版本安装对应的 PyTorch 版本。

当前本机实验主要使用 conda 环境：

```bash
conda activate LB
```

## 四、训练命令

RAF-DB 上的基础训练命令：

```bash
python train.py --device cuda:0 --dataset rafdb
```

如果只想在 CPU 上快速检查代码是否能跑通，可以使用：

```bash
python train.py --device cpu --dataset rafdb --epochs 1 --batch-size 2 --workers 0 --no-pretrained-pcnn
```

## 五、测试命令

RAF-DB 上的基础测试命令：

```bash
python val.py --device cuda:0 --dataset rafdb
```

如果只想在 CPU 上快速检查测试流程，可以使用：

```bash
python val.py --device cpu --dataset rafdb --batch-size 1 --workers 0
```

## 六、当前说明

本项目已经在原始 PCNN 复现基础上加入了 GAPW 优化相关代码。原始 PCNN 复现说明仍保留在本文档中，GAPW 的具体方法、训练命令和实验结果请查看：

```text
GAPW_OPTIMIZATION_README.md
```
