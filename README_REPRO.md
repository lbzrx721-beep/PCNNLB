# PCNN Reproduction

This folder is prepared to reproduce:

https://github.com/hellloxiaotian/PCNN

## Directory Layout

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

## Required External Files

The original author provides the pretrained ResNet-18 weights, PCNN weights, and a tiny RAF-DB demo dataset on Google Drive:

https://drive.google.com/drive/folders/1st0sETk5Jw0Qs6o4qcAKPn5EWAJJR_vc?usp=sharing

Place files as follows:

```text
models/resnet18_msceleb.pth
experiment/rafdb/rafdb.pth
dataset/rafdb/train/<class_name>/*.jpg
dataset/rafdb/test/<class_name>/*.jpg
```

`train.py` and `val.py` use `torchvision.datasets.ImageFolder`, so every class must be a subdirectory.

## Environment

```bash
cd /home/ag/LB/PCNN
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if you want GPU acceleration.

## Train

```bash
python train.py --device cuda:0 --dataset rafdb
```

For CPU smoke testing:

```bash
python train.py --device cpu --dataset rafdb --epochs 1 --batch-size 2 --workers 0 --no-pretrained-pcnn
```

## Evaluate

```bash
python val.py --device cuda:0 --dataset rafdb
```

For CPU smoke testing:

```bash
python val.py --device cpu --dataset rafdb --batch-size 1 --workers 0
```
