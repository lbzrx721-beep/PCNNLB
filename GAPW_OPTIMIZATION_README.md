# GAPW Optimization Notes

## Overview

This repository adds a model-internal optimization for PCNN:

```text
GAPW: Global-guided Adaptive Patch Weighting
中文: 全局上下文引导的自适应 Patch 区域加权
```

The goal is to improve local expression modeling and occlusion robustness without relying on output-level fusion or test-time augmentation as the main contribution.

## Motivation

Original PCNN uses global face features together with several fixed local semantic regions. This is effective, but fixed local modeling can be sensitive to occlusion, pose variation, and individual appearance differences.

GAPW changes the local modeling strategy from fixed local-region usage to sample-adaptive patch weighting. For each input image, the model uses the global face context to estimate which local patches are more discriminative and which patches are less reliable.

## Method

In `network/models.py`, GAPW is implemented as `GlobalGuidedPatchWeighting` and enabled through:

```bash
--model pcnn_gapw
```

The module is inserted after PCNN builds the stitched local feature map `x10` and before the original STN-based global-local fusion path.

Main steps:

```text
1. Keep the PCNN global feature map and stitched local feature map.
2. Pool the local feature map into a 4x4 patch grid, producing 16 patch descriptors.
3. Build each patch context from global context, local patch descriptor, and their absolute difference.
4. Predict patch importance and patch reliability.
5. Combine importance and reliability to obtain adaptive patch weights.
6. Reweight the local feature map with a bounded residual scale.
7. Use the weighted local feature to lightly calibrate the global feature.
```

This keeps the original PCNN behavior at initialization because the GAPW residual scales are initialized to zero and bounded during training.

## Key Implementation Points

Code changes are mainly in:

```text
network/models.py
train.py
train_ferplus.py
val.py
```

Important command-line options:

```text
--model pcnn_gapw
--gapw-patch-grid 4
--gapw-temperature 1.0
--train-gapw-only
--seed
--grad-clip
```

`--train-gapw-only` freezes the existing PCNN parameters and trains only the GAPW module, which makes the comparison against the local PCNN baseline more controlled.

## RAF-DB Training Command

The 5-seed RAF-DB experiments used:

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

## Results

Local original PCNN baseline:

```text
RAF-DB:           88.331%
Occlusion-RAFDB: 84.196%
```

GAPW 5-seed results:

| Seed | RAF-DB | Occlusion-RAFDB |
| ---: | ---: | ---: |
| 2026 | 88.429 | 84.332 |
| 3407 | 88.396 | 84.332 |
| 42 | 88.364 | 84.332 |
| 1234 | 88.396 | 84.605 |
| 777 | 88.494 | 84.332 |

Mean result:

```text
RAF-DB:           88.416%  (+0.085)
Occlusion-RAFDB: 84.387%  (+0.191)
```

The improvement is modest but stable across seeds, and the occlusion-set gain is larger than the normal RAF-DB gain. This supports the interpretation that GAPW improves robustness by suppressing less reliable local regions and emphasizing more discriminative local patches.

Author public checkpoint reference:

```text
RAF-DB:           89.244%
Occlusion-RAFDB: 85.695%
```

This is kept only as a reference. The fair comparison for the current optimization is local original PCNN baseline versus local GAPW under the same environment and evaluation protocol.

## Thesis Positioning

This optimization should be described as a network-structure contribution:

```text
From fixed semantic local-region modeling
to global-context-guided adaptive patch-region weighting.
```

It should not be positioned as output-level head fusion or TTA. Those can be treated as auxiliary engineering observations, but the thesis innovation should focus on the GAPW module and its patch reliability modeling.
