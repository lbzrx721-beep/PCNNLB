import argparse
import datetime
import os
import random
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

from network.models import PCNN


plt.switch_backend("agg")


def load_checkpoint_compat(path, map_location):
    # PyTorch >=2.6 defaults to weights_only=True.
    # Try safe path first, then fall back for legacy checkpoints.
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        pass

    # Legacy checkpoint may reference "__main__.RecorderMeter".
    setattr(sys.modules["__main__"], "RecorderMeter", RecorderMeter)
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is not None:
        try:
            with safe_globals([RecorderMeter]):
                return torch.load(path, map_location=map_location, weights_only=False)
        except Exception:
            pass

    return torch.load(path, map_location=map_location, weights_only=False)


def parse_args(default_dataset="rafdb"):
    parser = argparse.ArgumentParser(description="Train PCNN for facial expression recognition.")
    parser.add_argument("--dataset", default=default_dataset)
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Direct dataset directory containing train/validation/test. "
            "When set, it takes precedence over --data-root/--dataset."
        ),
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--test-after-training", action="store_true")
    parser.add_argument(
        "--model",
        default="pcnn",
        choices=[
            "pcnn",
            "pcnn_local_enhanced",
            "pcnn_bi_interaction",
            "pcnn_gated_fusion",
            "pcnn_gapw",
            "pcnn_region_relation",
        ],
    )
    parser.add_argument(
        "--num-class",
        type=int,
        default=None,
        help="Defaults to 8 for FERPlus datasets and 7 otherwise.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used for model initialization, data shuffling, and augmentation.",
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-step", type=int, default=15)
    parser.add_argument("--base-lr-mult", type=float, default=1.0)
    parser.add_argument("--interaction-lr-mult", type=float, default=1.0)
    parser.add_argument("--interaction-scale-init", type=float, default=None)
    parser.add_argument("--gapw-scale-init", type=float, default=None)
    parser.add_argument("--gapw-patch-grid", type=int, default=4)
    parser.add_argument("--gapw-temperature", type=float, default=1.0)
    parser.add_argument("--gapw-overlap-ratio", type=float, default=0.0)
    parser.add_argument("--gapw-adaptive-crop", action="store_true")
    parser.add_argument("--gapw-max-offset", type=float, default=0.25)
    parser.add_argument("--gapw-attention-pool", action="store_true")
    parser.add_argument("--region-relation-heads", type=int, default=4)
    parser.add_argument("--region-dropout", type=float, default=0.2)
    parser.add_argument("--region-temperature", type=float, default=1.0)
    parser.add_argument(
        "--region-output-scale",
        type=float,
        default=0.8,
        help="Maximum residual scale of the region-relation classifier.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze ResNet feature extractors and STN localization network.",
    )
    parser.add_argument(
        "--train-gapw-only",
        action="store_true",
        help="Freeze all existing PCNN parameters and train only GAPW.",
    )
    parser.add_argument(
        "--train-region-only",
        action="store_true",
        help="Freeze the original PCNN and train only region-relation fusion.",
    )
    parser.add_argument(
        "--head-fusion-weight",
        type=float,
        default=0.0,
        help="Use out + weight * heads as the main prediction.",
    )
    parser.add_argument("--erasing-p", type=float, default=0.5)
    parser.add_argument("--erasing-scale-min", type=float, default=0.02)
    parser.add_argument("--erasing-scale-max", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=12)
    parser.add_argument("--beta", type=float, default=8)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--stop-on-nonfinite", action="store_true", default=True)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--backbone-path", default="models/resnet18_msceleb.pth")
    parser.add_argument(
        "--pretrained-pcnn",
        default=None,
        help="Defaults to the matching RAF-DB or FERPlus author checkpoint.",
    )
    parser.add_argument("--no-pretrained-pcnn", action="store_true")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume model, optimizer, epoch, and LR schedule from a training checkpoint.",
    )
    parser.add_argument("--allow-random-backbone", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--rafdb-numeric-remap",
        action="store_true",
        default=True,
        help="Remap numeric RAF-DB folders (1..7) to model label order.",
    )
    args = parser.parse_args()
    return apply_dataset_defaults(args)


def apply_dataset_defaults(args):
    is_ferplus = "ferplus" in args.dataset.lower()
    if args.num_class is None:
        args.num_class = 8 if is_ferplus else 7
    if args.pretrained_pcnn is None:
        checkpoint_dataset = "ferplus" if is_ferplus else "rafdb"
        args.pretrained_pcnn = f"experiment/{checkpoint_dataset}/{checkpoint_dataset}.pth"
    return args


def main(default_dataset="rafdb"):
    args = parse_args(default_dataset=default_dataset)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    now = datetime.datetime.now()
    time_str = now.strftime("[%m-%d]-[%H-%M]-")
    model_name = f"{args.dataset}_"
    checkpoint_path = f"checkpoints/{model_name}{time_str}.pth"
    best_checkpoint_path = f"checkpoints/{model_name}{time_str}_best.pth"
    log_path = f"logs/{model_name}{time_str}.txt"
    curve_path = f"logs/{model_name}{time_str}.png"

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    dataset_dir = args.data_dir or os.path.join(args.data_root, args.dataset)
    train_dir = os.path.join(dataset_dir, args.train_split)
    val_dir = os.path.join(dataset_dir, args.val_split)
    test_dir = os.path.join(dataset_dir, args.test_split)

    model = PCNN(
        num_class=args.num_class,
        device=device,
        backbone_path=args.backbone_path,
        require_backbone=not args.allow_random_backbone,
        use_local_enhancement=args.model == "pcnn_local_enhanced",
        use_bidirectional_interaction=args.model == "pcnn_bi_interaction",
        use_gated_fusion=args.model == "pcnn_gated_fusion",
        use_gapw=args.model == "pcnn_gapw",
        use_region_relation=args.model == "pcnn_region_relation",
        gapw_patch_grid=args.gapw_patch_grid,
        gapw_temperature=args.gapw_temperature,
        gapw_overlap_ratio=args.gapw_overlap_ratio,
        gapw_adaptive_crop=args.gapw_adaptive_crop,
        gapw_max_offset=args.gapw_max_offset,
        gapw_attention_pool=args.gapw_attention_pool,
        region_relation_heads=args.region_relation_heads,
        region_dropout=args.region_dropout,
        region_temperature=args.region_temperature,
        region_output_scale=args.region_output_scale,
    )

    resume_checkpoint = None
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        resume_checkpoint = load_checkpoint_compat(args.resume, map_location=device)
        state_dict = (
            resume_checkpoint["state_dict"]
            if "state_dict" in resume_checkpoint
            else resume_checkpoint
        )
        # Exact resume requires the same architecture as the saved run.
        model.load_state_dict(state_dict, strict=True)
    elif not args.no_pretrained_pcnn and os.path.exists(args.pretrained_pcnn):
        checkpoint = load_checkpoint_compat(args.pretrained_pcnn, map_location=device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)

    configure_interaction_module(model, args)
    configure_gapw_module(model, args)
    freeze_backbone_if_needed(model, args)
    model = model.to(device)
    criterion_cls = nn.CrossEntropyLoss().to(device)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)
    recorder = RecorderMeter(args.epochs)
    start_epoch = 0
    best_acc = 0.0
    best_model_path = best_checkpoint_path
    if resume_checkpoint is not None:
        if "optimizer" not in resume_checkpoint:
            raise KeyError("Resume checkpoint does not contain optimizer state.")
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        start_epoch = int(resume_checkpoint.get("epoch", 0))
        best_acc = float(resume_checkpoint.get("best_acc", 0.0))
        if start_epoch >= args.epochs:
            raise ValueError(
                f"Checkpoint is already at epoch {start_epoch}, but --epochs is {args.epochs}. "
                "Set --epochs to the desired total epoch count."
            )

        if "scheduler" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
        else:
            # Legacy checkpoints saved the optimizer after scheduler.step().
            # Its current LR is already correct; restoring last_epoch preserves
            # the original StepLR boundaries (75 and 90 for this RAF-DB run).
            scheduler.last_epoch = start_epoch
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]

        restore_recorder(recorder, resume_checkpoint.get("recorder"))
        previous_best_path = args.resume[:-4] + "_best.pth" if args.resume.endswith(".pth") else ""
        best_model_path = previous_best_path if os.path.isfile(previous_best_path) else args.resume
    # Keep deterministic cuDNN behavior whenever a seed is configured.
    # benchmark=True may select different convolution algorithms between runs.
    cudnn.deterministic = args.seed is not None
    cudnn.benchmark = torch.cuda.is_available() and args.seed is None

    train_loader = make_loader(train_dir, args.batch_size, args.workers, train=True, args=args)
    val_loader = make_loader(val_dir, args.batch_size, args.workers, train=False, args=args)

    write_log(log_path, f"Training time: {now:%m-%d %H:%M}")
    write_log(log_path, f"device: {device}")
    write_log(log_path, f"dataset: {args.dataset}")
    write_log(log_path, f"data_dir: {dataset_dir}")
    write_log(log_path, f"model: {args.model}")
    write_log(log_path, f"seed: {args.seed}")
    write_log(log_path, f"train_split: {args.train_split}")
    write_log(log_path, f"val_split: {args.val_split}")
    write_log(log_path, f"test_split: {args.test_split}")
    write_log(log_path, f"grad_clip: {args.grad_clip}")
    write_log(log_path, f"base_lr_mult: {args.base_lr_mult}")
    write_log(log_path, f"interaction_lr_mult: {args.interaction_lr_mult}")
    write_log(log_path, f"interaction_scale_init: {args.interaction_scale_init}")
    write_log(log_path, f"gapw_scale_init: {args.gapw_scale_init}")
    write_log(log_path, f"gapw_patch_grid: {args.gapw_patch_grid}")
    write_log(log_path, f"gapw_temperature: {args.gapw_temperature}")
    write_log(log_path, f"gapw_overlap_ratio: {args.gapw_overlap_ratio}")
    write_log(log_path, f"gapw_adaptive_crop: {args.gapw_adaptive_crop}")
    write_log(log_path, f"gapw_max_offset: {args.gapw_max_offset}")
    write_log(log_path, f"gapw_attention_pool: {args.gapw_attention_pool}")
    write_log(log_path, f"region_relation_heads: {args.region_relation_heads}")
    write_log(log_path, f"region_output_scale: {args.region_output_scale}")
    write_log(log_path, f"region_dropout: {args.region_dropout}")
    write_log(log_path, f"region_temperature: {args.region_temperature}")
    write_log(log_path, f"freeze_backbone: {args.freeze_backbone}")
    write_log(log_path, f"train_gapw_only: {args.train_gapw_only}")
    write_log(log_path, f"train_region_only: {args.train_region_only}")
    write_log(log_path, f"head_fusion_weight: {args.head_fusion_weight}")
    write_log(log_path, f"erasing_p: {args.erasing_p}")
    write_log(log_path, f"erasing_scale: ({args.erasing_scale_min}, {args.erasing_scale_max})")
    write_log(log_path, f"resume: {args.resume}")
    write_log(log_path, f"start_epoch: {start_epoch}")
    write_log(log_path, f"restored_best_acc: {best_acc:.3f}")
    if args.val_split == args.test_split:
        write_log(log_path, "WARNING: val_split and test_split are identical. This is not a strict protocol.")

    for epoch in tqdm(range(start_epoch, args.epochs), initial=start_epoch, total=args.epochs):
        start_time = time.time()
        write_log(log_path, f"Current learning rate: {format_lrs(optimizer)}")

        train_acc, train_loss = train_one_epoch(
            train_loader,
            model,
            criterion_cls,
            optimizer,
            device,
            args,
            log_path,
            epoch + 1,
        )

        if args.skip_validation:
            val_acc, val_loss = train_acc, train_loss
        else:
            val_acc, val_loss = validate(
                val_loader,
                model,
                criterion_cls,
                device,
                args,
                log_path,
                phase="Validation",
            )

        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            write_log(log_path, "Stopping early because a non-finite loss was detected.")
            break

        scheduler.step()
        recorder.update(epoch, train_loss, train_acc, val_loss, val_acc)
        recorder.plot_curve(curve_path)

        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "recorder": recorder,
            },
            is_best,
            checkpoint_path,
            best_checkpoint_path,
        )
        if is_best:
            best_model_path = best_checkpoint_path

        write_log(log_path, f"Current best accuracy: {best_acc:.3f}")
        write_log(log_path, f"An epoch time: {time.time() - start_time:.2f}")

    print(f"best_checkpoint_path: {best_model_path}")

    if args.test_after_training:
        write_log(log_path, "Loading best checkpoint for final test...")
        checkpoint = load_checkpoint_compat(best_model_path, map_location=device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        test_loader = make_loader(test_dir, args.batch_size, args.workers, train=False, args=args)
        test_acc, test_loss = validate(
            test_loader,
            model,
            criterion_cls,
            device,
            args,
            log_path,
            phase="Final Test",
        )
        write_log(log_path, f"Final test accuracy: {test_acc:.3f}")
        write_log(log_path, f"Final test loss: {test_loss:.4f}")


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def configure_interaction_module(model, args):
    if args.interaction_scale_init is None:
        return
    module = getattr(model, "bidirectional_interaction", None)
    if module is None:
        return
    with torch.no_grad():
        module.local_scale.fill_(args.interaction_scale_init)
        module.global_scale.fill_(args.interaction_scale_init)


def configure_gapw_module(model, args):
    if args.gapw_scale_init is None:
        return
    module = getattr(model, "gapw", None)
    if module is None:
        return
    with torch.no_grad():
        module.enhance_scale.fill_(args.gapw_scale_init)
        module.patch_scale.fill_(args.gapw_scale_init)
        module.global_scale.fill_(args.gapw_scale_init)


def freeze_backbone_if_needed(model, args):
    if args.train_gapw_only:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("gapw.")
        return

    if args.train_region_only:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("region_relation.")
        return

    if not args.freeze_backbone:
        return

    trainable_prefixes = (
        "bidirectional_interaction.",
        "gated_fusion.",
        "gapw.",
        "fc.",
        "fc2.",
        "fc3.",
        "fc4.",
        "fc6.",
        "fc7.",
    )
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(trainable_prefixes)


def build_optimizer(model, args):
    interaction_params = []
    base_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(
            ("bidirectional_interaction.", "gated_fusion.", "gapw.", "region_relation.")
        ):
            interaction_params.append(param)
        else:
            base_params.append(param)

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": args.lr * args.base_lr_mult, "name": "base"})
    if interaction_params:
        param_groups.append(
            {
                "params": interaction_params,
                "lr": args.lr * args.interaction_lr_mult,
                "name": "interaction",
            }
        )

    return torch.optim.SGD(
        param_groups,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def format_lrs(optimizer):
    return ", ".join(
        f"{group.get('name', idx)}={group['lr']:.6g}"
        for idx, group in enumerate(optimizer.param_groups)
    )


def restore_recorder(recorder, saved_recorder):
    """Copy compatible history into a recorder sized for the requested run."""
    if saved_recorder is None:
        return

    saved_losses = getattr(saved_recorder, "epoch_losses", None)
    saved_accuracy = getattr(saved_recorder, "epoch_accuracy", None)
    if saved_losses is not None:
        rows = min(recorder.epoch_losses.shape[0], saved_losses.shape[0])
        recorder.epoch_losses[:rows] = saved_losses[:rows]
    if saved_accuracy is not None:
        rows = min(recorder.epoch_accuracy.shape[0], saved_accuracy.shape[0])
        recorder.epoch_accuracy[:rows] = saved_accuracy[:rows]
    recorder.current_epoch = min(
        int(getattr(saved_recorder, "current_epoch", 0)), recorder.total_epoch
    )


def make_loader(path, batch_size, workers, train, args=None):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset folder not found: {path}")

    if train:
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.RandomRotation(20), transforms.RandomCrop(224, padding=32)],
                    p=0.2,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(
                    p=args.erasing_p if args is not None else 0.5,
                    scale=(
                        args.erasing_scale_min if args is not None else 0.02,
                        args.erasing_scale_max if args is not None else 0.25,
                    ),
                ),
            ]
        )
        shuffle = True
    else:
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        shuffle = False

    dataset = SafeImageFolder(path, transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        target = self.samples[index][1]
        failures = []
        for candidate_index in self._candidate_indices(index, target):
            path = self.samples[candidate_index][0]
            try:
                if failures:
                    tqdm.write(f"WARNING: using replacement image {path}.")
                return super().__getitem__(candidate_index)
            except (FileNotFoundError, OSError) as exc:
                failures.append(f"{path}: {exc}")
                continue

        raise RuntimeError(
            "Failed to load any image from the same class. "
            f"First failures: {' | '.join(failures[:5])}"
        )

    def _candidate_indices(self, index, target):
        yield index
        total = len(self.samples)
        for offset in range(1, total):
            next_index = (index + offset) % total
            if self.samples[next_index][1] == target:
                yield next_index


def train_one_epoch(train_loader, model, criterion_cls, optimizer, device, args, log_path, epoch):
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Accuracy", ":6.3f")
    progress = ProgressMeter(len(train_loader), [losses, top1], prefix=f"Epoch: [{epoch}]")
    model.train()
    if args.train_gapw_only or args.train_region_only:
        # Frozen PCNN BatchNorm buffers must remain unchanged; otherwise this
        # is not a module-only ablation even though parameters require no grad.
        model.eval()
        if args.train_gapw_only:
            model.gapw.train()
        else:
            model.region_relation.train()

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        targets = remap_targets_if_needed(
            targets, args.dataset, train_loader.dataset.classes, args.rafdb_numeric_remap
        )

        optimizer.zero_grad()
        out, heads = model(images)
        logits = fuse_logits(out, heads, args.head_fusion_weight)
        loss = criterion_cls(logits, targets) * args.alpha + criterion_cls(heads, targets) * args.beta
        if args.stop_on_nonfinite and not torch.isfinite(loss):
            write_log(log_path, f"Non-finite training loss at epoch {epoch}, batch {i}. Stop this run.")
            return 0.0, float("nan")
        acc = accuracy(logits, targets)

        losses.update(loss.item(), images.size(0))
        top1.update(acc.item(), images.size(0))

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if i % args.print_freq == 0:
            progress.display(i, log_path)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion_cls, device, args, log_path, phase="Validation"):
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Accuracy", ":6.3f")
    progress = ProgressMeter(len(val_loader), [losses, top1], prefix=f"{phase}: ")
    model.eval()

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            targets = remap_targets_if_needed(
                targets, args.dataset, val_loader.dataset.classes, args.rafdb_numeric_remap
            )
            out, heads = model(images)
            logits = fuse_logits(out, heads, args.head_fusion_weight)
            loss = criterion_cls(logits, targets) * args.alpha + criterion_cls(heads, targets) * args.beta
            if args.stop_on_nonfinite and not torch.isfinite(loss):
                write_log(log_path, f"Non-finite {phase.lower()} loss at batch {i}.")
                return 0.0, float("nan")
            acc = accuracy(logits, targets)

            losses.update(loss.item(), images.size(0))
            top1.update(acc.item(), images.size(0))

            if i % args.print_freq == 0:
                progress.display(i, log_path)

    write_log(log_path, f"{phase} accuracy: {top1.avg:.3f}")
    return top1.avg, losses.avg


def accuracy(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean() * 100.0


def fuse_logits(out, heads, weight):
    if weight == 0:
        return out
    return out + weight * heads


def remap_targets_if_needed(targets, dataset_name, classes, enable_remap):
    if not enable_remap:
        return targets
    dataset_name = dataset_name.lower()

    if "rafdb" in dataset_name and list(classes) == ["1", "2", "3", "4", "5", "6", "7"]:
        # RAF-DB basic labels:
        # 1:Surprise 2:Fear 3:Disgust 4:Happiness 5:Sadness 6:Anger 7:Neutral
        # Model order used by this checkpoint:
        # 0:Neutral 1:Happiness 2:Sadness 3:Surprise 4:Fear 5:Disgust 6:Anger
        lut = torch.tensor([3, 4, 5, 1, 2, 6, 0], device=targets.device, dtype=torch.long)
        return lut[targets.long()]

    if "ferplus" in dataset_name and list(classes) == [
        "angry",
        "contempt",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "suprise",
    ]:
        # This local FER-PLUS folder is alphabetical. The author's checkpoint
        # uses a different output order, inferred from the checkpoint outputs.
        lut = torch.tensor([4, 7, 5, 6, 1, 0, 3, 2], device=targets.device, dtype=torch.long)
        return lut[targets.long()]

    if "ferplus" in dataset_name and list(classes) == [
        "anger",
        "contempt",
        "disgust",
        "fear",
        "happiness",
        "neutral",
        "sadness",
        "surprise",
    ]:
        lut = torch.tensor([4, 7, 5, 6, 1, 0, 3, 2], device=targets.device, dtype=torch.long)
        return lut[targets.long()]

    return targets


def save_checkpoint(state, is_best, checkpoint_path, best_checkpoint_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


def write_log(path, text):
    tqdm.write(text)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text + "\n")


class AverageMeter:
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, log_path):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        write_log(log_path, "\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class RecorderMeter:
    def __init__(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        fig = plt.figure(figsize=(18, 8))
        x_axis = np.arange(self.total_epoch)
        plt.xlim(0, self.total_epoch)
        y_max = max(float(np.max(self.epoch_losses[:, 0])), 1.0)
        plt.ylim(0, y_max)
        plt.grid()
        plt.title("Training Loss Curve", fontsize=20)
        plt.xlabel("Training Epoch", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.plot(x_axis, self.epoch_losses[:, 0], color="r", linestyle="-", label="Train Loss", lw=2)
        plt.legend(loc=4, fontsize=10)
        fig.savefig(save_path, dpi=80, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
