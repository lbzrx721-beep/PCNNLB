import argparse
import datetime
import os
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
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        pass

    setattr(sys.modules["__main__"], "RecorderMeter", RecorderMeter)
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is not None:
        try:
            with safe_globals([RecorderMeter]):
                return torch.load(path, map_location=map_location, weights_only=False)
        except Exception:
            pass

    return torch.load(path, map_location=map_location, weights_only=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PCNN for FERPlus.")
    parser.add_argument("--data-dir", default="/media/ag/SSD/LB/MyDatasets/FERPlus_ImageFolder_argmax")
    parser.add_argument("--train-split", default="train", choices=["train"])
    parser.add_argument("--val-split", default="validation", choices=["validation", "test"])
    parser.add_argument("--test-split", default="test", choices=["test"])
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
        ],
    )
    parser.add_argument("--num-class", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
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
        "--head-fusion-weight",
        type=float,
        default=0.0,
        help="Use out + weight * heads as the main prediction.",
    )
    parser.add_argument("--alpha", type=float, default=12)
    parser.add_argument("--beta", type=float, default=8)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--stop-on-nonfinite", action="store_true", default=True)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--backbone-path", default="models/resnet18_msceleb.pth")
    parser.add_argument("--pretrained-pcnn", default="experiment/ferplus/ferplus.pth")
    parser.add_argument("--no-pretrained-pcnn", action="store_true")
    parser.add_argument("--allow-random-backbone", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    now = datetime.datetime.now()
    time_str = now.strftime("[%m-%d]-[%H-%M]-")
    model_name = "ferplus_"
    checkpoint_path = f"checkpoints/{model_name}{time_str}.pth"
    best_checkpoint_path = f"checkpoints/{model_name}{time_str}_best.pth"
    log_path = f"logs/{model_name}{time_str}.txt"
    curve_path = f"logs/{model_name}{time_str}.png"

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    train_dir = os.path.join(args.data_dir, args.train_split)
    val_dir = os.path.join(args.data_dir, args.val_split)
    test_dir = os.path.join(args.data_dir, args.test_split)

    model = PCNN(
        num_class=args.num_class,
        device=device,
        backbone_path=args.backbone_path,
        require_backbone=not args.allow_random_backbone,
        use_local_enhancement=args.model == "pcnn_local_enhanced",
        use_bidirectional_interaction=args.model == "pcnn_bi_interaction",
        use_gated_fusion=args.model == "pcnn_gated_fusion",
        use_gapw=args.model == "pcnn_gapw",
        gapw_patch_grid=args.gapw_patch_grid,
        gapw_temperature=args.gapw_temperature,
    )

    if not args.no_pretrained_pcnn and os.path.exists(args.pretrained_pcnn):
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
    cudnn.benchmark = torch.cuda.is_available()

    train_loader = make_loader(train_dir, args.batch_size, args.workers, train=True)
    val_loader = make_loader(val_dir, args.batch_size, args.workers, train=False)

    best_acc = 0.0
    write_log(log_path, f"Training time: {now:%m-%d %H:%M}")
    write_log(log_path, f"device: {device}")
    write_log(log_path, "dataset: ferplus")
    write_log(log_path, f"model: {args.model}")
    write_log(log_path, f"seed: {args.seed}")
    write_log(log_path, f"data_dir: {args.data_dir}")
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
    write_log(log_path, f"freeze_backbone: {args.freeze_backbone}")
    write_log(log_path, f"train_gapw_only: {args.train_gapw_only}")
    write_log(log_path, f"head_fusion_weight: {args.head_fusion_weight}")
    if args.val_split == args.test_split:
        write_log(log_path, "WARNING: val_split and test_split are identical. This is not a strict protocol.")

    for epoch in tqdm(range(args.epochs)):
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
                "recorder": recorder,
            },
            is_best,
            checkpoint_path,
            best_checkpoint_path,
        )

        write_log(log_path, f"Current best accuracy: {best_acc:.3f}")
        write_log(log_path, f"An epoch time: {time.time() - start_time:.2f}")

    print(f"best_checkpoint_path: {best_checkpoint_path}")

    if args.test_after_training:
        write_log(log_path, "Loading best checkpoint for final test...")
        checkpoint = load_checkpoint_compat(best_checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        test_loader = make_loader(test_dir, args.batch_size, args.workers, train=False)
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
        if name.startswith(("bidirectional_interaction.", "gated_fusion.", "gapw.")):
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


def make_loader(path, batch_size, workers, train):
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
                transforms.RandomErasing(scale=(0.02, 0.25)),
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

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        targets = remap_ferplus_targets(targets, train_loader.dataset.classes)

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
            targets = remap_ferplus_targets(targets, val_loader.dataset.classes)
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


def remap_ferplus_targets(targets, classes):
    class_order = list(classes)

    if class_order == ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "suprise"]:
        lut = torch.tensor([4, 7, 5, 6, 1, 0, 3, 2], device=targets.device, dtype=torch.long)
        return lut[targets.long()]

    if class_order == [
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

    raise ValueError(
        "Unexpected FERPlus class order. "
        f"Got {class_order}."
    )


def accuracy(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean() * 100.0


def fuse_logits(out, heads, weight):
    if weight == 0:
        return out
    return out + weight * heads


def save_checkpoint(state, is_best, checkpoint_path, best_checkpoint_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


def write_log(log_path, content):
    print(content)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(content + "\n")


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

    def display(self, batch, log_path=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        message = "\t".join(entries)
        if log_path is None:
            print(message)
        else:
            write_log(log_path, message)

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
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = "Accuracy/Loss Curve"
        dpi = 80
        width, height = 1200, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel("Epoch", fontsize=16)
        plt.ylabel("Accuracy", fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color="g", linestyle="-", label="train-accuracy", lw=2)
        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color="y", linestyle="-", label="valid-accuracy", lw=2)

        plt.legend(loc=4, fontsize=legend_fontsize)

        ax2 = plt.gca().twinx()
        ax2.set_ylabel("Loss", fontsize=16)
        ax2.set_ylim(0, max(1, np.max(self.epoch_losses[: self.current_epoch]) * 1.1))
        y_axis[:] = self.epoch_losses[:, 0]
        ax2.plot(x_axis, y_axis, color="g", linestyle=":", label="train-loss", lw=2)
        y_axis[:] = self.epoch_losses[:, 1]
        ax2.plot(x_axis, y_axis, color="y", linestyle=":", label="valid-loss", lw=2)
        ax2.legend(loc=7, fontsize=legend_fontsize)

        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
