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


def parse_args():
    parser = argparse.ArgumentParser(description="Train PCNN for facial expression recognition.")
    parser.add_argument("--dataset", default="rafdb")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--num-class", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-step", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=12)
    parser.add_argument("--beta", type=float, default=8)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--backbone-path", default="models/resnet18_msceleb.pth")
    parser.add_argument("--pretrained-pcnn", default="experiment/rafdb/rafdb.pth")
    parser.add_argument("--no-pretrained-pcnn", action="store_true")
    parser.add_argument("--allow-random-backbone", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--rafdb-numeric-remap",
        action="store_true",
        default=True,
        help="Remap numeric RAF-DB folders (1..7) to model label order.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
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

    train_dir = os.path.join(args.data_root, args.dataset, args.train_split)
    val_dir = os.path.join(args.data_root, args.dataset, args.val_split)

    model = PCNN(
        num_class=args.num_class,
        device=device,
        backbone_path=args.backbone_path,
        require_backbone=not args.allow_random_backbone,
    )

    if not args.no_pretrained_pcnn and os.path.exists(args.pretrained_pcnn):
        checkpoint = load_checkpoint_compat(args.pretrained_pcnn, map_location=device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    criterion_cls = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)
    recorder = RecorderMeter(args.epochs)
    cudnn.benchmark = torch.cuda.is_available()

    train_loader = make_loader(train_dir, args.batch_size, args.workers, train=True)
    val_loader = make_loader(val_dir, args.batch_size, args.workers, train=False)

    best_acc = 0.0
    write_log(log_path, f"Training time: {now:%m-%d %H:%M}")
    write_log(log_path, f"device: {device}")
    write_log(log_path, f"dataset: {args.dataset}")
    write_log(log_path, f"train_split: {args.train_split}")
    write_log(log_path, f"val_split: {args.val_split}")

    for epoch in tqdm(range(args.epochs)):
        start_time = time.time()
        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        write_log(log_path, f"Current learning rate: {current_lr}")

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
            val_acc, val_loss = validate(val_loader, model, criterion_cls, device, args, log_path)

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

    dataset = datasets.ImageFolder(path, transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(train_loader, model, criterion_cls, optimizer, device, args, log_path, epoch):
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Accuracy", ":6.3f")
    progress = ProgressMeter(len(train_loader), [losses, top1], prefix=f"Epoch: [{epoch}]")
    model.train()

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        targets = remap_targets_if_needed(
            targets, args.dataset, train_loader.dataset.classes, args.rafdb_numeric_remap
        )

        optimizer.zero_grad()
        out, heads = model(images)
        loss = criterion_cls(out, targets) * args.alpha + criterion_cls(heads, targets) * args.beta
        acc = accuracy(out, targets)

        losses.update(loss.item(), images.size(0))
        top1.update(acc.item(), images.size(0))

        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            progress.display(i, log_path)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion_cls, device, args, log_path):
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Accuracy", ":6.3f")
    progress = ProgressMeter(len(val_loader), [losses, top1], prefix="Test: ")
    model.eval()

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            targets = remap_targets_if_needed(
                targets, args.dataset, val_loader.dataset.classes, args.rafdb_numeric_remap
            )
            out, heads = model(images)
            loss = criterion_cls(out, targets) * args.alpha + criterion_cls(heads, targets) * args.beta
            acc = accuracy(out, targets)

            losses.update(loss.item(), images.size(0))
            top1.update(acc.item(), images.size(0))

            if i % args.print_freq == 0:
                progress.display(i, log_path)

    write_log(log_path, f"Accuracy {top1.avg:.3f}")
    return top1.avg, losses.avg


def accuracy(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean() * 100.0


def remap_targets_if_needed(targets, dataset_name, classes, enable_remap):
    if not enable_remap:
        return targets
    dataset_name = dataset_name.lower()

    if dataset_name == "rafdb" and list(classes) == ["1", "2", "3", "4", "5", "6", "7"]:
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
