import argparse
import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm

from network.models import PCNN


plt.switch_backend("agg")


class RecorderMeter:
    """Compatibility placeholder for legacy checkpoints pickled from __main__."""


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
    parser = argparse.ArgumentParser(description="Evaluate PCNN on FERPlus.")
    parser.add_argument(
        "--data-dir",
        default="/media/ag/SSD/LB/MyDatasets/DATA/FER-PLUS/archive (1)",
        help="FERPlus root directory that contains train/validation/test.",
    )
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--num-class", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--backbone-path", default="models/resnet18_msceleb.pth")
    parser.add_argument("--model-path", default="experiment/ferplus/ferplus.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cudnn.benchmark = torch.cuda.is_available()

    model = PCNN(
        num_class=args.num_class,
        device=device,
        backbone_path=args.backbone_path,
        require_backbone=True,
    ).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Missing PCNN checkpoint: {args.model_path}")

    tqdm.write("Loading FERPlus checkpoint...")
    checkpoint = load_checkpoint_compat(args.model_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)

    data_path = os.path.join(args.data_dir, args.split)
    val_loader = make_loader(data_path, args.batch_size, args.workers)
    criterion_cls = nn.CrossEntropyLoss().to(device)

    now = datetime.datetime.now()
    time_str = now.strftime("[%m-%d]-[%H-%M]-")
    os.makedirs("logs", exist_ok=True)
    cm_path = f"logs/ferplus_{args.split}_{time_str}_confusion_matrix.png"
    validate(val_loader, model, criterion_cls, device, args, cm_path)


def make_loader(path, batch_size, workers):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset folder not found: {path}")

    dataset = datasets.ImageFolder(
        path,
        transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def validate(val_loader, model, criterion_cls, device, args, cm_path):
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Accuracy", ":6.3f")
    progress = ProgressMeter(len(val_loader), [losses, top1], prefix=f"FERPlus-{args.split}: ")
    labels_name = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            targets = remap_ferplus_targets(targets, val_loader.dataset.classes)

            out, heads = model(images)
            loss = (criterion_cls(out, targets) + criterion_cls(heads, targets)) * 10
            acc = accuracy(out, targets)

            losses.update(loss.item(), images.size(0))
            top1.update(acc.item(), images.size(0))
            all_preds.extend(np.argmax(out.cpu().numpy(), axis=-1))
            all_targets.extend(targets.cpu().numpy())

            if i % args.print_freq == 0:
                progress.display(i)

    tqdm.write(f"Accuracy {top1.avg:.3f}")
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_name)
    disp.plot(include_values=True, cmap="Blues", xticks_rotation="vertical")
    plt.title(f"FERPlus Confusion Matrix ({args.split})")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()
    tqdm.write(f"Confusion matrix saved to {cm_path}")
    return top1.avg, losses.avg


def accuracy(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean() * 100.0


def remap_ferplus_targets(targets, classes):
    expected_classes = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "suprise"]
    if list(classes) != expected_classes:
        raise ValueError(
            "Unexpected FERPlus class order. "
            f"Expected {expected_classes}, but got {list(classes)}."
        )

    # Alphabetical folder order -> checkpoint output order used by the author's FERPlus weights.
    # Folder order: angry, contempt, disgust, fear, happy, neutral, sad, suprise
    # Model order : neutral, happy, suprise, sad, angry, disgust, fear, contempt
    lut = torch.tensor([4, 7, 5, 6, 1, 0, 3, 2], device=targets.device, dtype=torch.long)
    return lut[targets.long()]


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

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        tqdm.write("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
