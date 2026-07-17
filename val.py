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
    parser = argparse.ArgumentParser(description="Evaluate PCNN.")
    parser.add_argument("--dataset", default=default_dataset)
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Direct dataset directory containing the requested split. "
            "When set, it takes precedence over --data-root/--dataset."
        ),
    )
    parser.add_argument("--split", default="test")
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--backbone-path", default="models/resnet18_msceleb.pth")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Defaults to the matching RAF-DB or FERPlus author checkpoint.",
    )
    parser.add_argument(
        "--head-fusion-weight",
        type=float,
        default=0.0,
        help="Use out + weight * heads as the main prediction.",
    )
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
    parser.add_argument("--allow-random-backbone", action="store_true")
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
    if args.model_path is None:
        checkpoint_dataset = "ferplus" if is_ferplus else "rafdb"
        args.model_path = f"experiment/{checkpoint_dataset}/{checkpoint_dataset}.pth"
    return args


def main(default_dataset="rafdb"):
    args = parse_args(default_dataset=default_dataset)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cudnn.benchmark = torch.cuda.is_available()

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
    ).to(device)

    if os.path.exists(args.model_path):
        tqdm.write("Loading PCNN checkpoint...")
        checkpoint = load_checkpoint_compat(args.model_path, map_location=device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(
            f"Missing PCNN checkpoint: {args.model_path}. "
            "Download the author's PCNN weights and place them under experiment/rafdb/."
        )

    dataset_dir = args.data_dir or os.path.join(args.data_root, args.dataset)
    data_path = os.path.join(dataset_dir, args.split)
    val_loader = make_loader(data_path, args.batch_size, args.workers)
    criterion_cls = nn.CrossEntropyLoss().to(device)

    now = datetime.datetime.now()
    time_str = now.strftime("[%m-%d]-[%H-%M]-")
    os.makedirs("logs", exist_ok=True)
    cm_path = f"logs/{args.dataset}_{time_str}_confusion_matrix.png"
    validate(val_loader, model, criterion_cls, device, args, cm_path)


def make_loader(path, batch_size, workers):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset folder not found: {path}")

    dataset = datasets.ImageFolder(
        path,
        transforms.Compose(
            [
                transforms.Lambda(lambda image: image.convert("RGB")),
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
    progress = ProgressMeter(len(val_loader), [losses, top1], prefix="Test: ")
    labels_name = display_labels_for_dataset(args.dataset, args.num_class)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            targets = remap_targets_if_needed(
                targets, args.dataset, val_loader.dataset.classes, args.rafdb_numeric_remap
            )
            out, heads = model(images)
            logits = fuse_logits(out, heads, args.head_fusion_weight)
            loss = (criterion_cls(logits, targets) + criterion_cls(heads, targets)) * 10
            acc = accuracy(logits, targets)

            losses.update(loss.item(), images.size(0))
            top1.update(acc.item(), images.size(0))
            all_preds.extend(np.argmax(logits.cpu().numpy(), axis=-1))
            all_targets.extend(targets.cpu().numpy())

            if i % args.print_freq == 0:
                progress.display(i)

    tqdm.write(f"Accuracy {top1.avg:.3f}")
    cm = confusion_matrix(all_targets, all_preds)
    display_labels = labels_name[: len(val_loader.dataset.classes)]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(include_values=True, cmap="Blues", xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()
    tqdm.write(f"Confusion matrix saved to {cm_path}")
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


def display_labels_for_dataset(dataset_name, num_class):
    if "ferplus" in dataset_name.lower():
        return ["Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear", "Contempt"]
    labels = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger"]
    if num_class == 8:
        labels.append("Contempt")
    return labels


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
