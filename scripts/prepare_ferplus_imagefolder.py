import argparse
import csv
import shutil
from pathlib import Path


EMOTIONS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]


def majority_label(votes):
    total = sum(votes)
    if total <= 0:
        return None
    best_idx = max(range(len(votes)), key=votes.__getitem__)
    if votes[best_idx] <= 0.5 * total:
        return None
    if best_idx >= len(EMOTIONS):
        return None
    return EMOTIONS[best_idx]


def convert_split(source_dir, output_dir, split_name):
    label_path = source_dir / "label.csv"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    counts = {name: 0 for name in EMOTIONS}
    skipped = 0
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in EMOTIONS:
        (split_dir / name).mkdir(parents=True, exist_ok=True)

    with label_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            image_name = row[0]
            votes = [float(value) for value in row[2:]]
            label = majority_label(votes)
            if label is None:
                skipped += 1
                continue

            source_image = source_dir / image_name
            if not source_image.exists():
                skipped += 1
                continue
            shutil.copy2(source_image, split_dir / label / image_name)
            counts[label] += 1

    return counts, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Convert official FERPlus label.csv folders to ImageFolder majority labels."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    split_map = {
        "FER2013Train": "train",
        "FER2013Valid": "validation",
        "FER2013Test": "test",
    }

    for source_name, output_name in split_map.items():
        counts, skipped = convert_split(args.source / source_name, args.output, output_name)
        total = sum(counts.values())
        print(f"{output_name}: kept={total} skipped={skipped}")
        for name in EMOTIONS:
            print(f"  {name}: {counts[name]}")


if __name__ == "__main__":
    main()
