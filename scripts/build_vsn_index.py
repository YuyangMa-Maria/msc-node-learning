"""Build the initial visual binary-classification dataset index."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "vsn_binary" / "vsn_binary_index.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def add_mendeley(rows: list[dict[str, str]]) -> None:
    root = DATASET_ROOT / "concrete-crack-images-for-classification-mendeley"
    mapping = {"Positive": 1, "Negative": 0}
    for folder, label in mapping.items():
        class_dir = root / folder
        for image in sorted(class_dir.iterdir()):
            if image.suffix.lower() in IMAGE_EXTS:
                rows.append(
                    {
                        "path": str(image.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "label": str(label),
                        "label_name": "crack" if label == 1 else "normal",
                        "source": "mendeley_concrete_crack",
                    }
                )


def add_multimodal_images(rows: list[dict[str, str]]) -> None:
    root = DATASET_ROOT / "multimodal-concrete-crack-detection-dataset"
    csv_path = root / "dataset.csv"
    if not csv_path.exists():
        return
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for item in reader:
            label_name = item["label"].strip().lower()
            if label_name not in {"normal", "abnormal"}:
                continue
            label = 1 if label_name == "abnormal" else 0
            image_path = root / item["image_path"]
            if image_path.exists():
                rows.append(
                    {
                        "path": str(image_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "label": str(label),
                        "label_name": "crack_or_abnormal" if label == 1 else "normal",
                        "source": "multimodal_concrete_crack_image",
                    }
                )


def stratified_split(rows: list[dict[str, str]], train: float, val: float, seed: int) -> None:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["label"])].append(row)

    rng = random.Random(seed)
    for group_rows in groups.values():
        rng.shuffle(group_rows)
        n = len(group_rows)
        n_train = int(n * train)
        n_val = int(n * val)
        for idx, row in enumerate(group_rows):
            if idx < n_train:
                row["split"] = "train"
            elif idx < n_train + n_val:
                row["split"] = "val"
            else:
                row["split"] = "test"


def summarize(rows: list[dict[str, str]]) -> None:
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        counts[(row["source"], row["split"], row["label"])] += 1
    print("source,split,label,count")
    for key in sorted(counts):
        print(f"{key[0]},{key[1]},{key[2]},{counts[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VSN binary image index.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sources", nargs="+", default=["mendeley", "multimodal"], choices=["mendeley", "multimodal"])
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train <= 0 or args.val <= 0 or args.train + args.val >= 1:
        raise ValueError("Expected train > 0, val > 0, and train + val < 1.")

    rows: list[dict[str, str]] = []
    if "mendeley" in args.sources:
        add_mendeley(rows)
    if "multimodal" in args.sources:
        add_multimodal_images(rows)

    if not rows:
        raise RuntimeError("No rows found for selected sources.")

    stratified_split(rows, args.train, args.val, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "label_name", "source", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")
    print(f"Rows: {len(rows)}")
    summarize(rows)


if __name__ == "__main__":
    main()
