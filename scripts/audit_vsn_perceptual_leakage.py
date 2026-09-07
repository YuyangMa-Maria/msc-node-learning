"""Detect exact and perceptually similar VSN images before model evaluation.

SHA-256 identifies byte-for-byte copies, while a perceptual hash catches crops
or re-encodings that remain visually near-identical. The audit reports
cross-split relationships; the grouped-split builder is responsible for keeping
every connected similarity component in one partition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT_ROOT / "experiments" / "vsn_binary" / "vsn_binary_index.csv"
SPLITS = ["train", "val", "test"]


@dataclass
class Record:
    path: str
    split: str
    label: str
    label_name: str
    source: str
    sha256: str
    phash: int


@dataclass
class BKNode:
    value: int
    children: dict[int, "BKNode"] = field(default_factory=dict)


class BKTree:
    """Index perceptual hashes for efficient Hamming-distance neighbour search."""

    def __init__(self) -> None:
        self.root: BKNode | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = BKNode(value)
            return
        node = self.root
        while True:
            distance = self.distance(value, node.value)
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(value)
                return
            node = child

    def query(self, value: int, max_distance: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        matches: list[tuple[int, int]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = self.distance(value, node.value)
            if distance <= max_distance:
                matches.append((node.value, distance))
            lower = distance - max_distance
            upper = distance + max_distance
            for edge, child in node.children.items():
                if lower <= edge <= upper:
                    stack.append(child)
        return matches


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def file_sha256(path: Path) -> str:
    """Hash file content so duplicate detection is independent of file names."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> int:
    """Return a DCT-based perceptual hash for near-duplicate comparison."""
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(grayscale, dtype=np.float32)
    coefficients = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low_frequency = coefficients[:8, :8].copy()
    median = float(np.median(low_frequency.flatten()[1:]))
    bits = low_frequency >= median
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def cross_split_count(records_a: list[Record], records_b: list[Record], same_bucket: bool) -> int:
    if same_bucket:
        counts = Counter(record.split for record in records_a)
        return sum(counts[left] * counts[right] for idx, left in enumerate(SPLITS) for right in SPLITS[idx + 1 :])
    return sum(1 for left in records_a for right in records_b if left.split != right.split)


def add_examples(
    examples: list[dict[str, object]],
    records_a: list[Record],
    records_b: list[Record],
    distance: int,
    same_bucket: bool,
    limit: int,
) -> None:
    if len(examples) >= limit:
        return
    for idx, left in enumerate(records_a):
        start = idx + 1 if same_bucket else 0
        for right in records_b[start:]:
            if left.split == right.split:
                continue
            examples.append(
                {
                    "distance": distance,
                    "left_path": left.path,
                    "left_split": left.split,
                    "left_label": left.label,
                    "left_source": left.source,
                    "right_path": right.path,
                    "right_split": right.split,
                    "right_label": right.label,
                    "right_source": right.source,
                    "label_conflict": left.label != right.label,
                }
            )
            if len(examples) >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact and perceptual cross-split duplicates in the VSN index.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "conference_data_integrity",
    )
    parser.add_argument("--max-hamming-distance", type=int, default=4)
    parser.add_argument("--example-limit", type=int, default=200)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.index.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    records: list[Record] = []
    failures: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        path = resolve_path(row["path"])
        try:
            records.append(
                Record(
                    path=row["path"],
                    split=row["split"],
                    label=row["label"],
                    label_name=row.get("label_name", row["label"]),
                    source=row.get("source", "unknown"),
                    sha256=file_sha256(path),
                    phash=perceptual_hash(path),
                )
            )
        except Exception as error:
            failures.append({"path": row["path"], "error": repr(error)})
        if index % 2000 == 0:
            print(f"[hash] {index}/{len(rows)}")

    sha_buckets: dict[str, list[Record]] = defaultdict(list)
    phash_buckets: dict[int, list[Record]] = defaultdict(list)
    for record in records:
        sha_buckets[record.sha256].append(record)
        phash_buckets[record.phash].append(record)

    with (args.output_dir / "vsn_hash_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "split", "label", "label_name", "source", "sha256", "phash_hex"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "path": record.path,
                    "split": record.split,
                    "label": record.label,
                    "label_name": record.label_name,
                    "source": record.source,
                    "sha256": record.sha256,
                    "phash_hex": f"{record.phash:016x}",
                }
            )

    exact_groups = [
        bucket for bucket in sha_buckets.values() if len({record.split for record in bucket}) > 1
    ]
    exact_examples: list[dict[str, object]] = []
    exact_pair_count = 0
    exact_cross_split_images: set[str] = set()
    for bucket in exact_groups:
        exact_pair_count += cross_split_count(bucket, bucket, same_bucket=True)
        exact_cross_split_images.update(record.path for record in bucket)
        add_examples(exact_examples, bucket, bucket, 0, True, args.example_limit)

    tree = BKTree()
    perceptual_pair_count = 0
    perceptual_cross_split_images: set[str] = set()
    perceptual_examples: list[dict[str, object]] = []
    label_conflict_pairs = 0
    unique_hashes = list(phash_buckets)
    for idx, hash_value in enumerate(unique_hashes):
        bucket = phash_buckets[hash_value]
        within_pairs = cross_split_count(bucket, bucket, same_bucket=True)
        if within_pairs:
            perceptual_pair_count += within_pairs
            perceptual_cross_split_images.update(record.path for record in bucket)
            add_examples(perceptual_examples, bucket, bucket, 0, True, args.example_limit)
            label_conflict_pairs += sum(
                1
                for left_index, left in enumerate(bucket)
                for right in bucket[left_index + 1 :]
                if left.split != right.split and left.label != right.label
            )

        for other_hash, distance in tree.query(hash_value, args.max_hamming_distance):
            other_bucket = phash_buckets[other_hash]
            pair_count = cross_split_count(bucket, other_bucket, same_bucket=False)
            if pair_count == 0:
                continue
            perceptual_pair_count += pair_count
            for left in bucket:
                for right in other_bucket:
                    if left.split != right.split:
                        perceptual_cross_split_images.add(left.path)
                        perceptual_cross_split_images.add(right.path)
                        if left.label != right.label:
                            label_conflict_pairs += 1
            add_examples(perceptual_examples, bucket, other_bucket, distance, False, args.example_limit)
        tree.add(hash_value)
        if (idx + 1) % 5000 == 0:
            print(f"[compare] {idx + 1}/{len(unique_hashes)} unique hashes")

    source_counts = Counter(record.source for record in records)
    split_counts = Counter(record.split for record in records)
    report = {
        "index": str(args.index),
        "protocol": {
            "exact_hash": "SHA-256 file-content hash",
            "perceptual_hash": "64-bit DCT pHash",
            "near_duplicate_threshold": f"Hamming distance <= {args.max_hamming_distance}",
            "scope": "cross-split matches only",
        },
        "dataset": {
            "indexed_rows": len(rows),
            "successfully_hashed": len(records),
            "failures": len(failures),
            "split_counts": dict(split_counts),
            "source_counts": dict(source_counts),
        },
        "exact_content": {
            "cross_split_duplicate_groups": len(exact_groups),
            "cross_split_pair_count": exact_pair_count,
            "affected_image_count": len(exact_cross_split_images),
            "examples": exact_examples,
        },
        "perceptual_near_duplicate": {
            "cross_split_pair_count": perceptual_pair_count,
            "affected_image_count": len(perceptual_cross_split_images),
            "label_conflict_pair_count": label_conflict_pairs,
            "examples": perceptual_examples,
        },
        "failures": failures,
        "limitations": [
            "Perceptual hashes are a screening tool and may produce false positives or miss crop/rotation variants.",
            "Same-surface leakage cannot be ruled out without acquisition-session or specimen identifiers.",
            "Examples are capped, while aggregate counts cover all detected hash matches.",
        ],
    }
    with (args.output_dir / "vsn_perceptual_leakage_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    lines = [
        "# VSN Perceptual Leakage Audit",
        "",
        "## Protocol",
        "",
        "- Exact duplicate detection: SHA-256 file-content hashes.",
        f"- Near-duplicate screening: 64-bit DCT perceptual hash with Hamming distance <= {args.max_hamming_distance}.",
        "- Only cross-split train/validation/test matches are reported.",
        "",
        "## Results",
        "",
        f"- Successfully hashed images: {len(records)} / {len(rows)}",
        f"- Exact cross-split duplicate groups: {len(exact_groups)}",
        f"- Exact cross-split duplicate pairs: {exact_pair_count}",
        f"- Images affected by exact duplicates: {len(exact_cross_split_images)}",
        f"- Perceptual cross-split near-duplicate pairs: {perceptual_pair_count}",
        f"- Images affected by perceptual near-duplicates: {len(perceptual_cross_split_images)}",
        f"- Near-duplicate pairs with conflicting labels: {label_conflict_pairs}",
        f"- Hash/read failures: {len(failures)}",
        "",
        "## Interpretation",
        "",
        "Any detected pair should be treated as a leakage warning until visually reviewed. "
        "A clean hash audit still cannot rule out same-specimen or same-surface leakage when source metadata is absent.",
    ]
    (args.output_dir / "VSN_PERCEPTUAL_LEAKAGE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved VSN leakage audit to {args.output_dir}")


if __name__ == "__main__":
    main()
