"""Construct leakage-resistant grouped splits for formal VSN evaluation.

Exact duplicates and perceptual neighbours are joined transitively before any
split is assigned. A whole connected component therefore moves together,
preventing near-identical patches from inflating validation or test results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from audit_vsn_perceptual_leakage import BKTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HASH_INVENTORY = PROJECT_ROOT / "outputs" / "conference_data_integrity" / "vsn_hash_inventory.csv"
DEFAULT_ASN_INDEX = PROJECT_ROOT / "experiments" / "asn_audio" / "asn_audio_index.csv"
SPLITS = ["train", "val", "test"]
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


class UnionFind:
    """Maintain transitive duplicate groups while similarity edges are added."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass
class CanonicalImage:
    row: dict[str, str]
    sha256: str
    phash: int
    duplicate_paths: list[str]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def normalise_path(value: str) -> str:
    return value.replace("\\", "/").lower()


def source_priority(source: str) -> int:
    priorities = {
        "mendeley_concrete_crack": 0,
        "cconcrack": 1,
        "multimodal_concrete_crack_image": 2,
    }
    return priorities.get(source, 10)


def choose_canonical(rows: list[dict[str, str]]) -> dict[str, str]:
    """Choose one representative path without changing the component label."""
    return sorted(
        rows,
        key=lambda row: (
            source_priority(row.get("source", "")),
            len(row["path"]),
            row["path"],
        ),
    )[0]


def assign_groups(
    groups: dict[int, list[int]],
    canonical: list[CanonicalImage],
    seed: int,
) -> dict[int, str]:
    rng = random.Random(seed)
    assignments: dict[int, str] = {}
    for label in sorted({image.row["label"] for image in canonical}):
        label_groups = [
            (root, members)
            for root, members in groups.items()
            if canonical[members[0]].row["label"] == label
        ]
        rng.shuffle(label_groups)
        label_groups.sort(key=lambda item: len(item[1]), reverse=True)
        total = sum(len(members) for _, members in label_groups)
        targets = {split: total * RATIOS[split] for split in SPLITS}
        current = {split: 0 for split in SPLITS}
        for root, members in label_groups:
            size = len(members)
            split = max(
                SPLITS,
                key=lambda name: (
                    (targets[name] - current[name]) / max(targets[name], 1.0),
                    -current[name],
                ),
            )
            assignments[root] = split
            current[split] += size
    return assignments


def distribution(rows: list[dict[str, str]]) -> dict[str, object]:
    split_counts = Counter(row["split"] for row in rows)
    by_label = {
        split: dict(Counter(row["label"] for row in rows if row["split"] == split))
        for split in SPLITS
    }
    return {
        "total": len(rows),
        "split_counts": dict(split_counts),
        "label_counts": by_label,
    }


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create exact-deduplicated, perceptual-grouped, cross-modal aligned conference splits."
    )
    parser.add_argument("--hash-inventory", type=Path, default=DEFAULT_HASH_INVENTORY)
    parser.add_argument("--asn-index", type=Path, default=DEFAULT_ASN_INDEX)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "conference_grouped",
    )
    parser.add_argument("--max-hamming-distance", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    inventory = read_rows(args.hash_inventory)
    asn_rows = read_rows(args.asn_index)

    by_sha: dict[str, list[dict[str, str]]] = defaultdict(list)
    path_to_inventory: dict[str, dict[str, str]] = {}
    for row in inventory:
        by_sha[row["sha256"]].append(row)
        path_to_inventory[normalise_path(row["path"])] = row

    canonical: list[CanonicalImage] = []
    sha_to_index: dict[str, int] = {}
    path_to_canonical: dict[str, int] = {}
    for sha256, rows in sorted(by_sha.items()):
        labels = {row["label"] for row in rows}
        if len(labels) != 1:
            raise RuntimeError(f"Exact duplicate label conflict for SHA-256 {sha256}")
        chosen = choose_canonical(rows)
        index = len(canonical)
        canonical.append(
            CanonicalImage(
                row={
                    "path": chosen["path"],
                    "label": chosen["label"],
                    "label_name": chosen["label_name"],
                    "source": chosen["source"],
                    "split": "",
                },
                sha256=sha256,
                phash=int(chosen["phash_hex"], 16),
                duplicate_paths=sorted(row["path"] for row in rows),
            )
        )
        sha_to_index[sha256] = index
        for row in rows:
            path_to_canonical[normalise_path(row["path"])] = index

    union_find = UnionFind(len(canonical))
    phash_buckets: dict[int, list[int]] = defaultdict(list)
    for index, image in enumerate(canonical):
        phash_buckets[image.phash].append(index)
    tree = BKTree()
    label_conflicts: list[dict[str, object]] = []
    near_hash_links = 0
    for hash_value, indices in phash_buckets.items():
        for other_hash, distance in tree.query(hash_value, args.max_hamming_distance):
            for left in indices:
                for right in phash_buckets[other_hash]:
                    if canonical[left].row["label"] != canonical[right].row["label"]:
                        if len(label_conflicts) < 200:
                            label_conflicts.append(
                                {
                                    "distance": distance,
                                    "left": canonical[left].row["path"],
                                    "right": canonical[right].row["path"],
                                }
                            )
                        continue
                    union_find.union(left, right)
                    near_hash_links += 1
        for left in indices[1:]:
            if canonical[left].row["label"] == canonical[indices[0]].row["label"]:
                union_find.union(indices[0], left)
                near_hash_links += 1
        tree.add(hash_value)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(canonical)):
        groups[union_find.find(index)].append(index)
    assignments = assign_groups(groups, canonical, args.seed)

    vsn_rows: list[dict[str, str]] = []
    image_to_group: dict[int, str] = {}
    for root, members in groups.items():
        split = assignments[root]
        group_id = f"g_{root:06d}"
        for index in members:
            canonical[index].row["split"] = split
            vsn_rows.append(canonical[index].row.copy())
            image_to_group[index] = group_id

    asn_group_rows: list[dict[str, str]] = []
    missing_pairs: list[str] = []
    for row in asn_rows:
        inventory_row = path_to_inventory.get(normalise_path(row["paired_image_path"]))
        if inventory_row is None:
            missing_pairs.append(row["paired_image_path"])
            continue
        canonical_index = sha_to_index[inventory_row["sha256"]]
        root = union_find.find(canonical_index)
        updated = row.copy()
        updated["split"] = assignments[root]
        asn_group_rows.append(updated)

    group_manifest: list[dict[str, object]] = []
    for root, members in groups.items():
        split = assignments[root]
        group_id = f"g_{root:06d}"
        for index in members:
            image = canonical[index]
            group_manifest.append(
                {
                    "group_id": group_id,
                    "split": split,
                    "label": image.row["label"],
                    "canonical_path": image.row["path"],
                    "sha256": image.sha256,
                    "phash_hex": f"{image.phash:016x}",
                    "exact_duplicate_count": len(image.duplicate_paths),
                    "exact_duplicate_paths": "|".join(image.duplicate_paths),
                }
            )

    output_vsn = args.output_dir / "vsn_binary_index_grouped.csv"
    output_asn = args.output_dir / "asn_audio_index_grouped.csv"
    write_rows(output_vsn, vsn_rows, ["path", "label", "label_name", "source", "split"])
    write_rows(output_asn, asn_group_rows, list(asn_rows[0]))
    write_rows(
        args.output_dir / "vsn_group_manifest.csv",
        [{key: str(value) for key, value in row.items()} for row in group_manifest],
        list(group_manifest[0]),
    )

    group_split_sets: dict[str, set[str]] = defaultdict(set)
    for row in group_manifest:
        group_split_sets[str(row["group_id"])].add(str(row["split"]))
    cross_split_group_errors = sum(len(splits) > 1 for splits in group_split_sets.values())
    exact_duplicates_removed = len(inventory) - len(canonical)
    group_sizes = sorted((len(members) for members in groups.values()), reverse=True)
    report = {
        "purpose": "Publication-oriented leakage-resistant VSN/ASN split construction.",
        "protocol": {
            "exact_duplicate_policy": "Retain one canonical image per SHA-256 content hash.",
            "near_duplicate_policy": (
                f"Place canonical images with 64-bit pHash Hamming distance <= {args.max_hamming_distance} "
                "in the same split when labels agree."
            ),
            "cross_modal_policy": "Assign each ASN audio row to the split of its paired image content group.",
            "ratios": RATIOS,
            "seed": args.seed,
        },
        "input": {
            "vsn_rows": len(inventory),
            "asn_rows": len(asn_rows),
            "unique_exact_images": len(canonical),
        },
        "grouping": {
            "exact_duplicate_rows_removed": exact_duplicates_removed,
            "perceptual_groups": len(groups),
            "near_hash_links": near_hash_links,
            "largest_group_size": max(group_sizes, default=0),
            "groups_larger_than_one": sum(size > 1 for size in group_sizes),
            "label_conflict_links_not_merged": len(label_conflicts),
            "missing_asn_pairs": len(missing_pairs),
            "cross_split_group_errors": cross_split_group_errors,
        },
        "output": {
            "vsn_index": str(output_vsn),
            "asn_index": str(output_asn),
            "vsn": distribution(vsn_rows),
            "asn": distribution(asn_group_rows),
        },
        "limitations": [
            "Perceptual grouping is a conservative automated approximation, not specimen/session metadata.",
            "Large transitive pHash groups should be inspected before final publication use.",
            "All local encoders must be retrained from scratch on these new indices.",
            "VBN remains a separate proxy dataset.",
        ],
        "label_conflict_examples": label_conflicts,
        "missing_pair_examples": missing_pairs[:100],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "grouped_split_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    lines = [
        "# Conference Grouped Split Report",
        "",
        "## Protocol",
        "",
        "- One canonical VSN image retained per SHA-256 content hash.",
        f"- Perceptual neighbours with pHash Hamming distance <= {args.max_hamming_distance} assigned to one group.",
        "- Each group assigned wholly to train, validation, or test.",
        "- ASN audio follows the group of its paired image content.",
        "",
        "## Results",
        "",
        f"- Input VSN rows: {len(inventory)}",
        f"- Exact duplicate rows removed: {exact_duplicates_removed}",
        f"- Canonical VSN images: {len(canonical)}",
        f"- Perceptual groups: {len(groups)}",
        f"- Largest group: {max(group_sizes, default=0)} images",
        f"- Groups larger than one: {sum(size > 1 for size in group_sizes)}",
        f"- Label-conflicting near links not merged: {len(label_conflicts)}",
        f"- Missing ASN pairs: {len(missing_pairs)}",
        f"- Cross-split group errors: {cross_split_group_errors}",
        f"- VSN split counts: {distribution(vsn_rows)['split_counts']}",
        f"- ASN split counts: {distribution(asn_group_rows)['split_counts']}",
        "",
        "## Required Next Step",
        "",
        "Retrain VSN and ASN local encoders from scratch on these indices before repeating shared-layer experiments. "
        "Existing checkpoints have seen the old random splits and cannot be used for the final publication result.",
    ]
    (args.output_dir / "CONFERENCE_GROUPED_SPLIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved grouped conference indices to {args.output_dir}")


if __name__ == "__main__":
    main()
