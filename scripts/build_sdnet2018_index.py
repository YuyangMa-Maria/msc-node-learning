"""Build component-aware SDNET2018 train, validation and test indices.

Patch names are traced to their source surface image before splitting. Exact
content with conflicting labels is excluded and recorded explicitly, so no hash
can contribute contradictory supervision.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SURFACES = ("Decks", "Pavements", "Walls")
CLASS_DIRS = {"Non-cracked": 0, "Cracked": 1}
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


class UnionFind:
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


@dataclass(frozen=True)
class ImageRow:
    path: Path
    relative_path: str
    label: int
    label_name: str
    surface: str
    parent_group: str
    sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_dataset(dataset_root: Path, project_root: Path) -> list[ImageRow]:
    rows: list[ImageRow] = []
    for surface in SURFACES:
        for class_dir, label in CLASS_DIRS.items():
            class_root = dataset_root / surface / class_dir
            if not class_root.is_dir():
                raise FileNotFoundError(f"Missing SDNET2018 directory: {class_root}")
            for path in sorted(class_root.glob("*.jpg")):
                parent_prefix = path.stem.split("-", maxsplit=1)[0]
                rows.append(
                    ImageRow(
                        path=path,
                        relative_path=path.relative_to(project_root).as_posix(),
                        label=label,
                        label_name="cracked" if label else "non_cracked",
                        surface=surface,
                        parent_group=f"{surface}:{parent_prefix}",
                        sha256=file_sha256(path),
                    )
                )
    if not rows:
        raise RuntimeError(f"No images found under {dataset_root}")
    return rows


def duplicate_safe_components(
    rows: list[ImageRow],
) -> tuple[dict[str, str], dict[str, object], set[str], list[dict[str, object]]]:
    parent_groups = sorted({row.parent_group for row in rows})
    parent_index = {group: index for index, group in enumerate(parent_groups)}
    union_find = UnionFind(len(parent_groups))
    by_hash: dict[str, list[ImageRow]] = defaultdict(list)
    for row in rows:
        by_hash[row.sha256].append(row)

    conflicting_hashes: list[dict[str, object]] = []
    cross_parent_duplicate_hashes = 0
    duplicate_files = 0
    for sha256, members in by_hash.items():
        if len(members) <= 1:
            continue
        duplicate_files += len(members) - 1
        labels = {member.label for member in members}
        if len(labels) > 1:
            conflicting_hashes.append(
                {
                    "sha256": sha256,
                    "paths": [member.relative_path for member in members],
                    "labels": sorted(labels),
                }
            )
            continue
        member_groups = sorted({member.parent_group for member in members})
        if len(member_groups) > 1:
            cross_parent_duplicate_hashes += 1
            anchor = parent_index[member_groups[0]]
            for group in member_groups[1:]:
                union_find.union(anchor, parent_index[group])

    root_to_groups: dict[int, list[str]] = defaultdict(list)
    for group in parent_groups:
        root_to_groups[union_find.find(parent_index[group])].append(group)

    parent_to_component: dict[str, str] = {}
    for component_number, (_, groups) in enumerate(sorted(root_to_groups.items())):
        component_id = f"sdnet_component_{component_number:04d}"
        for group in groups:
            parent_to_component[group] = component_id

    audit = {
        "unique_sha256": len(by_hash),
        "duplicate_files_beyond_first": duplicate_files,
        "cross_parent_duplicate_hashes": cross_parent_duplicate_hashes,
        "parent_groups": len(parent_groups),
        "duplicate_safe_components": len(root_to_groups),
        "label_conflicting_hashes": len(conflicting_hashes),
        "label_conflicting_files_excluded": sum(
            len(record["paths"]) for record in conflicting_hashes
        ),
    }
    conflict_hashes = {str(record["sha256"]) for record in conflicting_hashes}
    return parent_to_component, audit, conflict_hashes, conflicting_hashes


def component_vectors(
    rows: list[ImageRow], parent_to_component: dict[str, str]
) -> tuple[list[str], dict[str, Counter[tuple[str, int]]], dict[str, Counter[str]]]:
    vectors: dict[str, Counter[tuple[str, int]]] = defaultdict(Counter)
    group_vectors: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        component = parent_to_component[row.parent_group]
        vectors[component][(row.surface, row.label)] += 1
        group_vectors[component][row.parent_group] = 1
    return sorted(vectors), vectors, group_vectors


def assignment_score(
    assignments: dict[str, str],
    components: list[str],
    vectors: dict[str, Counter[tuple[str, int]]],
    group_vectors: dict[str, Counter[str]],
) -> float:
    total_samples: Counter[tuple[str, int]] = Counter()
    total_groups: Counter[str] = Counter()
    actual_samples: dict[str, Counter[tuple[str, int]]] = {split: Counter() for split in SPLITS}
    actual_groups: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for component in components:
        total_samples.update(vectors[component])
        for parent_group in group_vectors[component]:
            total_groups[parent_group.split(":", maxsplit=1)[0]] += 1
        split = assignments[component]
        actual_samples[split].update(vectors[component])
        for parent_group in group_vectors[component]:
            actual_groups[split][parent_group.split(":", maxsplit=1)[0]] += 1

    score = 0.0
    for split in SPLITS:
        ratio = RATIOS[split]
        for stratum, total in total_samples.items():
            target = total * ratio
            score += ((actual_samples[split][stratum] - target) / max(target, 1.0)) ** 2
        for surface, total in total_groups.items():
            target = total * ratio
            score += 0.25 * ((actual_groups[split][surface] - target) / max(target, 1.0)) ** 2
    return score


def optimise_assignment(
    components: list[str],
    vectors: dict[str, Counter[tuple[str, int]]],
    group_vectors: dict[str, Counter[str]],
    seed: int,
    trials: int,
) -> tuple[dict[str, str], float]:
    rng = random.Random(seed)
    n_components = len(components)
    n_train = round(n_components * RATIOS["train"])
    n_val = round(n_components * RATIOS["val"])
    best_assignment: dict[str, str] | None = None
    best_score = float("inf")

    candidate = components.copy()
    for _ in range(trials):
        rng.shuffle(candidate)
        assignment = {
            component: (
                "train"
                if index < n_train
                else "val"
                if index < n_train + n_val
                else "test"
            )
            for index, component in enumerate(candidate)
        }
        score = assignment_score(assignment, components, vectors, group_vectors)
        if score < best_score:
            best_score = score
            best_assignment = assignment.copy()

    if best_assignment is None:
        raise RuntimeError("Failed to construct a grouped split")
    return best_assignment, best_score


def nested_distribution(records: list[dict[str, str]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for split in SPLITS:
        split_rows = [row for row in records if row["split"] == split]
        result[split] = {
            "images": len(split_rows),
            "labels": dict(sorted(Counter(row["label_name"] for row in split_rows).items())),
            "surfaces": dict(sorted(Counter(row["surface"] for row in split_rows).items())),
            "surface_labels": {
                surface: dict(
                    sorted(
                        Counter(
                            row["label_name"]
                            for row in split_rows
                            if row["surface"] == surface
                        ).items()
                    )
                )
                for surface in SURFACES
            },
            "parent_groups": len({row["parent_group"] for row in split_rows}),
            "components": len({row["component_id"] for row in split_rows}),
        }
    return result


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-resistant SDNET2018 grouped index.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "sdnet2018",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "vsn_sdnet2018",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--trials", type=int, default=50000)
    args = parser.parse_args()

    project_root = PROJECT_ROOT.resolve()
    dataset_root = args.dataset_root.resolve()
    rows = scan_dataset(dataset_root, project_root)
    parent_to_component, duplicate_audit, conflict_hashes, conflicting_hashes = (
        duplicate_safe_components(rows)
    )
    clean_rows = [row for row in rows if row.sha256 not in conflict_hashes]
    components, vectors, group_vectors = component_vectors(clean_rows, parent_to_component)
    assignments, score = optimise_assignment(
        components,
        vectors,
        group_vectors,
        seed=args.seed,
        trials=args.trials,
    )

    records: list[dict[str, str]] = []
    for row in clean_rows:
        component = parent_to_component[row.parent_group]
        records.append(
            {
                "path": row.relative_path,
                "label": str(row.label),
                "label_name": row.label_name,
                "source": "sdnet2018",
                "surface": row.surface,
                "parent_group": row.parent_group,
                "component_id": component,
                "sha256": row.sha256,
                "split": assignments[component],
            }
        )
    records.sort(key=lambda row: (row["split"], row["surface"], row["parent_group"], row["path"]))

    component_split_sets: dict[str, set[str]] = defaultdict(set)
    parent_split_sets: dict[str, set[str]] = defaultdict(set)
    hash_split_sets: dict[str, set[str]] = defaultdict(set)
    for row in records:
        component_split_sets[row["component_id"]].add(row["split"])
        parent_split_sets[row["parent_group"]].add(row["split"])
        hash_split_sets[row["sha256"]].add(row["split"])

    report = {
        "dataset": "SDNET2018",
        "purpose": "Frozen leakage-resistant target-domain split for VSN zero-shot and transfer experiments.",
        "protocol": {
            "grouping": "All patches with the same surface and original-image filename prefix remain together.",
            "exact_duplicate_policy": "Parent groups connected by an exact SHA-256 duplicate are merged before assignment.",
            "assignment": "Random-search grouped assignment minimising surface-by-label distribution error.",
            "ratios": RATIOS,
            "seed": args.seed,
            "trials": args.trials,
            "test_freeze_policy": "The test split must not be used for target-domain training, threshold selection, or calibration.",
        },
        "inventory": {
            "physical_images_scanned": len(rows),
            "images_retained": len(records),
            "label_conflicting_files_excluded": len(rows) - len(clean_rows),
            "labels": dict(sorted(Counter(row["label_name"] for row in records).items())),
            "surfaces": dict(sorted(Counter(row["surface"] for row in records).items())),
            **duplicate_audit,
        },
        "label_conflict_records": conflicting_hashes,
        "assignment_score": score,
        "distribution": nested_distribution(records),
        "sanity": {
            "component_cross_split_errors": sum(len(value) > 1 for value in component_split_sets.values()),
            "parent_group_cross_split_errors": sum(len(value) > 1 for value in parent_split_sets.values()),
            "exact_hash_cross_split_errors": sum(len(value) > 1 for value in hash_split_sets.values()),
        },
        "claims_boundary": (
            "The split supports visual crack domain-shift evaluation. It does not provide structural "
            "failure probabilities or post-disaster field validation."
        ),
    }
    if any(report["sanity"].values()):
        raise RuntimeError(f"Split sanity check failed: {report['sanity']}")

    output_dir = args.output_dir
    index_path = output_dir / "sdnet2018_index_grouped.csv"
    write_csv(index_path, records, list(records[0]))
    conflict_rows = [
        {
            "sha256": str(record["sha256"]),
            "labels": "|".join(str(value) for value in record["labels"]),
            "paths": "|".join(str(value) for value in record["paths"]),
            "action": "excluded_all_copies",
        }
        for record in conflicting_hashes
    ]
    if conflict_rows:
        write_csv(
            output_dir / "sdnet2018_label_conflicts.csv",
            conflict_rows,
            list(conflict_rows[0]),
        )

    component_rows: list[dict[str, str]] = []
    for component in components:
        parents = sorted(group_vectors[component])
        component_rows.append(
            {
                "component_id": component,
                "split": assignments[component],
                "parent_groups": "|".join(parents),
                "parent_group_count": str(len(parents)),
                "image_count": str(sum(vectors[component].values())),
                **{
                    f"{surface}_{label_name}": str(vectors[component][(surface, label)])
                    for surface in SURFACES
                    for label, label_name in ((0, "non_cracked"), (1, "cracked"))
                },
            }
        )
    write_csv(output_dir / "sdnet2018_group_manifest.csv", component_rows, list(component_rows[0]))
    (output_dir / "sdnet2018_split_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# SDNET2018 Grouped Split Report",
        "",
        "## Protocol",
        "",
        "- Group key: surface type plus original-image filename prefix.",
        "- Exact duplicates connect their parent groups before split assignment.",
        "- Target ratios: 70% train, 15% validation, and 15% frozen test.",
        "- The frozen test split must not be used for fine-tuning, calibration, or threshold selection.",
        "",
        "## Audit",
        "",
        f"- Physical images scanned: {len(rows):,}",
        f"- Images retained: {len(records):,}",
        f"- Label-conflicting files excluded: {len(rows) - len(clean_rows)}",
        f"- Label-conflicting exact hashes: {duplicate_audit['label_conflicting_hashes']}",
        f"- Parent groups: {duplicate_audit['parent_groups']}",
        f"- Duplicate-safe components: {duplicate_audit['duplicate_safe_components']}",
        f"- Exact duplicate files beyond first occurrence: {duplicate_audit['duplicate_files_beyond_first']}",
        f"- Cross-split component errors: {report['sanity']['component_cross_split_errors']}",
        f"- Cross-split parent-group errors: {report['sanity']['parent_group_cross_split_errors']}",
        f"- Cross-split exact-hash errors: {report['sanity']['exact_hash_cross_split_errors']}",
        "",
        "## Distribution",
        "",
        "| Split | Images | Non-cracked | Cracked | Parent groups | Components |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        values = report["distribution"][split]
        lines.append(
            f"| {split} | {values['images']:,} | "
            f"{values['labels'].get('non_cracked', 0):,} | "
            f"{values['labels'].get('cracked', 0):,} | "
            f"{values['parent_groups']} | {values['components']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This split is suitable for zero-shot reporting and future target-domain transfer experiments. "
            "The complete dataset may be reported as an additional descriptive zero-shot result because no "
            "SDNET2018 sample was used to train the existing VSN, but all model selection after this point must "
            "use the grouped train/validation subsets and preserve the grouped test subset.",
        ]
    )
    (output_dir / "SDNET2018_SPLIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
