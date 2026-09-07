"""Build binary and multi-label CODEBRIM indices from official annotations.

The official train/validation/test partitions are retained. Crops derived from
the same parent image carry one group identifier, and all-zero annotation rows
are audited separately rather than silently treated as clean backgrounds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = (
    "Background",
    "Crack",
    "Spallation",
    "Efflorescence",
    "ExposedBars",
    "CorrosionStain",
)
DEFECT_LABELS = LABELS[1:]
SPLITS = ("train", "val", "test")
PARENT_PATTERN = re.compile(r"^(image_\d+)_crop_\d+\.png$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_annotations(metadata_dir: Path) -> dict[str, dict[str, int]]:
    annotations: dict[str, dict[str, int]] = {}
    collisions: list[str] = []
    for xml_path in sorted(metadata_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        for defect in root.findall("Defect"):
            filename = defect.attrib["name"]
            labels = {
                label: int(defect.findtext(label, default="0"))
                for label in LABELS
            }
            if filename in annotations and annotations[filename] != labels:
                collisions.append(filename)
            annotations[filename] = labels
    if collisions:
        raise RuntimeError(f"Conflicting XML annotations for {len(collisions)} filenames")
    return annotations


def scan_physical_images(
    dataset_root: Path,
    annotations: dict[str, dict[str, int]],
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    rows: list[dict[str, object]] = []
    physical_names: set[str] = set()
    duplicate_physical_names: list[str] = []
    unannotated_physical: list[str] = []
    for split in SPLITS:
        for storage_class in ("background", "defects"):
            folder = dataset_root / split / storage_class
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing CODEBRIM directory: {folder}")
            for path in sorted(folder.glob("*.png")):
                filename = path.name
                if filename in physical_names:
                    duplicate_physical_names.append(filename)
                    continue
                physical_names.add(filename)
                annotation = annotations.get(filename)
                if annotation is None:
                    unannotated_physical.append(filename)
                    continue
                match = PARENT_PATTERN.match(filename)
                if match is None:
                    raise RuntimeError(f"Unexpected CODEBRIM filename: {filename}")
                damage = int(any(annotation[label] for label in DEFECT_LABELS))
                rows.append(
                    {
                        "path": path,
                        "filename": filename,
                        "split": split,
                        "storage_class": storage_class,
                        "parent_group": match.group(1),
                        "damage": damage,
                        **annotation,
                        "sha256": file_sha256(path),
                    }
                )
    metadata_without_image = sorted(set(annotations) - physical_names)
    return rows, metadata_without_image, unannotated_physical + duplicate_physical_names


def labels_signature(row: dict[str, object]) -> tuple[int, ...]:
    return tuple(int(row[label]) for label in LABELS)


def deduplicate(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    by_hash: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_hash[str(row["sha256"])].append(row)

    retained: list[dict[str, object]] = []
    duplicate_audit: list[dict[str, object]] = []
    excluded_files = 0
    duplicate_files_beyond_first = 0
    cross_split_hashes = 0
    label_conflict_hashes = 0
    within_split_deduplicated_hashes = 0
    for sha256, members in sorted(by_hash.items()):
        if len(members) == 1:
            retained.append(members[0])
            continue
        duplicate_files_beyond_first += len(members) - 1
        splits = sorted({str(member["split"]) for member in members})
        signatures = sorted({labels_signature(member) for member in members})
        paths = sorted(str(member["path"]) for member in members)
        if len(splits) > 1 or len(signatures) > 1:
            action = "exclude_all_copies"
            excluded_files += len(members)
            cross_split_hashes += int(len(splits) > 1)
            label_conflict_hashes += int(len(signatures) > 1)
        else:
            action = "retain_lexicographically_first"
            within_split_deduplicated_hashes += 1
            retained.append(sorted(members, key=lambda row: str(row["path"]))[0])
            excluded_files += len(members) - 1
        duplicate_audit.append(
            {
                "sha256": sha256,
                "copies": len(members),
                "splits": splits,
                "label_signatures": [list(signature) for signature in signatures],
                "paths": paths,
                "action": action,
            }
        )
    summary = {
        "unique_sha256": len(by_hash),
        "duplicate_files_beyond_first": duplicate_files_beyond_first,
        "cross_split_duplicate_hashes": cross_split_hashes,
        "label_conflicting_duplicate_hashes": label_conflict_hashes,
        "within_split_deduplicated_hashes": within_split_deduplicated_hashes,
        "files_excluded_by_duplicate_policy": excluded_files,
    }
    return retained, duplicate_audit, summary


def distribution(rows: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        output[split] = {
            "images": len(split_rows),
            "parent_groups": len({row["parent_group"] for row in split_rows}),
            "damage": dict(
                sorted(Counter(str(int(row["damage"])) for row in split_rows).items())
            ),
            "label_counts": {
                label: sum(int(row[label]) for row in split_rows)
                for label in LABELS
            },
            "multi_defect_images": sum(
                sum(int(row[label]) for label in DEFECT_LABELS) > 1
                for row in split_rows
            ),
        }
    return output


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an audited CODEBRIM multi-label index.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "codebrim" / "classification_dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "vsn_codebrim",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    annotations = parse_annotations(dataset_root / "metadata")
    scanned_rows, metadata_without_image, physical_inventory_errors = scan_physical_images(
        dataset_root,
        annotations,
    )
    all_zero_rows = [
        row for row in scanned_rows if sum(int(row[label]) for label in LABELS) == 0
    ]
    supervised_rows = [
        row for row in scanned_rows if sum(int(row[label]) for label in LABELS) > 0
    ]
    retained_rows, duplicate_audit, duplicate_summary = deduplicate(supervised_rows)

    parent_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    storage_label_mismatches: list[str] = []
    for row in retained_rows:
        parent_splits[str(row["parent_group"])].add(str(row["split"]))
        hash_splits[str(row["sha256"])].add(str(row["split"]))
        expected_background = int(row["storage_class"] == "background")
        if int(row["Background"]) != expected_background:
            storage_label_mismatches.append(str(row["filename"]))

    manifest_rows: list[dict[str, str]] = []
    for row in retained_rows:
        manifest_rows.append(
            {
                "path": Path(row["path"]).relative_to(PROJECT_ROOT).as_posix(),
                "filename": str(row["filename"]),
                "split": str(row["split"]),
                "source": "codebrim",
                "storage_class": str(row["storage_class"]),
                "parent_group": str(row["parent_group"]),
                "damage": str(int(row["damage"])),
                **{label: str(int(row[label])) for label in LABELS},
                "sha256": str(row["sha256"]),
            }
        )
    manifest_rows.sort(key=lambda row: (row["split"], row["parent_group"], row["filename"]))

    report = {
        "dataset": "CODEBRIM classification dataset",
        "task_definition": {
            "binary_head": "damage = OR(Crack, Spallation, Efflorescence, ExposedBars, CorrosionStain)",
            "multi_label_heads": list(DEFECT_LABELS),
            "background": "Background is represented by damage=0 and retained as an audit field.",
            "warning": "Defect type labels are not ordinal risk or severity levels.",
        },
        "protocol": {
            "split_policy": "Retain official train/val/test splits.",
            "parent_group": "image_NNNNNNN source identifier",
            "duplicate_policy": (
                "Exclude all exact copies if labels conflict or copies cross splits; otherwise retain "
                "one canonical copy within a split."
            ),
            "test_policy": "Do not use official test images for model selection, calibration, or threshold tuning.",
        },
        "inventory": {
            "xml_annotation_records": len(annotations),
            "physical_images_joined": len(scanned_rows),
            "all_zero_annotation_files_excluded": len(all_zero_rows),
            "all_zero_annotation_paths": [
                Path(row["path"]).relative_to(PROJECT_ROOT).as_posix()
                for row in all_zero_rows
            ],
            "manifest_images_retained": len(manifest_rows),
            "metadata_records_without_physical_image": metadata_without_image,
            "physical_inventory_errors": physical_inventory_errors,
            **duplicate_summary,
        },
        "distribution": distribution(retained_rows),
        "sanity": {
            "parent_groups_crossing_splits": sum(
                len(splits) > 1 for splits in parent_splits.values()
            ),
            "exact_hashes_crossing_splits_after_policy": sum(
                len(splits) > 1 for splits in hash_splits.values()
            ),
            "storage_class_label_mismatches": len(storage_label_mismatches),
            "storage_class_label_mismatch_examples": storage_label_mismatches[:100],
        },
        "duplicate_records": duplicate_audit,
        "claims_boundary": (
            "CODEBRIM supports visual defect evidence classification. It does not provide "
            "certified structural severity, safety, or collapse-probability labels."
        ),
    }
    if physical_inventory_errors:
        raise RuntimeError(f"Physical inventory errors: {physical_inventory_errors[:10]}")
    if any(
        (
            report["sanity"]["parent_groups_crossing_splits"],
            report["sanity"]["exact_hashes_crossing_splits_after_policy"],
            report["sanity"]["storage_class_label_mismatches"],
        )
    ):
        raise RuntimeError(f"CODEBRIM manifest sanity failed: {report['sanity']}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "codebrim_multitask_index.csv", manifest_rows, list(manifest_rows[0]))
    if all_zero_rows:
        all_zero_audit = [
            {
                "path": Path(row["path"]).relative_to(PROJECT_ROOT).as_posix(),
                "filename": str(row["filename"]),
                "split": str(row["split"]),
                "storage_class": str(row["storage_class"]),
                "parent_group": str(row["parent_group"]),
                "action": "excluded_from_supervised_manifest",
                "reason": "all_six_xml_labels_are_zero",
            }
            for row in all_zero_rows
        ]
        write_csv(
            output_dir / "codebrim_all_zero_annotation_audit.csv",
            all_zero_audit,
            list(all_zero_audit[0]),
        )
    (output_dir / "codebrim_index_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    if duplicate_audit:
        duplicate_rows = [
            {
                "sha256": str(record["sha256"]),
                "copies": str(record["copies"]),
                "splits": "|".join(record["splits"]),
                "label_signatures": json.dumps(record["label_signatures"]),
                "paths": "|".join(record["paths"]),
                "action": str(record["action"]),
            }
            for record in duplicate_audit
        ]
        write_csv(
            output_dir / "codebrim_exact_duplicate_audit.csv",
            duplicate_rows,
            list(duplicate_rows[0]),
        )

    distribution_values = report["distribution"]
    lines = [
        "# CODEBRIM Multi-Label Index Report",
        "",
        "## Task Definition",
        "",
        "- Binary evidence head: any annotated defect.",
        "- Multi-label heads: Crack, Spallation, Efflorescence, ExposedBars, and CorrosionStain.",
        "- Background is represented by binary evidence label 0.",
        "- Defect labels are not ordinal severity or risk levels.",
        "",
        "## Integrity",
        "",
        f"- XML records: {len(annotations):,}",
        f"- Physical images joined: {len(scanned_rows):,}",
        f"- All-zero annotation files excluded: {len(all_zero_rows)}",
        f"- Manifest images retained: {len(manifest_rows):,}",
        f"- Metadata records without a physical image: {len(metadata_without_image)}",
        f"- Exact duplicate files beyond first copy: {duplicate_summary['duplicate_files_beyond_first']}",
        f"- Parent groups crossing official splits: {report['sanity']['parent_groups_crossing_splits']}",
        f"- Exact hashes crossing splits after policy: {report['sanity']['exact_hashes_crossing_splits_after_policy']}",
        "",
        "## Split Distribution",
        "",
        "| Split | Images | Background | Any defect | Parent groups | Multi-defect images |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        values = distribution_values[split]
        lines.append(
            f"| {split} | {values['images']:,} | {values['damage'].get('0', 0):,} | "
            f"{values['damage'].get('1', 0):,} | {values['parent_groups']} | "
            f"{values['multi_defect_images']:,} |"
        )
    lines.extend(
        [
            "",
            "## Test Policy",
            "",
            "The official test split remains frozen. Model selection and calibration must use validation only.",
        ]
    )
    (output_dir / "CODEBRIM_INDEX_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output_dir / 'codebrim_multitask_index.csv'}")


if __name__ == "__main__":
    main()
