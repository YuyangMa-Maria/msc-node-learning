"""Summarise locally available datasets without redistributing their contents.

The scan combines portable provenance metadata with local file counts. Its
outputs document what was available for a run but do not modify any dataset or
construct a train/test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from nlrisk.data.registry import load_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac"}


def count_files(path: Path) -> dict[str, object]:
    files = [p for p in path.rglob("*") if p.is_file()]
    exts = Counter(p.suffix.lower() or "<none>" for p in files)
    return {
        "exists": path.exists(),
        "file_count": len(files),
        "size_mb": round(sum(p.stat().st_size for p in files) / (1024 * 1024), 2),
        "extensions": dict(exts.most_common()),
    }


def multimodal_summary(path: Path) -> dict[str, object]:
    csv_path = path / "dataset.csv"
    if not csv_path.exists():
        return {}
    labels = Counter()
    rows = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows += 1
            labels[row.get("label", "<missing>")] += 1
    return {"rows": rows, "labels": dict(labels)}


def mendeley_summary(path: Path) -> dict[str, object]:
    return {
        "positive_images": len(list((path / "Positive").glob("*.jpg"))),
        "negative_images": len(list((path / "Negative").glob("*.jpg"))),
    }


def rdd_summary(dataset_root: Path) -> dict[str, object]:
    rows = []
    for folder in sorted(dataset_root.glob("rdd2022-*")):
        if folder.name == "rdd2022-metadata" or not folder.is_dir():
            continue
        inner = next((p for p in folder.iterdir() if p.is_dir()), None)
        if inner is None:
            continue
        rows.append(
            {
                "dataset": folder.name,
                "train_images": len(list((inner / "train" / "images").glob("*.jpg"))),
                "train_xml": len(list((inner / "train" / "annotations" / "xmls").glob("*.xml"))),
                "test_images": len(list((inner / "test" / "images").glob("*.jpg"))),
            }
        )
    return {"subsets": rows}


def orion_summary(path: Path) -> dict[str, object]:
    torque_counts: dict[str, int] = defaultdict(int)
    for mat in path.glob("*.mat"):
        match = re.search(r"salves_out_([^_]+)_B_", mat.name)
        if match:
            torque_counts[match.group(1)] += 1
    return {"torque_file_counts": dict(sorted(torque_counts.items()))}


def write_markdown(inventory: dict[str, object], md_path: Path) -> None:
    lines = ["# Dataset Inventory", "", "Generated from local files.", ""]
    for item in inventory["datasets"]:
        lines.append(f"## {item['id']}")
        lines.append(f"- Path: `{item['path']}`")
        lines.append(f"- Available locally: **{'yes' if item['counts']['exists'] else 'no'}**")
        lines.append(f"- Official source: {item['source_url']}")
        lines.append(f"- Version: {item['version']}")
        lines.append(f"- Licence: {item['licence']['name']} (`{item['licence']['identifier']}`)")
        lines.append(f"- Citation keys: {', '.join(item['citation'])}")
        lines.append(f"- Access: {item['access_notes']}")
        lines.append(f"- Expected layout: {item['expected_layout']}")
        lines.append(f"- Project subset: {item['subset_used']}")
        lines.append(f"- Role: `{item['role']}`")
        lines.append(f"- Modalities: {', '.join(item['modalities'])}")
        lines.append(f"- Used for: {', '.join(item['used_for'])}")
        counts = item["counts"]
        lines.append(f"- Files: {counts['file_count']} ({counts['size_mb']} MB)")
        lines.append(f"- Extensions: `{counts['extensions']}`")
        if item.get("summary"):
            lines.append(f"- Summary: `{item['summary']}`")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory registered datasets and print official sources for missing data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for dataset_inventory.json and dataset_inventory.md.",
    )
    args = parser.parse_args()

    dataset_root_name, entries = load_registry(PROJECT_ROOT)
    dataset_root = PROJECT_ROOT / dataset_root_name
    inventory: dict[str, object] = {"dataset_root": str(dataset_root), "datasets": []}

    for entry in entries:
        path = entry.absolute_path(PROJECT_ROOT, dataset_root_name)
        if "*" in entry.path:
            counts = count_files(dataset_root)
        else:
            counts = count_files(path)

        summary: dict[str, object] = {}
        if entry.id == "multimodal_concrete_crack":
            summary = multimodal_summary(path)
        elif entry.id == "mendeley_concrete_crack_classification":
            summary = mendeley_summary(path)
        elif entry.id == "rdd2022_subsets":
            summary = rdd_summary(dataset_root)
        elif entry.id == "orion_ae_sensor_b_subset":
            summary = orion_summary(path)

        inventory["datasets"].append(
            {
                "id": entry.id,
                "path": entry.path,
                "source_url": entry.source_url,
                "version": entry.version,
                "citation": entry.citation,
                "access_notes": entry.access_notes,
                "licence": {
                    "name": entry.licence.name,
                    "identifier": entry.licence.identifier,
                    "url": entry.licence.url,
                    "redistribution": entry.licence.redistribution,
                },
                "role": entry.role,
                "modalities": entry.modalities,
                "labels": entry.labels,
                "used_for": entry.used_for,
                "subset_used": entry.subset_used,
                "expected_layout": entry.expected_layout,
                "notes": entry.notes,
                "counts": counts,
                "summary": summary,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "dataset_inventory.json"
    md_path = args.output_dir / "dataset_inventory.md"
    json_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(inventory, md_path)
    for item in inventory["datasets"]:
        state = "available" if item["counts"]["exists"] else "missing"
        print(f"[{state}] {item['id']}: {item['path']}")
        if state == "missing":
            print(f"  source: {item['source_url']}")
            print(f"  access: {item['access_notes']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
