"""Align paired VSN and ASN records after leakage-resistant VSN splitting.

The image member determines the split for each image-audio pair. Copying that
assignment to ASN prevents paired evidence from the same event appearing in
different train, validation and test partitions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VSN_INDEX = PROJECT_ROOT / "experiments" / "vsn_binary" / "vsn_binary_index.csv"
DEFAULT_ASN_INDEX = PROJECT_ROOT / "experiments" / "asn_audio" / "asn_audio_index.csv"


def normalise_path(value: str) -> str:
    return value.replace("\\", "/").lower()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def distribution(rows: list[dict[str, str]]) -> dict[str, object]:
    split_counts = Counter(row["split"] for row in rows)
    split_labels: dict[str, dict[str, int]] = {}
    for split in sorted(split_counts):
        split_labels[split] = dict(
            sorted(Counter(row.get("label_name", row["label"]) for row in rows if row["split"] == split).items())
        )
    return {
        "total": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "label_distribution": split_labels,
    }


def main() -> None:
    """Join pair identifiers to frozen VSN assignments and write aligned indices."""
    parser = argparse.ArgumentParser(
        description="Create VSN/ASN indices that exclude paired events assigned to different modality splits."
    )
    parser.add_argument("--vsn-index", type=Path, default=DEFAULT_VSN_INDEX)
    parser.add_argument("--asn-index", type=Path, default=DEFAULT_ASN_INDEX)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "conference_aligned",
    )
    args = parser.parse_args()

    vsn_rows = read_rows(args.vsn_index)
    asn_rows = read_rows(args.asn_index)
    vsn_by_path = {normalise_path(row["path"]): row for row in vsn_rows}

    matched_asn: list[dict[str, str]] = []
    mismatched_asn: list[dict[str, str]] = []
    missing_pair_asn: list[dict[str, str]] = []
    paired_paths: set[str] = set()
    mismatched_paths: set[str] = set()
    label_mismatches: list[dict[str, str]] = []

    for asn_row in asn_rows:
        paired_path = normalise_path(asn_row["paired_image_path"])
        paired_paths.add(paired_path)
        vsn_row = vsn_by_path.get(paired_path)
        if vsn_row is None:
            missing_pair_asn.append(asn_row)
            continue
        if vsn_row["label"] != asn_row["label"]:
            label_mismatches.append(
                {
                    "paired_image_path": asn_row["paired_image_path"],
                    "vsn_label": vsn_row["label"],
                    "asn_label": asn_row["label"],
                }
            )
            mismatched_paths.add(paired_path)
            continue
        if vsn_row["split"] != asn_row["split"]:
            mismatched_asn.append(asn_row)
            mismatched_paths.add(paired_path)
            continue
        matched_asn.append(asn_row)

    aligned_vsn = [
        row
        for row in vsn_rows
        if normalise_path(row["path"]) not in paired_paths
        or normalise_path(row["path"]) not in mismatched_paths
    ]
    aligned_asn = matched_asn

    output_vsn = args.output_dir / "vsn_binary_index_aligned.csv"
    output_asn = args.output_dir / "asn_audio_index_aligned.csv"
    write_rows(output_vsn, aligned_vsn, list(vsn_rows[0]))
    write_rows(output_asn, aligned_asn, list(asn_rows[0]))

    report = {
        "purpose": "Exclude cross-modal paired events whose VSN and ASN split assignments disagree.",
        "policy": (
            "Keep each paired image/audio event only when both original modality indices assign it to the same split. "
            "Do not reassign samples, so no retained test sample moves into training for either encoder."
        ),
        "input": {
            "vsn_index": str(args.vsn_index),
            "asn_index": str(args.asn_index),
            "vsn": distribution(vsn_rows),
            "asn": distribution(asn_rows),
        },
        "audit": {
            "paired_events": len(asn_rows),
            "same_split_pairs_retained": len(matched_asn),
            "cross_modal_split_mismatches_excluded": len(mismatched_asn),
            "missing_paired_images": len(missing_pair_asn),
            "label_mismatches_excluded": len(label_mismatches),
        },
        "output": {
            "vsn_index": str(output_vsn),
            "asn_index": str(output_asn),
            "vsn": distribution(aligned_vsn),
            "asn": distribution(aligned_asn),
        },
        "limitations": [
            "This fixes event-level split disagreement for the paired multimodal dataset only.",
            "It does not detect perceptual near-duplicate images from other VSN sources.",
            "The retained modalities are still evaluated independently and are not synchronous fusion validation.",
            "The frozen encoders were trained on their original modality-specific splits; retained test rows remain original test rows.",
        ],
    }
    with (args.output_dir / "alignment_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    lines = [
        "# Conference-Aligned VSN/ASN Split Report",
        "",
        "## Policy",
        "",
        report["policy"],
        "",
        "## Audit",
        "",
        f"- Paired events inspected: {len(asn_rows)}",
        f"- Same-split pairs retained: {len(matched_asn)}",
        f"- Cross-modal split mismatches excluded: {len(mismatched_asn)}",
        f"- Missing paired images: {len(missing_pair_asn)}",
        f"- Label mismatches excluded: {len(label_mismatches)}",
        "",
        "## Output Counts",
        "",
        f"- VSN rows: {len(aligned_vsn)}",
        f"- ASN rows: {len(aligned_asn)}",
        f"- VSN split counts: {distribution(aligned_vsn)['split_counts']}",
        f"- ASN split counts: {distribution(aligned_asn)['split_counts']}",
        "",
        "## Interpretation",
        "",
        "These indices are the minimum defensible dataset protocol for heterogeneous shared-layer experiments. "
        "They prevent one physical image/audio pair from contributing to training in one modality and testing in another.",
        "",
        "This remains a software-level heterogeneous-node experiment, not synchronous tri-modal sensing.",
    ]
    (args.output_dir / "CONFERENCE_ALIGNED_SPLIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved aligned indices and report to {args.output_dir}")


if __name__ == "__main__":
    main()
