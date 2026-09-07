"""Build the structural time-series proxy index from Orion AE recordings."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "orion-ae-sensor-b-subset"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "vbn_orion" / "vbn_orion_index.csv"

TORQUE_RE = re.compile(r"_(\d+)cNm_")


def torque_from_name(path: Path) -> int:
    match = TORQUE_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse torque from {path.name}")
    return int(match.group(1))


def risk_label(torque: int) -> tuple[int | None, str]:
    if torque <= 10:
        return 0, "low_torque_proxy_normal"
    if torque >= 40:
        return 1, "high_torque_proxy_abnormal"
    return None, "moderate_torque_excluded_binary"


def split_for_index(index: int, count: int) -> str:
    train_cut = int(count * 0.60)
    val_cut = int(count * 0.80)
    if index < train_cut:
        return "train"
    if index < val_cut:
        return "val"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VBN Orion AE proxy index.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    files = sorted(DATASET_ROOT.glob("*.mat"))
    by_torque: dict[int, list[Path]] = {}
    for path in files:
        by_torque.setdefault(torque_from_name(path), []).append(path)

    rows: list[dict[str, str]] = []
    for torque, paths in sorted(by_torque.items()):
        label, label_name = risk_label(torque)
        for idx, path in enumerate(paths):
            split = split_for_index(idx, len(paths))
            rows.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "torque_cnm": str(torque),
                    "label": "" if label is None else str(label),
                    "label_name": label_name,
                    "source": "orion_ae_sensor_b_proxy",
                    "split": split,
                    "use_binary": "0" if label is None else "1",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "torque_cnm", "label", "label_name", "source", "split", "use_binary"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")
    print(f"Rows: {len(rows)}")
    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (row["torque_cnm"], row["split"], row["use_binary"])
        counts[key] = counts.get(key, 0) + 1
    print("torque,split,use_binary,count")
    for key in sorted(counts, key=lambda x: (int(x[0]), x[1], x[2])):
        print(f"{key[0]},{key[1]},{key[2]},{counts[key]}")


if __name__ == "__main__":
    main()
