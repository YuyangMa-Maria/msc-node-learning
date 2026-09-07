"""Restore tracked formal split manifests to historical experiment paths.

Existing files are preserved unless ``--force`` is explicit. The operation
copies metadata only; it never downloads or modifies third-party samples.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKED_ROOT = PROJECT_ROOT / "reproducibility" / "splits"
TARGET_ROOT = PROJECT_ROOT / "experiments"

SPLIT_SETS = {
    "conference_grouped": "conference_grouped",
    "codebrim": "vsn_codebrim",
    "sdnet2018": "vsn_sdnet2018",
    "vbn_orion": "vbn_orion",
}


def restore_split_set(name: str, force: bool) -> tuple[int, int]:
    """Copy one tracked manifest set and report copied/skipped file counts."""
    source_dir = TRACKED_ROOT / name
    target_dir = TARGET_ROOT / SPLIT_SETS[name]
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for source in sorted(path for path in source_dir.iterdir() if path.is_file()):
        target = target_dir / source.name
        if target.exists() and not force:
            skipped += 1
            continue
        shutil.copy2(source, target)
        copied += 1
    return copied, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy the tracked formal manifests into experiments/."
    )
    parser.add_argument(
        "--set",
        choices=["all", *SPLIT_SETS],
        default="all",
        help="Manifest set to restore (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace files already present under experiments/.",
    )
    args = parser.parse_args()

    selected = SPLIT_SETS if args.set == "all" else {args.set: SPLIT_SETS[args.set]}
    for name in selected:
        copied, skipped = restore_split_set(name, args.force)
        print(f"{name}: copied={copied}, skipped_existing={skipped}")


if __name__ == "__main__":
    main()
