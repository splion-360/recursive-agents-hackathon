"""Command-line entry point for deterministic ChartQA manifest generation."""

import argparse
from pathlib import Path

from memory_distillation.benchmark.manifests import (
    CHARTQA_REVIEWED_REVISION,
    generate_default_manifest_files,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frozen ChartQA experiment manifests")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-revision",
        required=True,
        help=(
            "Immutable source revision; reviewed upstream commit: "
            f"{CHARTQA_REVIEWED_REVISION}"
        ),
    )
    parser.add_argument("--seed", type=int, default=20261016)
    return parser.parse_args()


def main() -> None:
    """Generate manifest JSON files from a locally supplied ChartQA checkout."""

    args = _parse_args()
    generate_default_manifest_files(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        dataset_revision=args.dataset_revision,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
