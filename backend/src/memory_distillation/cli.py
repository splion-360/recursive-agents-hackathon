"""Command-line entry point for explicit experiment lifecycle operations."""

import argparse

from memory_distillation.calibration.command import main as calibrate_main
from memory_distillation.versioning.command import main as lifecycle_main


def main() -> None:
    parser = argparse.ArgumentParser(prog="memory-distillation")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser(
        "calibrate",
        add_help=False,
        help="Calibrate and freeze live ChartQA Kappa guidance",
    )
    subcommands.add_parser(
        "lifecycle",
        add_help=False,
        help="Run one measured Collection-to-Promotion lifecycle",
    )
    args, remaining = parser.parse_known_args()
    if args.command == "calibrate":
        calibrate_main(remaining)
    elif args.command == "lifecycle":
        lifecycle_main(remaining)


if __name__ == "__main__":
    main()
