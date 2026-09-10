#!/usr/bin/env python3
"""Run GraphKnowledge Phase 1 against the checked-in raw corpus."""

import argparse
from pathlib import Path

from phase1_pipeline import run_phase1


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Meditations knowledge graph")
    parser.add_argument(
        "input_file",
        nargs="?",
        default="data/meditations_raw.txt",
        help="raw Meditations text file",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="data/graph",
        help="directory for generated graph artifacts",
    )
    parser.add_argument("--force-extraction", action="store_true")
    parser.add_argument("--force-dedup", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        parser.error(f"input file not found: {input_path}")

    run_phase1(
        input_path.read_text(encoding="utf-8"),
        output_dir=args.output_dir,
        force_extraction=args.force_extraction,
        force_dedup=args.force_dedup,
    )


if __name__ == "__main__":
    main()