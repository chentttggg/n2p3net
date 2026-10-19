"""Freeze original Huebner marker/NEMAR inputs for the existing public caller."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from data.huebner2017_preparation import write_huebner_candidate_declarations  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-marker-census", required=True)
    parser.add_argument("--nemar-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    record = write_huebner_candidate_declarations(args.original_marker_census, args.output, nemar_root=args.nemar_root)
    print({"participants": len(record["participant_declarations"]), "records": len(record["record_statuses"]),
           "public_training_fits": 0})


if __name__ == "__main__":
    main()
