"""Select one declared nominal montage from existing, frozen EEG preprocessing."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402

from data.domain import adapt_observed_montage  # noqa: E402
from data.epochs import load_epoch_dataset, save_epoch_dataset  # noqa: E402
from data.source_records import SIX, file_sha256  # noqa: E402
from research.execution import atomic_write_json  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    parent = json.loads(args.prepared.read_text(encoding="utf-8"))
    if parent["status"] != "prepared":
        raise ValueError("nominal view requires a complete validated parent preprocessing")
    result = copy.deepcopy(parent)
    result.update(parent_prepared=str(args.prepared.resolve()), parent_prepared_sha256=file_sha256(args.prepared),
        fitted_preprocessing_states_changed=False, scientific_fits=0)
    for name, cell in result["cells"].items():
        if name.startswith("basic_"):
            continue
        folder = output / name
        folder.mkdir()
        for row in cell["rows"]:
            path = Path(row["path"])
            if file_sha256(path) != row["sha256"]:
                raise ValueError("parent preprocessing input bytes changed")
            before = load_epoch_dataset(path, validation="attested")
            after = adapt_observed_montage(before, SIX, preserve_reference_label=True)
            if (not np.array_equal(before.X, after.X) or not np.array_equal(before.y, after.y)
                    or before.event_timeline.n_events != after.event_timeline.n_events):
                raise RuntimeError("nominal selection unexpectedly changed EEG values, labels or schedule")
            row["parent_cache_sha256"] = row["sha256"]
            target = save_epoch_dataset(folder / (row["session_id"] + ".npz"), after, compressed=False, exclusive=True)
            row.update(path=str(target), sha256=file_sha256(target))
        print(name, "nominal grid selected; EEG values unchanged", flush=True)
    atomic_write_json(output / "prepared.json", result)


if __name__ == "__main__":
    main()
