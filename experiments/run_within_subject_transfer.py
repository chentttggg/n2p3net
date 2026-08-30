"""Evaluate chronological single-subject prefix -> suffix digit decoding.

The cache must be causal (``filter_phase=forward``) and must expose a complete
candidate/repetition chain. Training uses earlier repetitions of every digit;
evaluation uses only later repetitions of the same subject.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.contract import (  # noqa: E402
    GTN_SINGLE_SUBJECT_CAUSAL_DATA_CONTRACT,
    PAPER_GTN_CAUSAL_DATA_CONTRACT,
    assert_causal_p300_input_contract,
    assert_gtn_causal_input_contract,
    assert_paper_gtn_causal_input_contract,
)
from data.epochs import load_epoch_dataset  # noqa: E402
from models.n2p3net import POOLING_MODES, N2P3Net  # noqa: E402
from train.device import get_device  # noqa: E402
from transfer.evaluation import hit_at_repetition  # noqa: E402
from transfer.subject_adapter import SubjectAdapter, SubjectAdapterConfig  # noqa: E402
from transfer.within_subject import causal_prefix_suffix_split  # noqa: E402


def _load_trunk(path: str | Path, dataset, *, target_subject: str) -> N2P3Net:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("trunk_state_dict")
    if state is None:
        raise ValueError(f"{path} is not an N2P3 pretraining checkpoint.")
    training_subject_keys = payload.get("training_subject_keys")
    if not isinstance(training_subject_keys, list) or not all(
        isinstance(value, str) for value in training_subject_keys
    ):
        raise ValueError("pretraining checkpoint lacks auditable training_subject_keys.")
    target_key = f"{dataset.name}\0{target_subject}"
    if target_key in set(training_subject_keys):
        raise ValueError(
            f"pretraining checkpoint includes target subject {target_subject!r}; "
            "use a leave-target-out checkpoint."
        )
    trunk = N2P3Net(
        dataset.n_channels,
        n_times=dataset.n_times,
        sfreq=dataset.preprocessing.sfreq,
        tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
        pooling_mode="ms_flatten",
    )
    missing, unexpected = trunk.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint does not match the dataset trunk: missing={missing} unexpected={unexpected}."
        )
    return trunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prefix-reps", type=int, default=8)
    parser.add_argument("--test-reps", type=int, default=8)
    parser.add_argument("--head", choices=("linear", "mlp16", "full_fine"), default="linear")
    parser.add_argument(
        "--pooling-mode",
        default="ms_flatten",
        choices=sorted(POOLING_MODES - {"latency_marginal_contrast"}),
        help="Scratch-trunk readout hypothesis; ignored when --checkpoint is given.",
    )
    parser.add_argument(
        "--temporal-kernel-size",
        type=int,
        default=65,
        help="ST temporal kernel width for the scratch trunk (odd, >=3).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument(
        "--cohort",
        choices=("default", "gtn", "gtn_paper"),
        default="default",
        help=(
            "Causal contract family to assert: 'gtn' enforces the revised 0.1 Hz / "
            "1200 ms child-cohort contract; 'gtn_paper' enforces the paper-aligned "
            "0.5 Hz / 1200 ms contract for SOTA anchors; 'default' enforces 2 Hz / 800 ms."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device != "auto" else get_device()
    dataset = load_epoch_dataset(args.dataset_cache, require_labels=True, validation="attested")
    causal_contract = None
    if args.cohort == "gtn":
        assert_gtn_causal_input_contract(dataset.preprocessing)
        causal_contract = GTN_SINGLE_SUBJECT_CAUSAL_DATA_CONTRACT
    elif args.cohort == "gtn_paper":
        assert_paper_gtn_causal_input_contract(dataset.preprocessing)
        causal_contract = PAPER_GTN_CAUSAL_DATA_CONTRACT
    else:
        assert_causal_p300_input_contract(dataset.preprocessing)
    split = causal_prefix_suffix_split(
        dataset,
        prefix_repetitions=args.prefix_reps,
        test_repetitions=args.test_reps,
        contract=causal_contract,
    )

    records = []
    groups = split.usable_groups
    if args.max_subjects is not None:
        groups = groups[: args.max_subjects]
    for group in groups:
        group_rows = np.flatnonzero(split.group_ids == group)
        target_subjects = np.unique(np.asarray(dataset.subject_ids).astype(str)[group_rows])
        if len(target_subjects) != 1:
            raise ValueError(f"selection group {group!r} must map to exactly one target subject.")
        target_subject = str(target_subjects[0])
        pre_rows = group_rows[split.prefix_mask[group_rows]]
        post_rows = group_rows[split.suffix_mask[group_rows]]
        X_pre = dataset.X[pre_rows]
        y_pre = dataset.y[pre_rows]
        inner_groups = np.asarray(
            [f"{group}:rep{int(rep)}" for rep in split.repetition_indices[pre_rows]]
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed)
            trunk = _load_trunk(
                args.checkpoint,
                dataset,
                target_subject=target_subject,
            ) if args.checkpoint else N2P3Net(
                dataset.n_channels,
                n_times=dataset.n_times,
                sfreq=dataset.preprocessing.sfreq,
                tmin_s=dataset.preprocessing.tmin_ms / 1000.0,
                pooling_mode=args.pooling_mode,
                temporal_kernel_size=args.temporal_kernel_size,
            )
        adapter = SubjectAdapter(
            trunk,
            config=SubjectAdapterConfig(
                head_kind=args.head,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
            ),
            device=device,
        )
        adapter.fit(X_pre, y_pre, group_ids=inner_groups)
        suffix_logits = adapter.predict_logit(dataset.X[post_rows])
        y_post = dataset.y[post_rows]
        auc = None
        if len(np.unique(y_post)) == 2:
            auc = float(roc_auc_score(y_post, suffix_logits))
        hits = hit_at_repetition(
            suffix_logits,
            split.candidate_codes[post_rows],
            split.group_ids[post_rows],
            split.truth_by_group,
            split.suffix_repetition_indices[post_rows],
            aggregation="sum",
            max_repetitions=args.test_reps,
            candidate_vocabulary=split.candidate_vocab,
        )
        records.append(
            {
                "group": group,
                "target_subject": target_subject,
                "n_prefix": int(len(pre_rows)),
                "n_suffix": int(len(post_rows)),
                "binary_auc": auc,
                "hit_by_repetition": hits,
                "parameters": adapter.parameter_count(),
            }
        )
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)

    hit_curves = np.asarray([[rec["hit_by_repetition"].get(r, np.nan) for r in range(1, args.test_reps + 1)] for rec in records])
    summary = {
        "dataset_cache": str(Path(args.dataset_cache).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "prefix_reps": args.prefix_reps,
        "test_reps": args.test_reps,
        "head": args.head,
        "n_groups": len(records),
        "n_subjects": len({rec["target_subject"] for rec in records}),
        "requested_groups": sorted(np.unique(split.group_ids).tolist()),
        "excluded_groups": split.excluded_groups,
        "binary_auc_mean": float(np.nanmean([rec["binary_auc"] for rec in records if rec["binary_auc"] is not None])),
        "hit_mean_by_repetition": {
            str(r): float(np.nanmean(hit_curves[:, r - 1])) for r in range(1, args.test_reps + 1)
        },
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] {output}", flush=True)


if __name__ == "__main__":
    main()
