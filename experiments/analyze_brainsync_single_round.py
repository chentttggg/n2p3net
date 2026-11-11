"""Nine-choice R1 recognition: each nine-flash cycle is scored independently."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402
from scipy.special import softmax  # noqa: E402

from data.candidate_task import MembershipDecision  # noqa: E402
from data.epochs import load_epoch_dataset  # noqa: E402
from models.candidate_evidence import (  # noqa: E402
    CandidateEvidence,
    CandidateExpertIdentity,
    membership_decision_scores,
)
from research.contracts import semantic_sha256  # noqa: E402
from transfer.ensemble import EnsembleProgram, EnsembleScorer  # noqa: E402

DOSES = (1, 3, 5, 10)
METHODS = ("nn_source", "nn_scratch", "xdawn_erp", "three_experts")
LEAVES = ("nn_source", "waveform_lda", "xdawn_erp")
VOCAB = tuple(map(str, range(1, 10)))
LABELS = {
    "nn_source": "NN source",
    "nn_scratch": "NN scratch",
    "xdawn_erp": "xDAWN",
    "three_experts": "三混合：source＋LDA＋xDAWN",
}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )


def round_evidence(raw, decision, identity, *, participant, decision_id, support_digest):
    """Only the supplied nine events enter this call; no target label enters."""
    scores = membership_decision_scores(raw, decision, available=np.isfinite(raw))
    return CandidateEvidence.from_membership_scores(
        scores,
        decision,
        participant_key=participant,
        decision_id=decision_id,
        physical_support_digest=support_digest,
        expert_identities=(identity,),
    )


def metrics(records):
    count = len(records)
    return {
        "requested_rounds": count,
        "hits": sum(r["prediction"] == r["target"] for r in records),
        "predicted_rounds": sum(r["prediction"] is not None for r in records),
        "probability_rounds": sum(r["probability_available"] for r in records),
        "target_recognition_rate": sum(r["prediction"] == r["target"] for r in records) / count,
        "prediction_coverage": sum(r["prediction"] is not None for r in records) / count,
        "ties": sum(r["status"] == "tie" for r in records),
        "unavailable_rounds": sum(not r["probability_available"] for r in records),
    }


def aggregate(seed_records):
    ms = [r["metrics"] for r in seed_records]
    count = ms[0]["requested_rounds"]
    assert all(m["requested_rounds"] == count for m in ms)
    return {
        "requested_rounds_per_seed": count,
        "target_optimization_repeats": len(ms),
        "mean_hits": float(np.mean([m["hits"] for m in ms])),
        "target_recognition_rate": float(np.mean([m["target_recognition_rate"] for m in ms])),
        "mean_predicted_rounds": float(np.mean([m["predicted_rounds"] for m in ms])),
        "prediction_coverage": float(np.mean([m["prediction_coverage"] for m in ms])),
        "hit_range": [min(m["hits"] for m in ms), max(m["hits"] for m in ms)],
        "per_seed": seed_records,
    }


def numerical_checks():
    decision = MembershipDecision.direct(
        VOCAB,
        VOCAB,
        event_ids=tuple(f"event:{q}" for q in VOCAB),
        repetition_indices=np.zeros(9, dtype=int),
        repetitions=1,
    )

    def evidence(family, values):
        identity = CandidateExpertIdentity(
            family, "a" * 64, "b" * 64, None, "c" * 64, "membership_llr_sum/1", False
        )
        return round_evidence(
            np.asarray(values, dtype=float),
            decision,
            identity,
            participant="software",
            decision_id="round:1",
            support_digest="d" * 64,
        )

    first = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 7.0, 8.0])
    e = evidence("nn_source", first)
    assert e.predicted == "7" and np.allclose(e.probability, softmax(first), rtol=0, atol=1e-12)
    # A different cycle is never added to the first cycle's evidence.
    second = first[::-1] * 100
    evidence("nn_source", second)
    assert np.array_equal(e.probability, evidence("nn_source", first).probability)
    assert np.allclose(
        e.probability, evidence("nn_source", first + 23.0).probability, rtol=0, atol=1e-12
    )
    program = EnsembleProgram.equal_weights("three_experts", LEAVES)
    leaves = {f: evidence(f, first + i * np.arange(9)) for i, f in enumerate(LEAVES)}
    pooled = EnsembleScorer(program).score(leaves)
    assert np.allclose(
        pooled.probability,
        np.mean([p.probability for p in leaves.values()], axis=0),
        rtol=0,
        atol=1e-12,
    )
    missing = first.copy()
    missing[0] = np.nan
    leaves["waveform_lda"] = evidence("waveform_lda", missing)
    assert not EnsembleScorer(program).score(leaves).probability_available
    tied = evidence("nn_source", np.zeros(9))
    assert tied.probability_available and tied.predicted is None
    return {
        "nine_candidate_direct_oracle": True,
        "no_cumulative_cycles": True,
        "constant_class_cost_offset_invariance": True,
        "arithmetic_pool_oracle": True,
        "missing_leaf_preserves_unavailability": True,
        "tie_is_not_a_guessed_target": True,
    }


def analyze(run, output):
    original_plan = read(run / "plan.json")
    jobs = read(run / "jobs.json")
    parent_hashes = read(run / "analysis/result_bindings.json")
    recovery = read(run / "recovery_map.json")
    source = read(run / "execution_source/source.manifest.json")["files"]
    for path in (
        "src/models/candidate_evidence.py",
        "src/transfer/ensemble.py",
        "src/data/candidate_task.py",
    ):
        assert sha(ROOT / path) == source[path]
    rows = {r["session_id"]: r for r in original_plan["rows"]}
    index = {
        (j["pipeline"], j["dose"], j["family"], j["seed"]): j
        for j in jobs
        if j["backend"] == "current"
    }
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema": "n2p3_brainsync_single_round_plan/1",
        "human_confirmed_unit": "9 digits flashed once each; independent R1 recognition; no accumulation across the five cycles",
        "methods": list(METHODS),
        "three_expert_components": list(LEAVES),
        "calibration_record_counts": list(DOSES),
        "test_sessions": original_plan["test_sessions"],
        "candidate_vocabulary": list(VOCAB),
        "requested_rounds": 100,
        "epochs_per_round": 9,
        "random_guess_baseline": 1 / 9,
        "source_plan_sha256": sha(run / "plan.json"),
        "temperature": 1.0,
        "mixture_weights": [1 / 3] * 3,
        "missing_or_tied_prediction": "no hit; requested denominator stays 100",
        "fitting": "zero new EEG, preprocessing, probability, threshold or gating fits",
        "preprocessing_availability": "unchanged frozen whole-record processing; retrospective R1 epoch aggregation, not online single-cycle acquisition",
        "seed_scope": "three target optimization seeds at one fixed public source; xDAWN is reused deterministic state",
    }
    write(output / "plan.json", plan)
    checks = numerical_checks()
    program = EnsembleProgram.equal_weights("three_experts", LEAVES)
    summaries, full_records, bindings = {}, {}, {}
    total_oracles = 0
    for pipeline in ("basic", "reviewed"):
        datasets = {}
        for session in plan["test_sessions"]:
            entry = rows[session]["paths"][pipeline]
            assert sha(run / entry["path"]) == entry["sha256"]
            ds = load_epoch_dataset(run / entry["path"], require_labels=True, validation="attested")
            assert ds.event_timeline.n_events == 450
            datasets[session] = ds
            for round_index in range(50):
                ii = np.arange(round_index * 9, (round_index + 1) * 9)
                assert sorted(ds.event_timeline.candidate_ids[ii].astype(str)) == list(VOCAB)
        panel = {}
        for dose in DOSES:
            by_method = {m: [] for m in METHODS}
            for seed in original_plan["target_seeds"]:
                leaf_data = {}
                for family in ("nn_source", "nn_scratch", "xdawn_erp", "waveform_lda"):
                    actual_seed = (
                        seed if family.startswith("nn_") else original_plan["target_seeds"][0]
                    )
                    job = index[pipeline, dose, family, actual_seed]
                    path = run / "results" / job["id"] / "result.json"
                    if job["id"] in recovery:
                        path = run / recovery[job["id"]]["local_result"]
                    assert sha(path) == parent_hashes[job["id"]]
                    fitted = read(path)
                    assert fitted["status"] == "completed"
                    assert set(fitted["calibration_sessions"]) == set(
                        original_plan["calibration_order"][:dose]
                    )
                    assert not set(fitted["calibration_sessions"]) & set(plan["test_sessions"])
                    arrays, prediction_hashes = {}, {}
                    for session in plan["test_sessions"]:
                        ds = datasets[session]
                        pred = path.parent / f"{session}_predictions.npz"
                        prediction_hashes[session] = sha(pred)
                        with np.load(pred, allow_pickle=False) as archive:
                            assert np.array_equal(archive["event_ids"], ds.event_timeline.event_ids)
                            assert np.array_equal(
                                archive["evidence_indices"], ds.event_timeline.evidence_indices
                            )
                            indices = ds.event_timeline.evidence_indices
                            present = indices >= 0
                            values = np.full(450, np.nan)
                            values[present] = archive["logits"][indices[present]]
                        arrays[session] = values
                    bindings[job["id"]] = {
                        "result_file": str(path),
                        "result_sha256": sha(path),
                        "model_sha256": fitted["model_sha256"],
                        "prediction_sha256": prediction_hashes,
                        "actual_fit_job": fitted["job"],
                        "calibration_sessions": fitted["calibration_sessions"],
                    }
                    leaf_data[family] = (job, fitted, arrays, prediction_hashes)
                per_method = {m: [] for m in METHODS}
                for session in plan["test_sessions"]:
                    ds = datasets[session]
                    timeline = ds.event_timeline
                    truth = str(rows[session]["target"])
                    for round_index in range(50):
                        ii = np.arange(round_index * 9, (round_index + 1) * 9)
                        codes = timeline.candidate_ids[ii].astype(str)
                        event_ids = timeline.event_ids[ii].astype(str)
                        decision = MembershipDecision.direct(
                            codes,
                            VOCAB,
                            event_ids=event_ids,
                            repetition_indices=np.zeros(9, dtype=int),
                            repetitions=1,
                            evidence_indices=timeline.evidence_indices[ii],
                        )
                        round_id = (
                            f"{session}:block:{round_index // 5 + 1}:cycle:{round_index % 5 + 1}"
                        )
                        support = semantic_sha256(
                            {
                                "session_id": session,
                                "events": event_ids.tolist(),
                                "vocabulary": VOCAB,
                                "availability": "complete_original_recording",
                            }
                        )
                        evidence = {}
                        for family, (job, fitted, arrays, pred_hashes) in leaf_data.items():
                            values = arrays[session][ii]
                            identity = CandidateExpertIdentity(
                                family,
                                semantic_sha256(
                                    {
                                        "prediction_sha256": pred_hashes[session],
                                        "event_ids": event_ids.tolist(),
                                    }
                                ),
                                fitted["model_sha256"],
                                "935ee55de598256ca9299595ec899359751b1ebbf5fa0bb15af0f0fbb8bda3bd"
                                if family == "nn_source"
                                else None,
                                semantic_sha256(
                                    {
                                        "actual_job": fitted["job"],
                                        "result_sha256": parent_hashes[job["id"]],
                                    }
                                ),
                                "membership_llr_sum_R1/1",
                                False,
                            )
                            evidence[family] = round_evidence(
                                values,
                                decision,
                                identity,
                                participant="n2p3-origin://BrainSync-GTN/participant/P001",
                                decision_id=round_id,
                                support_digest=support,
                            )
                            if np.isfinite(values).all():
                                manual = np.array([values[codes == q].sum() for q in VOCAB])
                                assert np.allclose(
                                    evidence[family].probability,
                                    softmax(manual),
                                    rtol=0,
                                    atol=1e-12,
                                )
                                total_oracles += 1
                        evidence["three_experts"] = EnsembleScorer(program).score(
                            {f: evidence[f] for f in LEAVES}
                        )
                        for method in METHODS:
                            e = evidence[method]
                            per_method[method].append(
                                {
                                    "session_id": session,
                                    "block": round_index // 5 + 1,
                                    "cycle": round_index % 5 + 1,
                                    "round_id": round_id,
                                    "target": truth,
                                    "prediction": e.predicted,
                                    "status": e.status,
                                    "probability_available": e.probability_available,
                                    "observed_counts": list(e.observed_event_counts),
                                    "evidence": e.record(),
                                }
                            )
                for method in METHODS:
                    records = per_method[method]
                    assert len(records) == 100 and len({x["round_id"] for x in records}) == 100
                    result = {
                        "seed": seed,
                        "metrics": metrics(records),
                        "per_recording": {
                            session: metrics([x for x in records if x["session_id"] == session])
                            for session in plan["test_sessions"]
                        },
                    }
                    by_method[method].append(result)
                    full_records[f"{pipeline}_K{dose}_{method}_s{seed}"] = records
            panel[str(dose)] = {
                m: aggregate(v if m != "xdawn_erp" else v[:1]) for m, v in by_method.items()
            }
        summaries[pipeline] = panel
    result = {
        "schema": "n2p3_brainsync_single_round_results/1",
        "plan": plan,
        "results": summaries,
        "parent_bindings": bindings,
        "numerical_checks": checks,
        "independent_complete_round_oracles": total_oracles,
        "analyzer_sha256": sha(__file__),
    }
    write(output / "results.json", result)
    write(output / "round_predictions.json", full_records)
    render(output, result)
    write(
        output / "verification.json",
        {
            "status": "passed",
            "new_model_fits": 0,
            "requested_rounds_per_seed": 100,
            "observed_scheduled_flashes": 900,
            "all_rounds_contain_nine_distinct_candidates": True,
            "rounds_accumulated_together": 1,
            "mathematical_checks": checks,
            "complete_round_independent_oracles": total_oracles,
            "parent_model_cells": len(bindings),
            "results_sha256": sha(output / "results.json"),
            "round_predictions_sha256": sha(output / "round_predictions.json"),
        },
    )
    return result


def render(output, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    names = {
        "nn_source": "NN source",
        "nn_scratch": "NN scratch",
        "xdawn_erp": "xDAWN",
        "three_experts": "source + LDA + xDAWN",
    }
    for ax, pipeline in zip(axes, ["basic", "reviewed"], strict=True):
        for method in METHODS:
            values = [
                100 * result["results"][pipeline][str(d)][method]["target_recognition_rate"]
                for d in DOSES
            ]
            ax.plot(DOSES, values, "o-", label=names[method])
        ax.axhline(100 / 9, color="gray", linestyle="--", label="Random guess: 1/9")
        ax.set(
            xlabel="Calibration recordings",
            ylabel="Target recognition (%)",
            title=pipeline,
            xticks=DOSES,
            ylim=(0, 100),
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Independent R1 cycles: 9 flashes per choice, 100 requested choices\nNo accumulation across the five cycles of a block"
    )
    fig.savefig(output / "single_round.png", dpi=170)
    plt.close(fig)
    tables = []
    for pipeline in ["basic", "reviewed"]:
        body = []
        for dose in DOSES:
            cells = []
            for method in METHODS:
                m = result["results"][pipeline][str(dose)][method]
                cells.append(
                    f"<td><b>{100 * m['target_recognition_rate']:.2f}%</b><br>{m['mean_hits']:.2f}/100<br><small>可判{m['mean_predicted_rounds']:.2f}/100</small></td>"
                )
            body.append(f"<tr><th>{dose}份 / {450 * dose}次排程</th>" + "".join(cells) + "</tr>")
        tables.append(
            f"<h2>{pipeline}</h2><table><tr><th>校准量</th>"
            + "".join(f"<th>{LABELS[m]}</th>" for m in METHODS)
            + "</tr>"
            + "".join(body)
            + "</table>"
        )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>单轮九选一目标识别率</title><style>body{font:15px/1.7 system-ui;background:#f5f8fc;color:#173047;margin:32px}main{max-width:1250px;margin:auto}table{border-collapse:collapse;width:100%;background:white;margin:16px 0 32px}td,th{border:1px solid #d5dfeb;padding:12px;text-align:center}th{background:#eaf1f8}small{color:#52677c}.note{padding:16px;border-left:4px solid #347bab;background:white}img{max-width:100%}a{color:#12629c}</style><main><h1>单轮九选一：目标识别率</h1><div class="note"><b>每9次闪烁独立识别一次目标数字，随机基线1/9＝11.11%。</b>两份测试录制各50轮，共100次识别。每个45闪烁block拆成5个独立计分轮；不累积这5轮的分数，也不做目标/非目标二分类。<br>比较1、3、5、10份校准，NN source、scratch、xDAWN与三混合四种方法。三混合沿用source＋波形LDA＋xDAWN，平均本轮三个九候选概率向量，各1/3。候选始终保留1–9，阈值和权重未在测试集选择。<br>已有模型和逐epoch分数直接复用，新增fit为0。拒判、缺候选和tie保留在全部100轮分母。神经及混合为三个目标seed平均；确定性xDAWN只计一份拟合。<br>输入仍沿用原整份录制的冻结连续滤波；这里限制的是每次聚合的9个epoch，属于回顾性R1评分。测试只有同一人的两份录制、目标7和9；100轮相关，三个seed不扩大独立样本量。</div><p><img src="single_round.png" alt="单轮九选一目标识别率"></p>"""
    page += (
        "".join(tables)
        + """<p><a href="results.json">逐seed与逐录制读数、来源</a> · <a href="round_predictions.json">每轮九候选概率与预测</a> · <a href="plan.json">用户确认的计分口径</a> · <a href="verification.json">数值核验</a> · <a href="../index.html">原45闪烁block报告</a></p></main></html>"""
    )
    (output / "index.html").write_text(page, encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run.resolve(), args.output.resolve())
    for dose in DOSES:
        print(
            dose,
            {
                m: round(
                    100 * result["results"]["basic"][str(dose)][m]["target_recognition_rate"], 3
                )
                for m in METHODS
            },
        )


if __name__ == "__main__":
    main()
