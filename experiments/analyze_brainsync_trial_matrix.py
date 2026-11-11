"""Fixed per-flash binary diagnostics from the saved real-recording matrix.

This consumes immutable logits and their actual calibration class-risk records.
It fits no EEG model, input statistic, threshold, temperature or mixture weight.
The trial score pool is distinct from the earlier nine-candidate block pool.
"""

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
from scipy.special import expit  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from data.epochs import load_epoch_dataset  # noqa: E402
from research.contracts import semantic_sha256  # noqa: E402
from transfer.class_calibration import semantic_trial_logits  # noqa: E402

DOSES = (1, 3, 5, 10)
METHODS = ("nn_source", "nn_scratch", "xdawn_erp", "three_experts")
LEAVES = ("nn_source", "waveform_lda", "xdawn_erp")
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


def calibration_record(result):
    fit = result["selection"]["refit_history"]["probability_fit"]
    risk = fit["class_risk"]
    return {
        "origin": "target_calibration" if risk is not None else "target_generative",
        "pos_weight": risk["pos_weight"] if risk is not None else 1.0,
        "prior": risk["prior"] if risk is not None else fit["prior"],
        "temperature": 1.0,
        "fit_contract": fit,
    }


def common_prior_score(raw, record, *, prior):
    """A deterministic nominal score conversion, without calibration claims."""
    if not 0 < prior < 1:
        raise ValueError("reference class prior must be interior")
    llr, _ = semantic_trial_logits(raw, record)
    return expit(llr + np.log(prior / (1 - prior)))


def arithmetic_trial_pool(scores):
    """Require all three declared trial scores; missing leaves stay missing."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 3:
        raise ValueError("three-expert trial pool requires exactly three aligned score rows")
    available = np.isfinite(values).all(axis=0)
    result = np.full(values.shape[1], np.nan)
    result[available] = values[:, available].mean(axis=0)
    return result


def metrics(y, score):
    """Requested denominators are unchanged by missed/masked trial scores."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    if y.ndim != 1 or y.shape != score.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("binary metrics require one aligned requested roster with both classes")
    available = np.isfinite(score)
    predicted = score >= 0.5
    positive, negative = y == 1, y == 0
    tp = int(np.sum(available & positive & predicted))
    tn = int(np.sum(available & negative & ~predicted))
    fp = int(np.sum(available & negative & predicted))
    fn = int(np.sum(available & positive & ~predicted))
    missing_positive = int(np.sum(~available & positive))
    missing_negative = int(np.sum(~available & negative))
    p, n = int(positive.sum()), int(negative.sum())
    assert tp + fn + missing_positive == p and tn + fp + missing_negative == n
    return {
        "scheduled_trials": len(y),
        "target_trials": p,
        "nontarget_trials": n,
        "available_trials": int(available.sum()),
        "correct_trials": tp + tn,
        "accuracy": (tp + tn) / len(y),
        "target_recall": tp / p,
        "nontarget_recall": tn / n,
        "balanced_accuracy": 0.5 * (tp / p + tn / n),
        "coverage": float(available.mean()),
        "conditional_accuracy": (tp + tn) / int(available.sum()) if available.any() else None,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "missing_target": missing_positive,
        "missing_nontarget": missing_negative,
        "conditional_auc": float(roc_auc_score(y[available], score[available]))
        if len(np.unique(y[available])) == 2
        else None,
    }


def average(rows):
    keys = (
        "accuracy",
        "balanced_accuracy",
        "target_recall",
        "nontarget_recall",
        "coverage",
        "correct_trials",
        "available_trials",
        "TP",
        "TN",
        "FP",
        "FN",
        "missing_target",
        "missing_nontarget",
    )
    first = rows[0]
    assert all(r["scheduled_trials"] == first["scheduled_trials"] for r in rows)
    return {
        "scheduled_trials": first["scheduled_trials"],
        "target_trials": first["target_trials"],
        "nontarget_trials": first["nontarget_trials"],
        "optimization_repeats": len(rows),
        **{key: float(np.mean([r[key] for r in rows])) for key in keys},
        "accuracy_range": [min(r["accuracy"] for r in rows), max(r["accuracy"] for r in rows)],
        "per_seed": rows,
    }


def numerical_checks():
    y = np.tile(np.array([1] + [0] * 8), 100)
    baseline = metrics(y, np.zeros(900))
    assert baseline["correct_trials"] == 800 and baseline["balanced_accuracy"] == 0.5
    absent = metrics(y, np.full(900, np.nan))
    assert absent["accuracy"] == 0 and absent["balanced_accuracy"] == 0 and absent["coverage"] == 0
    sample = np.array([[0.9, 0.1, np.nan], [0.8, 0.2, 0.9], [0.7, 0.3, 0.8]])
    pooled = arithmetic_trial_pool(sample)
    assert np.allclose(pooled[:2], [0.8, 0.2]) and np.isnan(pooled[2])
    assert metrics(y, np.ones(900))["target_recall"] == 1
    return {
        "all_nontarget_baseline": baseline,
        "all_missing": absent,
        "strict_three_leaf_missingness": True,
        "independent_score_threshold_cases_passed": True,
    }


def analyze(run, output):
    plan = read(run / "plan.json")
    jobs = read(run / "jobs.json")
    effective_bindings = read(run / "analysis/result_bindings.json")
    recovery = read(run / "recovery_map.json")
    frozen_files = read(run / "execution_source/source.manifest.json")["files"]
    assert (
        sha(ROOT / "src/transfer/class_calibration.py")
        == frozen_files["src/transfer/class_calibration.py"]
    )
    assert sha(ROOT / "src/transfer/calibrated.py") == frozen_files["src/transfer/calibrated.py"]
    index = {
        (j["pipeline"], j["dose"], j["family"], j["seed"]): j
        for j in jobs
        if j["backend"] == "current"
    }
    rows = {r["session_id"]: r for r in plan["rows"]}
    declarations = {
        "schema": "n2p3_brainsync_per_trial_comparison/1",
        "methods": list(METHODS),
        "three_expert_components": list(LEAVES),
        "calibration_record_counts": list(DOSES),
        "test_sessions": plan["test_sessions"],
        "source_plan_sha256": sha(run / "plan.json"),
        "units": "one scheduled flash; binary target versus nontarget",
        "new_fits": 0,
        "nominal_operating_points": {"natural_prior": 1 / 9, "equal_class_prior": 0.5},
        "threshold": 0.5,
        "threshold_comparison": "greater_than_or_equal",
        "temperature": 1.0,
        "mixture": "arithmetic average of same-trial nominal scores; all three leaves required",
        "missing_scores": "no prediction; retain requested class and total denominators",
        "score_interpretation": "fixed score transformations; no claim of independently calibrated posterior",
        "legacy_NN_loss": "batch_class_mass; the certified-posterior API is not invoked",
        "selection": "no held-data threshold, prior, temperature or weight fitting",
    }
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", declarations)
    checks = numerical_checks()
    results, bindings, npz_predictions = {}, {}, {}
    cached = {}
    for pipeline in ("basic", "reviewed"):
        truths, event_ids, session_ids = [], [], []
        for name in plan["test_sessions"]:
            entry = rows[name]["paths"][pipeline]
            cache = run / entry["path"]
            assert sha(cache) == entry["sha256"]
            ds = load_epoch_dataset(cache, require_labels=True, validation="attested")
            y = (ds.event_timeline.candidate_ids.astype(str) == str(rows[name]["target"])).astype(
                int
            )
            observed = ds.event_timeline.evidence_indices >= 0
            assert np.array_equal(ds.y[ds.event_timeline.evidence_indices[observed]], y[observed])
            assert len(y) == 450 and y.sum() == 50
            cached[pipeline, name] = ds
            truths.append(y)
            event_ids.extend(ds.event_timeline.event_ids.astype(str))
            session_ids.extend([name] * 450)
        y = np.concatenate(truths)
        npz_predictions[pipeline + "_truth"] = y
        npz_predictions[pipeline + "_event_ids"] = np.asarray(event_ids)
        npz_predictions[pipeline + "_session_ids"] = np.asarray(session_ids)
        panel = {}
        for dose in DOSES:
            per_method = {
                mode: {m: [] for m in METHODS} for mode in ("natural_prior", "equal_class_prior")
            }
            for seed in plan["target_seeds"]:
                leaf_scores = {mode: {} for mode in per_method}
                for family in ("nn_source", "nn_scratch", "xdawn_erp", "waveform_lda"):
                    actual_seed = seed if family.startswith("nn_") else plan["target_seeds"][0]
                    job = index[pipeline, dose, family, actual_seed]
                    result_file = run / "results" / job["id"] / "result.json"
                    if job["id"] in recovery:
                        result_file = run / recovery[job["id"]]["local_result"]
                    assert sha(result_file) == effective_bindings[job["id"]]
                    result = read(result_file)
                    assert result["status"] == "completed"
                    assert set(result["calibration_sessions"]) == set(
                        plan["calibration_order"][:dose]
                    )
                    assert not set(result["calibration_sessions"]) & set(plan["test_sessions"])
                    record = calibration_record(result)
                    raw_parts = []
                    prediction_hashes = {}
                    for name in plan["test_sessions"]:
                        ds = cached[pipeline, name]
                        pred_path = result_file.parent / f"{name}_predictions.npz"
                        prediction_hashes[name] = sha(pred_path)
                        with np.load(pred_path, allow_pickle=False) as archive:
                            assert np.array_equal(archive["event_ids"], ds.event_timeline.event_ids)
                            assert np.array_equal(
                                archive["evidence_indices"], ds.event_timeline.evidence_indices
                            )
                            assert len(archive["logits"]) == ds.n_epochs
                            raw = np.full(450, np.nan)
                            present = ds.event_timeline.evidence_indices >= 0
                            raw[present] = archive["logits"][
                                ds.event_timeline.evidence_indices[present]
                            ]
                        raw_parts.append(raw)
                    raw = np.concatenate(raw_parts)
                    numerical_llr, _ = semantic_trial_logits(raw, record)
                    independent_llr = (
                        raw
                        - np.log(record["pos_weight"])
                        - np.log(record["prior"] / (1 - record["prior"]))
                    )
                    assert np.allclose(
                        numerical_llr, independent_llr, rtol=0, atol=1e-12, equal_nan=True
                    )
                    bindings[job["id"]] = {
                        "result_file": str(result_file),
                        "result_sha256": sha(result_file),
                        "model_sha256": result["model_sha256"],
                        "prediction_sha256": prediction_hashes,
                        "class_counts": record["fit_contract"]["class_counts"],
                        "pos_weight": record["pos_weight"],
                        "training_prior": record["prior"],
                        "class_risk": record["fit_contract"]["class_risk"],
                    }
                    for mode, prior in declarations["nominal_operating_points"].items():
                        leaf_scores[mode][family] = common_prior_score(raw, record, prior=prior)
                for mode in per_method:
                    leaf_scores[mode]["three_experts"] = arithmetic_trial_pool(
                        [leaf_scores[mode][f] for f in LEAVES]
                    )
                    for method in METHODS:
                        score = leaf_scores[mode][method]
                        key = f"{pipeline}_K{dose}_{method}_s{seed}_{mode}"
                        npz_predictions[key] = score
                        m = metrics(y, score)
                        m["seed"] = seed
                        m["per_recording"] = {
                            name: metrics(
                                y[i * 450 : (i + 1) * 450], score[i * 450 : (i + 1) * 450]
                            )
                            for i, name in enumerate(plan["test_sessions"])
                        }
                        if np.isfinite(score).all():
                            from sklearn.metrics import accuracy_score, balanced_accuracy_score

                            assert abs(m["accuracy"] - accuracy_score(y, score >= 0.5)) < 1e-14
                            assert (
                                abs(
                                    m["balanced_accuracy"]
                                    - balanced_accuracy_score(y, score >= 0.5)
                                )
                                < 1e-14
                            )
                        per_method[mode][method].append(m)
            panel[str(dose)] = {
                mode: {
                    method: average(values if method != "xdawn_erp" else values[:1])
                    for method, values in methods.items()
                }
                for mode, methods in per_method.items()
            }
        results[pipeline] = panel
    np.savez_compressed(output / "trial_scores.npz", **npz_predictions)
    record = {
        "schema": "n2p3_brainsync_per_trial_results/1",
        "plan": declarations,
        "results": results,
        "all_nontarget_baseline": {
            "accuracy": 8 / 9,
            "balanced_accuracy": 0.5,
            "target_recall": 0.0,
            "scheduled_trials": 900,
            "target_trials": 100,
            "nontarget_trials": 800,
        },
        "source_bindings": bindings,
        "numeric_checks": checks,
        "source_sha256": {
            "analyzer": sha(__file__),
            "class_calibration": sha(ROOT / "src/transfer/class_calibration.py"),
        },
        "scores_sha256": sha(output / "trial_scores.npz"),
    }
    write(output / "results.json", record)
    render(output, record)
    write(
        output / "verification.json",
        {
            "status": "passed",
            "new_model_fits": 0,
            "prior_oracle_atol": 1e-12,
            "sklearn_metrics_oracle": True,
            "scheduled_trials": 900,
            "target_trials": 100,
            "nontarget_trials": 800,
            "actual_parent_models": len(bindings),
            "source_bindings_sha256": semantic_sha256(bindings),
            "results_sha256": sha(output / "results.json"),
            "original_block_report_unchanged": True,
        },
    )
    return record


def render(output, record):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    names = {
        "nn_source": "NN source",
        "nn_scratch": "NN scratch",
        "xdawn_erp": "xDAWN",
        "three_experts": "source + LDA + xDAWN",
    }
    for ax, metric, title in zip(
        axes,
        ["accuracy", "balanced_accuracy"],
        ["Per-trial accuracy", "Per-trial balanced accuracy"],
        strict=True,
    ):
        for method in METHODS:
            values = [
                100 * record["results"]["basic"][str(d)]["natural_prior"][method][metric]
                for d in DOSES
            ]
            ax.plot(DOSES, values, "o-", label=names[method])
        ax.axhline(
            100 * (8 / 9 if metric == "accuracy" else 0.5),
            linestyle="--",
            color="gray",
            label="Always nontarget",
        )
        ax.set(xlabel="Calibration recordings", ylabel="%", title=title, xticks=DOSES)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("900 scheduled flashes; 100 target / 800 nontarget; fixed nominal prior 1/9")
    fig.savefig(output / "per_trial.png", dpi=170)
    plt.close(fig)
    sections = []
    for pipeline in ["basic", "reviewed"]:
        for mode in ["natural_prior", "equal_class_prior"]:
            rows = []
            for dose in DOSES:
                for method in METHODS:
                    m = record["results"][pipeline][str(dose)][mode][method]
                    rows.append(
                        f"<tr><td>{dose}</td><td>{LABELS[method]}</td><td><b>{100 * m['accuracy']:.2f}%</b></td>"
                        f"<td>{100 * m['balanced_accuracy']:.2f}%</td><td>{100 * m['target_recall']:.2f}%</td>"
                        f"<td>{100 * m['nontarget_recall']:.2f}%</td><td>{100 * m['coverage']:.2f}%</td>"
                        f"<td>{m['correct_trials']:.2f}/900</td></tr>"
                    )
            sections.append(
                f"<h2>{pipeline} · {'名义先验1/9，主表' if mode == 'natural_prior' else '等类先验1/2，固定阈值敏感性'}</h2>"
                "<table><tr><th>校准份数</th><th>方法</th><th>正确率</th><th>平衡正确率</th><th>目标召回</th><th>非目标召回</th><th>覆盖率</th><th>正确/全部请求</th></tr>"
                + "".join(rows)
                + "</table>"
            )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>BrainSync per-trial四方法对照</title><style>body{font:15px/1.7 system-ui;color:#17293d;background:#f5f8fc;margin:32px}main{max-width:1250px;margin:auto}table{border-collapse:collapse;width:100%;background:white;margin-bottom:32px}th,td{padding:8px 10px;border:1px solid #d7dfeb;text-align:center}th{background:#e9f0f8}img{max-width:100%}.note{padding:16px;background:white;border-left:4px solid #3278ad}a{color:#135d9e}</style><main><h1>Per-trial：NN source、scratch、xDAWN、三混合</h1>"""
    page += """<div class="note">本次trial指一次闪烁后的目标/非目标二分类。沿用固定的10月30日两份测试录制，共900次排程、100次目标、800次非目标。全部判非目标也有88.89%正确率，平衡正确率只有50%。<br>校准量仅比较1、3、5、10份。三混合仍为NN source＋波形LDA＋xDAWN；单独LDA不作为第五种比较方法。模型、校准范围和原预测均复用，不重新训练。<br>按保存的C内类别代价与先验，对原始分数做固定转换。主表使用协议给定1/9名义先验，目标分数≥0.5判target；另列1/2等类先验口径。阈值、温度和1/3权重均未在测试集选择。NN旧配方为batch_class_mass，结果是固定得分诊断，不宣称已得到独立校准的后验概率。<br>三混合在每个trial合并目标得分，与之前在45次闪烁后合并九候选概率的程序不同。缺任一所需专家分数则该trial拒判，仍保留在900与对应类别分母。神经及混合按三个目标seed平均，xDAWN为同一确定性拟合。测试trial相关，仍是单人两份录制的回顾性开发结果。</div><p><img src="per_trial.png" alt="逐trial正确率与平衡正确率"></p>"""
    page += (
        "".join(sections)
        + """<p><a href="results.json">分seed、混淆矩阵和来源</a> · <a href="trial_scores.npz">逐trial分数</a> · <a href="plan.json">固定计算口径</a> · <a href="verification.json">数值核验</a> · <a href="../index.html">原九选一block报告</a></p></main></html>"""
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
                    100 * result["results"]["basic"][str(dose)]["natural_prior"][m]["accuracy"], 3
                )
                for m in METHODS
            },
        )


if __name__ == "__main__":
    main()
