"""Post-matrix mechanism diagnostics using only each fold-0 training C scope.

No held samples or labels are used by these temporal refits. All alternatives
are reported; none changes the already sealed matrix or chooses a new default.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze_calibration_study import (  # noqa: E402
    acf_records,
    event_design,
    import_frozen,
    rebuild_scopes,
    sealed,
)


def diagnostics(noise, records):
    innovations = []
    for x in records:
        z = np.column_stack([np.ones(len(x)), np.linspace(-1, 1, len(x))])
        q, _ = np.linalg.qr(noise.whiten(z), mode="reduced")
        e = noise.whiten(x)
        innovations.append(e-q@(q.T@e))
    energies = sum(np.sum(e**2, axis=0) for e in innovations)
    acfs = acf_records(innovations)
    numerator = sum(np.sum(e[1:]*e[:-1]) for e in innovations)
    return {"mean_acf1": float(np.mean(acfs["1"])), "acf1_per_channel": acfs["1"],
            "energy_weighted_acf1": float(numerator/energies.sum()),
            "channel_energy_share": (energies/energies.sum()).tolist(),
            "largest_channel_energy_share": float(np.max(energies/energies.sum())),
            "innovation_mse": float(energies.sum()/sum(e.size for e in innovations)),
            "coefficients": noise.coefficients.tolist(), "pole_radius": noise.pole_radius,
            "startup_condition": noise.startup_condition}


def main():
    run = ROOT/"experiments/runs/operator_research_20261004/calibration_groups_v2"
    output = Path(__file__).with_name("fit_step_diagnostics.json")
    plan = json.loads((run/"plan.json").read_bytes())
    source_identity = json.loads((run/"source.json").read_bytes())
    module = import_frozen(run)
    from transfer.temporal_noise import TemporalNoise

    results = []
    with threadpool_limits(limits=2):
        for row in plan["records"]:
            subject = row["subject"]
            scope = json.loads((run/subject/"scopes.json").read_bytes())
            groups, _ = rebuild_scopes(row, plan, scope)
            training = groups[1:]  # Fixed fold0 training scopes, no held data in refits.
            directory = run/subject/"fold0_raw_ar8"
            record = json.loads((directory/"record.json").read_bytes())
            files = sealed(record, directory)
            model = module.OperatorGLS.load(io.BytesIO(files["model"]))
            white = TemporalNoise.from_parameters([], innovation_variance=1., reference_id="initial_OLS_reproduction")
            designs = [event_design(group, model.basis) for group in training]
            samples = [group["samples"] for group in training]
            D = np.vstack([module.project(white, d) for d in designs])
            Y = np.vstack([module.project(white, x) for x in samples])
            h_initial = np.linalg.solve(D.T@D+plan["ridge"]*np.eye(D.shape[1]), D.T@Y)
            initial, final = [], []
            for d, x in zip(designs, samples, strict=True):
                initial.append(module.project(white, x-d@h_initial))
                z = np.column_stack([np.ones(len(x)), np.linspace(-1, 1, len(x))])
                raw_residual = x-d@model.mean
                beta = np.linalg.lstsq(model.noise.whiten(z), model.noise.whiten(raw_residual), rcond=None)[0]
                final.append(raw_residual-z@beta)
            ids = [group["record_id"] for group in training]
            evidence = {"subject": subject, "fold": 0, "fit_labels": 600,
                "model_sha256": hashlib.sha256(files["model"]).hexdigest(), "fit_record_ids": ids,
                "frozen_original_on_initial_residual": diagnostics(model.noise, initial),
                "frozen_original_on_final_residual": diagnostics(model.noise, final),
                "fixed_refits": {}}
            channel_root = np.linalg.cholesky(model.channel_covariance)
            for stage, records in (("initial_OLS", initial), ("final_GLS", final)):
                scale = np.sqrt(sum(np.sum(x*x, axis=0) for x in records)/sum(len(x) for x in records))
                weights = {"none": records, "diagonal_std": [x/scale for x in records],
                           "frozen_channel_precision": [solve_triangular(channel_root, x.T, lower=True).T for x in records]}
                evidence["fixed_refits"][stage] = {}
                for name, fitting_records in weights.items():
                    fitted = TemporalNoise.fit(fitting_records, order=8, record_ids=ids, fit_scope="target_calibration")
                    evidence["fixed_refits"][stage][name] = diagnostics(fitted, records)
            endpoint_energy = []
            for h in (h_initial, model.mean):
                kernel = model.basis@h[model.basis.shape[1]:]
                endpoint_energy.append(float(np.linalg.norm(kernel[[0, -1]])/np.linalg.norm(kernel)))
            evidence["target_kernel_endpoint_norm_over_waveform_norm_initial_final"] = endpoint_energy
            results.append(evidence)
            print(json.dumps({"subject": subject,
                "initial_saved_AR_acf": evidence["frozen_original_on_initial_residual"]["mean_acf1"],
                "final_saved_AR_acf": evidence["frozen_original_on_final_residual"]["mean_acf1"],
                "final_refit_none_acf": evidence["fixed_refits"]["final_GLS"]["none"]["mean_acf1"],
                "final_refit_precision_acf": evidence["fixed_refits"]["final_GLS"]["frozen_channel_precision"]["mean_acf1"]}), flush=True)
    output.write_text(json.dumps({"schema": "n2p3_postmatrix_training_residual_diagnostic/1",
        "scope": "fixed_fold0_training600_labels_only_eight_subjects_no_held_refit", "order": 8,
        "source_run": str(run), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": source_identity["digest"],
        "analysis_dependency_sha256": hashlib.sha256(Path(__file__).parent.parent.joinpath("analyze_calibration_study.py").read_bytes()).hexdigest(),
        "interpretation": "posthoc_mechanism_readout_not_confirmatory_performance_or_parameter_selection",
        "records": results}, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
