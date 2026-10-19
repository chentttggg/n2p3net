"""Export actual P1 state through its original source, without fit or score.

This script runs in its own process. Its supplied original capsule exclusively
owns all imports that admit historical model/noise packages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(plan_file, subject, output):
    config = json.loads(Path(plan_file).read_bytes())
    root = Path(config["original_capsule"])
    sys.path[:0] = [str(root/"src"), str(root)]
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits

    from experiments.audit_channel_noise_mechanism import BoundInput, public_calibration_authority
    from research import channel_noise_mechanism as original_owner
    from transfer import continuous_decision, noise_model, temporal_noise

    modules = {"channel": original_owner, "continuous": continuous_decision,
               "temporal": temporal_noise, "noise_contract": noise_model}
    source = {name: {"path": str(Path(module.__file__).resolve()), "sha256": sha(module.__file__)}
              for name, module in modules.items()}
    if source != config["original_loader_sources"]:
        raise ValueError("export did not execute the original strict loader sources")
    for path, expected in config["original_capsule_files"].items():
        if sha(path) != expected:
            raise ValueError("original frozen source member changed")
    descriptor = config["participants"][subject]
    for path, expected in descriptor["original_files"].items():
        if sha(path) != expected:
            raise ValueError("original producer/input file changed")
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("forbidden_fit_or_score")
        raise RuntimeError("original export forbids fit and score")
    for owner in (original_owner.FittedChannelNoise, continuous_decision.ContinuousDecisionModel,
                  temporal_noise.TemporalNoise):
        owner.fit = classmethod(forbidden)
    original_owner.FittedChannelNoise.score = forbidden
    continuous_decision.ContinuousDecisionModel.score = forbidden
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # BoundInput's public seam is a pathlib.Path: it resolves the manifest's
    # parent before resolving descriptor-relative payload files.  The plan
    # stores a serialized string, so coerce at this caller boundary while
    # preserving the original manifest bytes and digest.
    manifest_path = Path(descriptor["input_manifest"]["path"])
    inputs = BoundInput(manifest_path)
    payload = json.loads(manifest_path.read_bytes())
    entry = payload["participants"][0]
    with threadpool_limits(limits=1):
        authority = public_calibration_authority(payload["source_records"], inputs)
        # These original ingress constructions were already measured with two
        # native threads. Remaining export and all later scoring stay at one.
        with threadpool_limits(limits=2):
            model = continuous_decision.ContinuousDecisionModel.load(
                inputs.file(entry["response"]["source_model"]))
        records = []
        for row in entry["training"]:
            view = row["view"]
            samples, onsets, labels, design = inputs.arrays(row["arrays"], "samples", "onsets", "labels", "design")
            actual = authority[view["record_id"]]
            if (actual.view.record() != view or not np.array_equal(actual.labels, labels)
                    or actual.target_candidate != row["target_candidate"]
                    or not np.array_equal(design, continuous_decision._event_design(onsets, labels, len(samples), model.basis))):
                raise ValueError("export C differs from original MAT or original model design")
            records.append({"record_id": view["record_id"], "participant_key": view["participant_key"],
                "recording_id": view["source_sha256"], "read_support": view["raw_scope"],
                "fit_scope": "target_calibration", "source_sha256": view["source_sha256"],
                "input_sha256": noise_model.array_digest(samples, onsets, labels),
                "n_samples": len(samples), "original_view": view})
        response = original_owner.FrozenERPResponse.from_continuous_model(model,
            geometry=original_owner.NoiseGeometry(**entry["geometry"]),
            source_model_sha256=entry["response"]["source_model"]["sha256"],
            response_id=entry["response"]["response_id"], training_records=records)
        if json.loads(json.dumps(response.record())) != descriptor["response_record"]:
            raise ValueError("actual original response factor export changed its producer")
        held = entry["held"][0]
        y, designs = inputs.arrays(held["arrays"], "samples", "candidate_designs")
        view, membership = held["view"], np.asarray(held["view"]["decision"]["membership"])
        if (authority[view["record_id"]].view.record() != view
                or not np.array_equal(designs, np.asarray([continuous_decision._event_design(
                    np.asarray(view["onsets"]), membership[:, q], len(y), model.basis)
                    for q in range(len(payload["candidate_vocabulary"]))]))):
            raise ValueError("export held onsets or candidates differ from original owner")
        noises = {}
        for arm, file in descriptor["noise_files"].items():
            with threadpool_limits(limits=2):
                noise = original_owner.FittedChannelNoise.load(file["path"])
            if json.loads(json.dumps(noise.record())) != json.loads(Path(file["path"]).read_bytes())["state"]:
                raise ValueError("original strict load did not retain actual frozen noise")
            noises[arm] = {"state": noise.record(), "state_sha256": noise._integrity,
                          "original_file": file}
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        arrays = output/"response.npz"
        np.savez(arrays, mean=response.mean, coefficient_factor=response.coefficient_factor)
        result = {"schema": "p1_original_strict_state_export/1", "subject": subject,
            "plan_file_sha256": sha(plan_file), "export_script_sha256": sha(__file__),
            "loader_sources": source, "response": response.record(),
            "response_arrays_sha256": sha(arrays), "noises": noises,
            "fit_calls": calls, "score_calls": 0,
            "original_input_digest": descriptor["input_manifest"]["sha256"]}
        inputs.verify_unchanged()
        for path, expected in descriptor["original_files"].items():
            if sha(path) != expected:
                raise ValueError("original files changed during export")
        (output/"export.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export(args.plan, args.subject, args.output)


if __name__ == "__main__":
    main()
