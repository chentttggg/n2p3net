"""Numeric frozen trial predictors shared by calibration and research replay."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from baselines.structured_erp import StructuredERP
from models.n2p3net import N2P3ArchitectureConfig, N2P3Net
from research.contracts import canonical_json_bytes, semantic_sha256
from research.frozen_metadata import freeze_metadata, plain_metadata
from transfer.binary import plain_trunk_configuration, target_shrinkage_features
from transfer.calibration_candidates import FittedCalibrationCandidate
from transfer.checkpoint import (
    checkpoint_input_stats,
    checkpoint_scores_to_llr,
    predict_n2p3_checkpoint,
)
from transfer.class_calibration import TrialProbabilityTransform

FROZEN_TRIAL_SCHEMA = "n2p3_frozen_trial_predictor/2"


class FrozenTrialPredictor:
    """An inference-only predictor decoded once from explicit numeric state."""

    def __init__(self, record, state):
        self._record = freeze_metadata(record)
        self._state = {name: value.detach().cpu().clone() for name, value in state.items()}
        self._probability_transform = None
        if any(not bool(torch.isfinite(value).all()) for value in self._state.values()):
            raise ValueError("frozen trial numeric state contains nonfinite values")
        if "class_calibration" in record:
            calibration = record["class_calibration"]
            expected_kinds = {"source_mixture": {"n2p3"}, "target_calibration": {"n2p3"},
                              "target_generative": {"structured_erp", "waveform_lda"}}
            score_kinds = {"source_mixture": "source_weighted_CE_logit", "target_calibration": "balanced_class_cost_logit",
                           "target_generative": "generative_posterior_logit"}
            if (record["kind"] not in expected_kinds.get(calibration["origin"], set())
                    or calibration["raw_score_kind"] != score_kinds.get(calibration["origin"])):
                raise ValueError("probability transform differs from its actual numerical producer kind")
            probability_calibration = calibration
            if calibration["origin"] == "source_mixture":
                # Retain the original external scalar/type gates while sharing
                # only the complete source contract snapshot owned by metadata.
                probability_calibration = {**calibration, "fit_contract": {
                    **calibration["fit_contract"], "training_contract": self._record[
                        "class_calibration"]["fit_contract"]["training_contract"]}}
            self._probability_transform = TrialProbabilityTransform.from_record(probability_calibration)
            fit = record["class_calibration"]["fit_contract"]
            if fit.get("frozen_state_digest") != _numeric_state_digest(self._state):
                raise ValueError("frozen probability contract differs from the actual numeric state")
            if record["class_calibration"]["origin"] == "source_mixture" and (
                record["class_calibration"]["pos_weight"] != record["source_calibration"]["pos_weight"]
                or record["class_calibration"]["prior"] != record["source_calibration"]["train_prior"]
                or record["class_calibration"]["temperature"] != record["source_calibration"]["temperature"]):
                raise ValueError("source probability contract differs from its actual source calibration")
        if "selection_epoch_capability" in record:
            from transfer.ensemble_views import SelectionEpochProfile

            capability = record["selection_epoch_capability"]
            profile = SelectionEpochProfile.from_record(capability["profile"])
            if (capability["profile_digest"] != profile.digest or len(profile.channel_names) != record["n_channels"]
                    or profile.preprocessing.n_times != record["n_times"] or profile.preprocessing.sfreq != record["sfreq"]
                    or profile.preprocessing.tmin_ms / 1000 != record["tmin_s"]):
                raise ValueError("source inference capability differs from its actual epoch programme")
            if "class_calibration" in record and record["class_calibration"]["origin"] == "source_mixture":
                from data.channel import canonical_channel_name

                source_input = record["class_calibration"]["fit_contract"]["training_contract"]["preprocessing"]
                # ``freeze_metadata`` canonicalizes JSON arrays to tuples when
                # a fit binding is attached.  Compare the serialized
                # contract, which treats list/tuple as the same JSON array,
                # while retaining every value and rejecting real changes.
                if (semantic_sha256(source_input["epoch"]) != semantic_sha256(profile.preprocessing.record())
                        or tuple(map(canonical_channel_name, source_input["channel_names"])) != tuple(map(canonical_channel_name, profile.channel_names))
                        or profile.reference_channel is not None and source_input["source_reference"] != "measured_channel:" + profile.reference_channel):
                    raise ValueError("source inference capability differs from its actual trained input contract")
        kind = record["kind"]
        self._model = None
        if kind == "structured_erp":
            self._model = StructuredERP.from_state(record["model"], {
                key.removeprefix("erp."): value.numpy() for key, value in self._state.items()
                if key.startswith("erp.")})
        elif kind == "n2p3":
            model = N2P3Net(record["n_channels"], n_times=record["n_times"], sfreq=record["sfreq"],
                tmin_s=record["tmin_s"], pooling_mode=record["pooling_mode"],
                **N2P3ArchitectureConfig(**record["configuration"]).model_kwargs())
            if semantic_sha256(model.architecture_record()) != semantic_sha256(record["architecture"]):
                raise ValueError("frozen trial architecture does not reconstruct exactly")
            model.load_state_dict({key.removeprefix("trunk."): value for key, value in self._state.items()
                                   if key.startswith("trunk.")})
            model.eval()
            self._model = model
        elif kind != "waveform_lda":
            raise ValueError("unsupported frozen trial predictor kind")
        required = ({"input_mean", "input_std"} if kind == "n2p3" else {"waveform_channels"})
        if kind == "waveform_lda":
            required |= {"feature_mean", "feature_scale", "coefficient", "intercept"}
        if not required <= self._state.keys():
            raise ValueError("frozen trial numeric state is incomplete")
        self._input_statistics = (checkpoint_input_stats({
            name: self._state[name].numpy() for name in ("input_mean", "input_std")
        }, record["n_channels"]) if kind == "n2p3" else None)
        if kind == "waveform_lda" and (not bool((self._state["feature_scale"] > 0).all())
                or self._state["feature_mean"].shape != self._state["coefficient"].shape
                or self._state["feature_scale"].shape != self._state["coefficient"].shape):
            raise ValueError("frozen waveform state has invalid feature shape or scale")

    @property
    def record(self):
        return self._record

    @property
    def posterior_available(self):
        calibration = self.record.get("class_calibration")
        return calibration is not None and (calibration["origin"] != "target_calibration"
            or calibration["fit_contract"]["class_risk"]["loss_reduction"] == "calibration_balanced_fixed_mass")

    def with_fit_binding(self, binding):
        return type(self)({**self._record, "ensemble_fit_binding": binding}, self._state)

    @classmethod
    def from_fitted(cls, fitted: FittedCalibrationCandidate):
        if fitted.epoch_quality is not None or fitted.channel_quality_availability:
            raise ValueError("trial snapshot requires an explicit upstream frozen quality view")
        if fitted.erp is not None:
            record = {"kind": "structured_erp", "model": fitted.erp.export_state()[0]}
        elif fitted.adapter is None:
            record = {"kind": "waveform_lda", "sfreq": fitted.sfreq,
                      "classifier_dtype": fitted.lda.steps[1][1].coef_.dtype.str}
        else:
            if fitted.lda is not None or fitted.adapter.head is not None:
                raise ValueError("neural snapshot requires the declared classifier-fine/full-fine head")
            trunk = fitted.adapter.trunk
            record = {"kind": "n2p3", "configuration": asdict(plain_trunk_configuration(trunk)),
                      "architecture": trunk.architecture_record(), "n_channels": trunk.n_channels,
                      "n_times": trunk.n_times, "sfreq": trunk.sfreq, "tmin_s": trunk.tmin_s,
                      "pooling_mode": trunk.pooling_mode}
        fit = fitted.fit_history["probability_fit"]
        risk = fit["class_risk"]
        calibration = {"origin": "target_calibration" if fitted.adapter is not None else "target_generative",
            "pos_weight": 1. if risk is None else risk["pos_weight"],
            "prior": fit["prior"] if risk is None else risk["prior"], "temperature": 1.,
            "raw_score_kind": "balanced_class_cost_logit" if risk is not None else "generative_posterior_logit",
            "fit_contract": {**fit, "frozen_state_digest": _numeric_state_digest(fitted.state)}}
        record["class_calibration"] = calibration
        return cls(record, fitted.state)

    @classmethod
    def from_source(cls, prepared, subject):
        """K0 freezes source weights, BN and input statistics without a target fit."""
        trunk = prepared.trunk_for_target(subject)
        if type(trunk) is not N2P3Net or trunk.splitbn_config is not None:
            raise ValueError("frozen source-only expert requires the plain declared N2P3 architecture")
        source_calibration = prepared.payload["source_calibration"]
        checkpoint_scores_to_llr({"source_calibration": source_calibration}, np.zeros(1))
        record = {"kind": "n2p3", "configuration": asdict(plain_trunk_configuration(trunk)),
                  "architecture": trunk.architecture_record(), "n_channels": trunk.n_channels,
                  "n_times": trunk.n_times, "sfreq": trunk.sfreq, "tmin_s": trunk.tmin_s,
                  "pooling_mode": trunk.pooling_mode, "source_calibration": source_calibration}
        if prepared.selection_epoch_capability is not None:
            record["selection_epoch_capability"] = prepared.selection_epoch_capability
        state = {f"trunk.{key}": value for key, value in trunk.state_dict().items()}
        state.update(input_mean=torch.as_tensor(prepared.input_statistics[0]),
                     input_std=torch.as_tensor(prepared.input_statistics[1]))
        if prepared.payload["training_contract"]["optimizer"].get("config", {}).get("pos_weight") is not None:
            record["class_calibration"] = {"origin": "source_mixture", "pos_weight": source_calibration["pos_weight"],
                "prior": source_calibration["train_prior"], "temperature": source_calibration["temperature"],
                "raw_score_kind": "source_weighted_CE_logit",
                "fit_contract": {"training_contract": prepared.payload["training_contract"],
                    "training_contract_digest": prepared.payload["training_contract_digest"],
                    "frozen_state_digest": _numeric_state_digest(state)}}
        return cls(record, state)

    def predict_raw_logit(self, X, trial_channel_mask=None, *, device="cpu"):
        if self.record["kind"] == "n2p3":
            logits = predict_n2p3_checkpoint(self._model, X, trial_channel_mask=trial_channel_mask,
                input_stats=self._input_statistics,
                device=torch.device(device))
            return logits
        channels = self._state["waveform_channels"].numpy()
        if trial_channel_mask is not None and not np.asarray(trial_channel_mask)[:, channels].all():
            raise ValueError("frozen classic expert requires all selected physical channels")
        values = np.asarray(X)[:, channels]
        if self._model is not None:
            return self._model.predict_logit(values)
        features = target_shrinkage_features(values, sfreq=self.record["sfreq"])
        features -= self._state["feature_mean"].numpy().astype(features.dtype)
        features /= self._state["feature_scale"].numpy().astype(features.dtype)
        dtype = np.dtype(self.record["classifier_dtype"])
        return (features @ self._state["coefficient"].numpy().astype(dtype)[:, None]
                + self._state["intercept"].numpy().astype(dtype)).ravel().astype(np.float64)

    def predict_logit(self, X, trial_channel_mask=None, *, device="cpu"):
        """Candidate evidence is an LLR; legacy packages retain their original score."""
        raw = self.predict_raw_logit(X, trial_channel_mask, device=device)
        if "class_calibration" in self.record:
            return self._probability_transform.apply(raw)[0]
        return checkpoint_scores_to_llr(self.record, raw)[0] if "source_calibration" in self.record else raw

    def probability_scores(self, X, trial_channel_mask=None, *, device="cpu"):
        if "class_calibration" not in self.record:
            raise ValueError("legacy trial score has no actual posterior contract")
        calibration = self.record["class_calibration"]
        if (calibration["origin"] == "target_calibration"
                and calibration["fit_contract"]["class_risk"]["loss_reduction"] != "calibration_balanced_fixed_mass"):
            raise ValueError("batch class-mass reduction cannot certify the declared C posterior programme")
        raw = self.predict_raw_logit(X, trial_channel_mask, device=device)
        llr, posterior = self._probability_transform.apply(raw)
        from scipy.special import expit

        return {"raw_logits": raw, "llr": llr, "posterior_logits": posterior, "probability": expit(posterior),
                "semantics": self.record["class_calibration"]}

    def save(self, path: str | Path):
        from research.source_execution import save_source_checkpoint

        save_source_checkpoint(path, {
            "schema": (FROZEN_TRIAL_SCHEMA if "class_calibration" in self.record
                       else "n2p3_frozen_trial_predictor/1"),
            "record": plain_metadata(self._record), "state": self._state,
        })
        return Path(path)

    def digest(self):
        state = {}
        for key, tensor in self._state.items():
            value = tensor.contiguous()
            state[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                          "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        # Metadata belongs to the immutable owner; numerical tensors remain
        # mutable private storage and must be hashed afresh, including numpy views.
        digest = hashlib.sha256()
        digest.update(b'{"record":')
        digest.update(canonical_json_bytes(self.record))
        digest.update(b',"state":')
        digest.update(canonical_json_bytes(state))
        digest.update(b'}')
        return digest.hexdigest()

    @classmethod
    def load(cls, path: str | Path):
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if payload["schema"] not in {FROZEN_TRIAL_SCHEMA, "n2p3_frozen_trial_predictor/1"}:
            raise ValueError("unsupported frozen trial predictor schema")
        if payload["schema"] == FROZEN_TRIAL_SCHEMA and "class_calibration" not in payload["record"]:
            raise ValueError("frozen probability predictor requires its actual fit calibration")
        if payload["schema"] == "n2p3_frozen_trial_predictor/1" and "class_calibration" in payload["record"]:
            raise ValueError("legacy predictor cannot claim the new actual posterior contract")
        return cls(payload["record"], payload["state"])


def _numeric_state_digest(state):
    return semantic_sha256({name: {"dtype": str(value.dtype), "shape": list(value.shape),
        "sha256": hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        for name, value in state.items()})
