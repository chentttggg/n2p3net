"""Numeric frozen trial predictors shared by calibration and research replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from baselines.structured_erp import StructuredERP
from models.n2p3net import N2P3ArchitectureConfig, N2P3Net
from research.contracts import semantic_sha256
from transfer.binary import plain_trunk_configuration, target_shrinkage_features
from transfer.calibration_candidates import FittedCalibrationCandidate
from transfer.checkpoint import checkpoint_input_stats, predict_n2p3_checkpoint
from transfer.noise_model import freeze_metadata

FROZEN_TRIAL_SCHEMA = "n2p3_frozen_trial_predictor/1"


class FrozenTrialPredictor:
    """An inference-only predictor decoded once from explicit numeric state."""

    def __init__(self, record, state):
        self._record = freeze_metadata(record)
        self._state = {name: value.detach().cpu().clone() for name, value in state.items()}
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
        if any(not bool(torch.isfinite(value).all()) for value in self._state.values()):
            raise ValueError("frozen trial numeric state contains nonfinite values")
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
        return cls(record, fitted.state)

    def predict_logit(self, X, trial_channel_mask=None, *, device="cpu"):
        if self.record["kind"] == "n2p3":
            return predict_n2p3_checkpoint(self._model, X, trial_channel_mask=trial_channel_mask,
                input_stats=self._input_statistics,
                device=torch.device(device))
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

    def save(self, path: str | Path):
        torch.save({"schema": FROZEN_TRIAL_SCHEMA,
                    "record": json.loads(json.dumps(self._record)), "state": self._state}, path)
        return Path(path)

    def digest(self):
        state = {}
        for key, tensor in self._state.items():
            value = tensor.contiguous()
            state[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                          "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        return semantic_sha256({"record": self.record, "state": state})

    @classmethod
    def load(cls, path: str | Path):
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if payload["schema"] != FROZEN_TRIAL_SCHEMA:
            raise ValueError("unsupported frozen trial predictor schema")
        return cls(payload["record"], payload["state"])
