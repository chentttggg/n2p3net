"""Actual calibration class risk and explicit posterior versus evidence scores."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from research.contracts import semantic_sha256

TRIAL_LOSS_REDUCTIONS = {"batch_class_mass", "calibration_balanced_fixed_mass"}


class CalibrationBalancedLoss(nn.Module):
    """A uniform-row minibatch estimates the fixed full-C weighted CE risk."""

    def __init__(self, *, pos_weight, class_mass, device):
        super().__init__()
        self.register_buffer("weight", torch.tensor([1., pos_weight], device=device))
        self.class_mass = class_mass

    def forward(self, logits, labels):
        return torch.nn.functional.cross_entropy(logits, labels, weight=self.weight, reduction="sum") / (
            len(labels) * self.class_mass)


def calibration_trial_loss(labels, *, reduction, device):
    if reduction not in TRIAL_LOSS_REDUCTIONS:
        raise ValueError("unknown calibration trial loss reduction")
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=2)
    if len(counts) != 2 or min(counts) < 1:
        raise ValueError("calibration class risk requires both actual fitted classes")
    weight = float(counts[0] / counts[1])
    mass = float((counts[0] + weight * counts[1]) / len(labels))
    record = {"schema": "n2p3_calibration_class_risk/1", "class_counts": counts.tolist(),
        "pos_weight": weight, "prior": float(counts[1] / len(labels)), "class_mass": mass,
        "loss_reduction": reduction, "fit_labels_sha256": semantic_sha256(labels.tolist())}
    loss = (CalibrationBalancedLoss(pos_weight=weight, class_mass=mass, device=device)
            if reduction == "calibration_balanced_fixed_mass" else
            nn.CrossEntropyLoss(weight=torch.tensor([1., weight], device=device)))
    return loss, record


def validate_class_calibration(record):
    weight, prior, temperature = (record[key] for key in ("pos_weight", "prior", "temperature"))
    if (any(isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value)
            for value in (weight, prior, temperature)) or weight <= 0 or not 0 < prior < 1 or temperature <= 0):
        raise ValueError("class calibration needs positive actual weight, interior prior and temperature")
    if record["origin"] == "target_calibration":
        risk = record["fit_contract"]["class_risk"]
        counts = risk["class_counts"]
        if (len(counts) != 2 or any(type(count) is not int or count < 1 for count in counts)
                or risk["loss_reduction"] not in TRIAL_LOSS_REDUCTIONS
                or weight != risk["pos_weight"] or prior != risk["prior"]
                or weight != counts[0] / counts[1] or prior != counts[1] / sum(counts)
                or risk["class_mass"] != (counts[0] + weight * counts[1]) / sum(counts)
                or record["fit_contract"]["class_counts"] != counts
                or record["fit_contract"]["prior"] != prior
                or record["fit_contract"]["recipe"].get("trial_loss_reduction", "batch_class_mass") != risk["loss_reduction"]):
            raise ValueError("target class calibration differs from the actual fitted risk")
    elif record["origin"] == "target_generative":
        contract = record["fit_contract"]
        counts = contract["class_counts"]
        if (weight != 1. or contract["class_risk"] is not None or len(counts) != 2
                or any(type(count) is not int or count < 1 for count in counts)
                or prior != contract["prior"] or prior != float(np.asarray(counts[1] / sum(counts),
                                                                          dtype=np.dtype(contract["prior_dtype"])))):
            raise ValueError("generative probability prior differs from its actual fitted classes")
    elif record["origin"] == "source_mixture":
        contract = record["fit_contract"]
        training = contract["training_contract"]
        if (semantic_sha256(training) != contract["training_contract_digest"]
                or training["optimizer"].get("config", {}).get("pos_weight") != weight):
            raise ValueError("source probability class cost differs from its actual training programme")
    else:
        raise ValueError("unknown fitted probability population")
    return float(weight), float(prior), float(temperature)


@dataclass(frozen=True)
class TrialProbabilityTransform:
    class_log_weight: float
    prior_logit: float
    temperature: float

    @classmethod
    def from_record(cls, calibration):
        weight, prior, temperature = validate_class_calibration(calibration)
        return cls(float(np.log(weight)), float(np.log(prior / (1 - prior))), temperature)

    def apply(self, raw):
        llr = (np.asarray(raw, dtype=np.float64) - self.class_log_weight - self.prior_logit) / self.temperature
        return llr, llr + self.prior_logit


def semantic_trial_logits(raw, calibration):
    return TrialProbabilityTransform.from_record(calibration).apply(raw)
