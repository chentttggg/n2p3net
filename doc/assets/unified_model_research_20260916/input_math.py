"""Check information-rank and interaction counterexamples without training a model."""

import json
from pathlib import Path

import numpy as np
import torch


def gate_gradients(gate_value: float, residual_weight: float) -> tuple[float, float]:
    """Differentiate one scalar residual example, without optimizer updates."""
    value = torch.tensor(2.0)
    gate = torch.tensor(gate_value, requires_grad=True)
    weight = torch.tensor(residual_weight, requires_grad=True)
    prediction = value + gate * weight * value
    (prediction - 3.0).square().backward()
    return float(gate.grad), float(weight.grad)


def main() -> None:
    root = Path(__file__).resolve().parent
    anchor = 1
    contrast_channels = [0, 2, 3, 4, 5, 6]
    difference = np.eye(7)[contrast_channels] - np.eye(7)[anchor]
    reversible = np.vstack((difference, np.eye(7)[anchor]))
    rng = np.random.default_rng(9)
    native = rng.normal(size=(7, 179))
    transformed = reversible @ native
    recovered = np.empty_like(native)
    recovered[anchor] = transformed[-1]
    recovered[contrast_channels] = transformed[:-1] + transformed[-1]
    common = rng.normal(size=(1, 179))
    assert np.linalg.matrix_rank(difference) == 6
    assert np.linalg.matrix_rank(reversible) == 7
    assert np.allclose(difference @ (native + common), difference @ native)
    assert np.allclose(recovered, native)
    gtn_difference = np.array([[1.0, -1.0, 0.0], [0.0, -1.0, 1.0]])
    assert np.linalg.matrix_rank(gtn_difference) == 2

    time = np.linspace(-0.2, 1.2, 179)
    wave = np.exp(-0.5 * ((time - 0.35) / 0.07) ** 2)
    power_difference = np.max(
        np.abs(np.abs(np.fft.rfft(wave)) ** 2 - np.abs(np.fft.rfft(-wave)) ** 2)
    )
    assert power_difference == 0
    dead = gate_gradients(0.0, 0.0)
    gate_only = gate_gradients(0.0, 1.0)
    assert dead == (0.0, 0.0)
    assert gate_only[0] != 0 and gate_only[1] == 0
    assert np.isclose(0.2 * 3, 0.4 * 1.5)
    base, only_s, only_m, both = 0.70, 0.70, 0.70, 0.73
    interaction = both - only_s - only_m + base
    shared = np.array([-1.0, 0.0, 1.0])
    private = shared**2
    private -= private.mean()
    covariance = float(np.mean(shared * private))
    assert covariance == 0 and np.ptp(private) > 0

    metrics = {
        "rank_Cz_difference": int(np.linalg.matrix_rank(difference)),
        "rank_difference_plus_native_Cz": int(np.linalg.matrix_rank(reversible)),
        "reconstruction_max_error": float(np.max(np.abs(recovered - native))),
        "reference_shift_difference_error": float(
            np.max(np.abs(difference @ (native + common) - difference @ native))
        ),
        "GTN_available_contrast_rank": int(np.linalg.matrix_rank(gtn_difference)),
        "GTN_native_rank": int(np.linalg.matrix_rank(np.eye(3))),
        "positive_negative_wave_energy_distance": float(power_difference),
        "zero_gate_zero_residual_gradients": dead,
        "zero_gate_nonzero_residual_gradients": gate_only,
        "gate_rescale_same_function": True,
        "interaction_missed_by_baseline_single_ablation": interaction,
        "orthogonal_but_deterministic_dependent_features_covariance": covariance,
    }
    (root / "input_math.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, allow_nan=False))


if __name__ == "__main__":
    main()
