"""Reject a fixed noise model on a different physical C or held time grid."""
from types import SimpleNamespace

import pytest

from experiments import audit_spatial_signal_prior as audit


@pytest.mark.parametrize("rates", [(256., 128.), (128., 256.)])
def test_fixed_noise_grid_must_match_both_actual_c_and_held_before_spatial_fit(tmp_path, monkeypatch, rates):
    paths = [tmp_path / name for name in ("input.mat", "noise.npz", "prior.json", "snapshot.json")]
    for path in paths:
        path.write_bytes(b"fixture identity; no scientific fit")
    baseline = SimpleNamespace(sfreq=128., noise=SimpleNamespace(startup_mode="stationary_innovations"))
    groups = tuple(SimpleNamespace(view=SimpleNamespace(sample_rate_hz=rate)) for rate in rates)
    monkeypatch.setattr(audit, "verified_executed_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(audit.ContinuousDecisionModel, "load", lambda path: baseline)
    monkeypatch.setattr(audit, "prepare_bnci_calibration_groups", lambda *args, **kwargs: SimpleNamespace(groups=groups))

    def forbidden(*args, **kwargs):
        pytest.fail("spatial fit consumed a mismatched physical time grid")

    monkeypatch.setattr(audit, "_layout_from_view", forbidden)
    with pytest.raises(ValueError, match="actual native C/held sampling"):
        audit.main(["--input", str(paths[0]), "--noise-model", str(paths[1]),
                    "--prior-program", str(paths[2]), "--snapshot", str(paths[3]),
                    "--fit-groups", "1", "--output", str(tmp_path / "output")])
