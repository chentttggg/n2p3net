import numpy as np

from data.harmonic import HarmonicConfig, clean_harmonic_segments


def test_frequency_drift_removal_preserves_erp_and_segment_isolation():
    fs = 250.0
    t = np.arange(3000) / fs
    erp = 12e-6 * np.exp(-0.5 * ((t - 5.3) / 0.13) ** 2)
    line = 8e-3 * np.sin(2 * np.pi * 50.01 * t + 0.3)
    raw = np.stack([erp + line, erp - 0.4 * line])
    config = HarmonicConfig()
    result = clean_harmonic_segments(raw, fs, [(0, len(t))], config)
    assert np.max(np.abs(result.cleaned - erp)) < 5e-6
    assert result.input_sha256 != result.output_sha256
    assert result.protocol == "offline_segment_local_estimation"
    raw_changed = np.concatenate([raw[:, :1500], raw[:, 1500:] * 100], axis=1)
    first = clean_harmonic_segments(raw, fs, [(0, 1500), (1500, 3000)], config)
    second = clean_harmonic_segments(raw_changed, fs, [(0, 1500), (1500, 3000)], config)
    np.testing.assert_array_equal(first.cleaned[:, :1500], second.cleaned[:, :1500])


def test_smooth_frequency_amplitude_phase_modulation_and_glitch():
    fs = 250.
    t = np.arange(4000) / fs
    clean = 12e-6 * np.exp(-0.5 * ((t - 7.3) / 0.13) ** 2)
    nuisance = .008 * (1 + .5 * np.sin(2 * np.pi * .1 * t))
    nuisance *= np.sin(2 * np.pi * (49.9 * t + .1 * t**2 / t[-1])
                       + .2 * np.sin(2 * np.pi * .3 * t))
    glitch = np.zeros(len(t))
    glitch[2700] = .02
    result = clean_harmonic_segments((clean + nuisance + glitch)[None], fs, [(0, len(t))])
    residual = result.cleaned[0] - clean - glitch
    assert np.sqrt(np.mean(residual**2)) < 5e-6
    # The glitch is retained as an acquisition/QC event, not silently inpainted.
    assert result.cleaned[0, 2700] > .019


def test_local_low_frequency_subspace_is_protected_from_line_regression():
    fs = 250.
    t = np.arange(1000) / fs
    line = .008 * np.sin(2 * np.pi * 50.01 * t)
    slow = 20e-6 * np.cos(2 * np.pi * 3 * (t + .5 / fs))
    base = clean_harmonic_segments(line[None], fs, [(0, len(t))])
    augmented = clean_harmonic_segments((line + slow)[None], fs, [(0, len(t))])
    np.testing.assert_allclose(augmented.cleaned - base.cleaned, slow[None], atol=1e-9, rtol=0)
