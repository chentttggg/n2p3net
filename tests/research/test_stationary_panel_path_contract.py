"""The stationary panel must preserve BoundInput's Path-based manifest seam."""
from __future__ import annotations

import inspect

import pytest

from experiments import export_frozen_p1_state
from experiments.audit_channel_noise_mechanism import BoundInput
from research import stationary_panel


def test_string_manifest_is_the_stage04_counterexample_and_path_is_admitted(tmp_path):
    manifest = tmp_path / "input.json"
    manifest.write_text("{}", encoding="utf-8")

    # This is the exact stage04 failure: BoundInput's contract uses
    # manifest.parent and therefore does not silently reinterpret strings.
    with pytest.raises(AttributeError, match="parent"):
        BoundInput(str(manifest))

    admitted = BoundInput(manifest)
    assert admitted.root == manifest.parent.resolve()


def test_stage05_real_callers_coerce_serialized_manifest_paths():
    export_source = inspect.getsource(export_frozen_p1_state.export)
    panel_source = inspect.getsource(stationary_panel.worker)
    assert "manifest_path = Path(descriptor[\"input_manifest\"][\"path\"])" in export_source
    assert "inputs = BoundInput(manifest_path)" in export_source
    assert "BoundInput(Path(descriptor[\"input_manifest\"][\"path\"]))" in panel_source
