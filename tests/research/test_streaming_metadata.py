"""Streaming metadata retains exact canonical values and prewrite admission."""
import json
from contextlib import contextmanager

import pytest

from research.contracts import semantic_sha256
from research.frozen_metadata import freeze_metadata


def test_owned_digest_streams_exact_scalar_semantics_without_materializing_bytes(monkeypatch):
    value = {"unicode": "中文 \" \\ \n", "zeros": [-0., 0.],
        "atoms": [None, False, True, 2**100, 1e-9, 2.5]}
    expected = semantic_sha256(value)
    frozen = freeze_metadata(value)

    def forbidden(*args):
        raise AssertionError("semantic digest materialized a complete canonical byte value")

    monkeypatch.setattr("research.contracts._encode_canonical_tree", forbidden)
    assert frozen.semantic_digest() == expected
    assert frozen._canonical_bytes is None
    assert frozen.semantic_digest() == expected


def test_stream_writer_counts_and_admits_exact_bytes_before_creating_file(tmp_path):
    from research.execution import atomic_write_json_stream

    path = tmp_path / "graph.json"
    payload = {"shared": {"members": [True, False] * 100}, "text": "中文"}
    expected = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    calls = []

    class Admission:
        @contextmanager
        def artifact_writer(self, target, nbytes, *, temporary_paths):
            assert target == path and nbytes == len(expected)
            assert not path.exists() and not any(p.exists() for p in temporary_paths)
            calls.append(nbytes)
            yield

    atomic_write_json_stream(path, payload, input_admission=Admission())
    assert path.read_bytes() == expected and calls == [len(expected)]


def test_stream_writer_failure_or_denial_preserves_previous_bytes(tmp_path):
    from research.execution import atomic_write_json_stream

    path = tmp_path / "graph.json"
    path.write_bytes(b"original")
    with pytest.raises(ValueError, match="Out of range float"):
        atomic_write_json_stream(path, {"bad": float("nan")})
    assert path.read_bytes() == b"original"

    class Denied:
        def artifact_writer(self, *args, **kwargs):
            raise ValueError("actual byte budget denied")

    with pytest.raises(ValueError, match="budget denied"):
        atomic_write_json_stream(path, {"valid": True}, input_admission=Denied())
    assert path.read_bytes() == b"original" and list(tmp_path.iterdir()) == [path]
