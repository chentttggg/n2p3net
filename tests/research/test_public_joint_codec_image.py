"""The joint numeric decoder consumes the same archive image it authenticated."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.private_ar1_noise_point import PrivateAR1NoisePoint
from research.public_joint_response_noise import load_public_joint_fit
from transfer.noise_model import array_digest


def test_joint_codec_decodes_the_authenticated_archive_image(tmp_path, monkeypatch):
    reference = SimpleNamespace(source=SimpleNamespace(identity='synthetic-source'),
        within_reference=SimpleNamespace(identity='synthetic-reference'), basis=np.array([[1.]]),
        dictionary=SimpleNamespace(digest=lambda: 'synthetic-dictionary'))

    def archive(marker):
        stream = io.BytesIO()
        metadata = {'schema': 'public_joint_response_noise_numeric_fit_artifact/1',
            'numeric_only_no_live_raw_source_or_C_grant': True,
            'source_identity': reference.source.identity, 'reference_identity': reference.within_reference.identity,
            'basis_sha256': array_digest(reference.basis), 'dictionary': reference.dictionary.digest(),
            'noise': {'marker': marker}}
        np.savez(stream, metadata=np.array(json.dumps(metadata)))
        return stream.getvalue()

    authenticated, replacement = archive('authenticated'), archive('replacement')
    path = tmp_path/'joint.npz'
    path.write_bytes(authenticated)
    actual_read, decoded = Path.read_bytes, []

    def read_then_replace(value):
        payload = actual_read(value)
        if value == path:
            value.write_bytes(replacement)
        return payload

    class DecodeReachedError(RuntimeError):
        pass

    def stop_after_decode(record):
        decoded.append(record['marker'])
        raise DecodeReachedError

    monkeypatch.setattr(Path, 'read_bytes', read_then_replace)
    monkeypatch.setattr(PrivateAR1NoisePoint, 'from_record', stop_after_decode)
    with pytest.raises(DecodeReachedError):
        load_public_joint_fit(path, expected_sha256=hashlib.sha256(authenticated).hexdigest(), reference=reference)
    assert decoded == ['authenticated']
    assert actual_read(path) == replacement
