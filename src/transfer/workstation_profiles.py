"""Resolve the named daily-workstation program without altering old packages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from transfer.recipes import TargetRecipe

OLD_BASIC_PROFILE = Path(__file__).resolve().parents[2] / 'configs/brainsync_old_basic_nn_source.json'


def validate_executed_profile_source(profile_record, actual_sha256):
    if actual_sha256 != profile_record['source_checkpoint_sha256']:
        raise ValueError('executed source SHA-256 differs from the frozen old basic profile')


def load_old_basic_profile(path=OLD_BASIC_PROFILE, source_checkpoint=None):
    path = Path(path).resolve()
    content = path.read_bytes()
    profile = json.loads(content)
    if (profile.get('schema') != 'n2p3_workstation_calibration_profile/1'
            or profile.get('id') != 'old_basic_nn_source'):
        raise ValueError('unsupported workstation calibration profile')
    source = Path(source_checkpoint).resolve() if source_checkpoint is not None else (
        path.parent / profile['source']['path']).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != profile['source']['sha256']:
        raise ValueError('old basic source SHA-256 differs from the frozen twelve-study checkpoint')
    dependencies = {name: (path.parent / profile[name]).resolve()
                    for name in ('preprocessing', 'training_budget')}
    dependency_hashes = {name: hashlib.sha256(value.read_bytes()).hexdigest()
                         for name, value in dependencies.items()}
    return {
        'source_checkpoint': source, 'recipe': TargetRecipe(**profile['recipe']),
        **dependencies,
        'record': {'id': profile['id'], 'profile_sha256': hashlib.sha256(content).hexdigest(),
                   'source_checkpoint_sha256': digest, 'dependency_sha256': dependency_hashes},
    }
