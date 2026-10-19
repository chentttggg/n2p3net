"""Output-path contract for newly derived epoch-cache files."""
from __future__ import annotations

from pathlib import Path

EPOCH_CACHE_OUTPUT_HELP = "New .npz cache file and .record.json sidecar; existing outputs are refused."


def new_epoch_cache_path(value: str | Path) -> Path:
    """Check the declared file path before expensive raw preparation.

    Publication still uses save_epoch_dataset(exclusive=True), which owns the
    concurrent writer check. This preflight creates no directories or files.
    """
    path = Path(value).expanduser()
    if path.suffix.lower() != '.npz':
        raise ValueError('epoch cache output must be a .npz file')
    for candidate in (path, path.with_suffix('.record.json')):
        if candidate.exists() or candidate.is_symlink():
            raise ValueError(f'epoch cache output already exists: {candidate}')
    for parent in path.parents:
        if parent.exists():
            if not parent.is_dir():
                raise ValueError(f'epoch cache parent is not a directory: {parent}')
            break
    return path
