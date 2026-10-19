"""Exclusive CLI publications for immutable BrainSync artifacts."""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def new_output_reservation(path, *, inputs=(), companion_paths=()):
    output = Path(path).resolve()
    input_paths = {Path(value).resolve() for value in inputs if value is not None}
    if {output, *(Path(value).resolve() for value in companion_paths)} & input_paths:
        raise ValueError('new output must differ from every input artifact')
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name('.'+output.name+'.writer.lock')
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'output': str(output)})+'\n')
    try:
        if output.exists() or any(Path(value).exists() for value in companion_paths):
            raise FileExistsError('new output and its companion artifacts must not exist')
        yield output
    finally:
        lock.unlink()


def publish_json_new(path, record):
    path = Path(path).resolve()
    content = (json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')
    with tempfile.TemporaryDirectory(prefix='.'+path.name+'.publish-', dir=path.parent) as directory:
        staged = Path(directory)/path.name
        with staged.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, path)
    return path


def publish_epoch_cache_new(path, dataset, *, compressed=True):
    from data.epochs import save_epoch_dataset

    return save_epoch_dataset(Path(path).resolve(), dataset, compressed=compressed, exclusive=True)
