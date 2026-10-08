"""Read-only indexed source rows without a complete raw subset allocation."""
from __future__ import annotations

import numpy as np


class SourceRows:
    """The same ordered rows as NumPy indexing, materialized only in bounded blocks."""

    def __init__(self, base: np.ndarray, rows: np.ndarray, channels=None):
        self.base = np.asarray(base).view()
        indices = np.asarray(rows)
        if indices.dtype.kind not in "iu":
            raise ValueError("source row view requires integer row indices")
        self.rows = np.array(indices, dtype=np.int64, copy=True)
        if (self.base.ndim != 3 or self.rows.ndim != 1 or not len(self.rows)
                or np.any(self.rows < 0) or np.any(self.rows >= len(self.base))):
            raise ValueError("source row view requires valid ordered EEG row indices")
        self.channels = np.arange(self.base.shape[1]) if channels is None else np.array(channels, dtype=np.int64, copy=True)
        if (self.channels.ndim != 1 or not len(self.channels) or np.any(self.channels < 0)
                or np.any(self.channels >= self.base.shape[1])):
            raise ValueError("source row view requires valid channel indices")
        self.base.flags.writeable = self.rows.flags.writeable = self.channels.flags.writeable = False
        self.shape = (len(self.rows), len(self.channels), self.base.shape[2])
        self.dtype = self.base.dtype
        self.ndim = 3
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if (not key or len(key) > 3 or any(value is Ellipsis or value is None for value in key)
                    or any(not isinstance(value, (int, np.integer, slice)) for value in key[1:])):
                raise ValueError("source row views support row-first indexing with basic channel/time axes")
            channel = key[1] if len(key) > 1 else slice(None)
            time = key[2] if len(key) > 2 else slice(None)
            if isinstance(channel, (int, np.integer)):
                return self.base[self.rows[key[0]], self.channels[channel], time]
            values = self.base[self.rows[key[0]]]
            return values[..., self.channels[channel], time]
        if key is Ellipsis or key is None:
            raise ValueError("source row views require explicit row-first indexing")
        values = self.base[self.rows[key]]
        if np.array_equal(self.channels, np.arange(self.base.shape[1])):
            return values
        return values[..., self.channels, :]

    def select(self, rows, channels):
        return SourceRows(self.base, self.rows[rows], self.channels[channels])

    def __array__(self, dtype=None, copy=None):
        raise ValueError("source row views must be consumed through bounded row blocks")

    def all_finite(self):
        return all(np.isfinite(self[start:start + 2048]).all() for start in range(0, len(self), 2048))
