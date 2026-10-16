"""P-only memo preserves the original scalar and numerical-error calculation."""
from types import SimpleNamespace

import numpy as np
import pytest

from research import joint_flat_collapsed_predictive as owner


def _tree(record_id='original-a', scale=1.):
    record = SimpleNamespace(record_id=record_id, response_width=1, channels=1, drift_width=1,
        startup_rows=np.array([[1., .2], [.4, -.1]])*scale,
        startup_errors=np.full((2, 2), 1e-14), startup_response_design=None)
    return SimpleNamespace(record=record, root=np.array([[1., .3, .2], [0., .5, -.1], [0., 0., .7]])*scale,
        errors=[None, np.full(3, 1e-13)])


def _bits(pair):
    return np.asarray(pair, dtype=np.float64).tobytes()


def _count_calls(monkeypatch):
    original, calls = owner._marginal, []
    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(owner, '_marginal', counted)
    return original, calls


def test_identical_actual_content_reuses_scalar_pair_across_distinct_objects(monkeypatch):
    original, calls = _count_calls(monkeypatch)
    cache, precision = owner._PrefixMarginalCache(65536), np.array([.7])
    first = cache((_tree(),), precision)
    second = cache((_tree(),), precision.copy())
    assert _bits(first) == _bits(second) == _bits(original((_tree(),), precision))
    assert (len(calls), cache.hits, cache.misses, len(cache.entries)) == (1, 1, 1, 1)


@pytest.mark.parametrize('change', ('lambda', 'root', 'errors', 'startup', 'startup_errors',
    'prior_error', 'root_dtype', 'lambda_dtype', 'layout', 'shape', 'record_id'))
def test_changed_original_operand_does_not_reuse_another_scalar_error_pair(monkeypatch, change):
    original, calls = _count_calls(monkeypatch)
    tree, precision, prior_error = _tree(), np.array([.7]), 0.
    cache = owner._PrefixMarginalCache(65536)
    cache((tree,), precision, prior_covariance_error=prior_error)
    if change == 'lambda':
        precision[0] = np.nextafter(precision[0], np.inf)
    elif change == 'root':
        tree.root[0, -1] += .01
    elif change == 'errors':
        tree.errors[1][0] *= 2
    elif change == 'startup':
        tree.record.startup_rows[0, -1] += .01
    elif change == 'startup_errors':
        tree.record.startup_errors[0, -1] *= 2
    elif change == 'prior_error':
        prior_error = 1e-8
    elif change == 'root_dtype':
        tree.root = tree.root.astype(np.float32)
    elif change == 'lambda_dtype':
        precision = precision.astype(np.float32)
    elif change == 'layout':
        tree.record.startup_errors = np.asfortranarray(tree.record.startup_errors)
    elif change == 'shape':
        tree.root = np.vstack((tree.root, np.zeros((1, 3))))
    else:
        tree.record.record_id = 'original-b'
    expected = original((tree,), precision, prior_covariance_error=prior_error)
    assert _bits(cache((tree,), precision, prior_covariance_error=prior_error)) == _bits(expected)
    assert (len(calls), cache.hits, cache.misses) == (2, 0, 2)


def test_true_record_and_precision_order_are_retained(monkeypatch):
    original, calls = _count_calls(monkeypatch)
    first, second = _tree(), _tree('original-b', 1.3)
    precision, cache = np.array([.7, 1.3]), owner._PrefixMarginalCache(65536)
    cache((first, second), precision)
    for records, values in (((second, first), precision), ((second, first), precision[::-1])):
        assert _bits(cache(records, values)) == _bits(original(records, values))
    assert (len(calls), cache.hits, cache.misses) == (3, 0, 3)


def test_bounded_lru_eviction_and_undersized_entry_recompute(monkeypatch):
    original, calls = _count_calls(monkeypatch)
    tree, first, second = _tree(), np.array([.7]), np.array([1.3])
    sizing = owner._PrefixMarginalCache(65536)
    sizing((tree,), first)
    entry_bytes = sizing.used_bytes
    cache = owner._PrefixMarginalCache(entry_bytes)
    for precision in (first, second, first):
        assert _bits(cache((tree,), precision)) == _bits(original((tree,), precision))
        assert cache.used_bytes <= cache.capacity_bytes and len(cache.entries) == 1
    assert (cache.hits, cache.misses) == (0, 3)
    tiny = owner._PrefixMarginalCache(entry_bytes-1)
    tiny((tree,), first)
    tiny((tree,), first)
    assert not tiny.entries and tiny.used_bytes == 0 and tiny.misses == 2
    assert len(calls) == 6


def test_zero_capacity_and_iterators_reach_the_original_arguments(monkeypatch):
    sentinel = (.25, 1e-8)
    calls = []
    def original(trees, precision, **kwargs):
        calls.append((trees, precision, kwargs))
        return sentinel
    monkeypatch.setattr(owner, '_marginal', original)
    tree, precision = _tree(), np.array([.7])
    cache = owner._PrefixMarginalCache(0)
    assert cache((tree,), precision) is sentinel
    assert calls[-1][1] is precision and cache.misses == 1 and not cache.entries
    values = iter((.7,))
    enabled = owner._PrefixMarginalCache(65536)
    assert enabled((tree,), values) is sentinel
    assert calls[-1][1] is values and next(values) == .7 and not enabled.entries


def test_original_startup_rejection_is_not_masked_or_cached(monkeypatch):
    _original, calls = _count_calls(monkeypatch)
    tree, precision = _tree(), np.array([.7])
    cache = owner._PrefixMarginalCache(65536)
    cache((tree,), precision)
    tree.record.startup_response_design = np.ones((2, 1))
    for _ in range(2):
        with pytest.raises(ValueError, match='structural startup response zero'):
            cache((tree,), precision)
    assert len(cache.entries) == 1 and cache.hits == 0 and len(calls) == 3


@pytest.mark.parametrize('capacity', (-1, 1.5, True))
def test_cache_budget_requires_a_nonnegative_integer(capacity):
    with pytest.raises(ValueError, match='nonnegative byte capacity'):
        owner._PrefixMarginalCache(capacity)
