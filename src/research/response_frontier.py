"""Complete discrete likelihood partitions at one shared continuous point.

Each declared independent variable prior enters once. All positive states and
current candidates remain in the sum; table limits preserve a named outcome.
Continuous integration and input authority belong to their respective owners.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
from scipy.special import logsumexp


class FrontierBudgetExceeded(ValueError):  # noqa: N818 - accepted named budget outcome
    pass


def _array(value, name, ndim):
    result = np.asarray(value, dtype=float).copy()
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} requires finite actual arrays of dimension {ndim}")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class StateKey:
    q: str | int
    shape: str | int

    def __post_init__(self):
        if type(self.q) not in (str, int) or type(self.shape) not in (str, int):
            raise ValueError("state keys must retain explicit integer or string identities")


@dataclass(frozen=True)
class Roster:
    selection_id: str
    states: tuple[StateKey, ...]
    log_mass: np.ndarray

    def __post_init__(self):
        states = tuple(self.states)
        masses = _array(self.log_mass, "complete positive log masses", 1)
        if (not self.selection_id or not states or len(set(states)) != len(states)
                or masses.shape != (len(states),)):
            raise ValueError("roster requires the complete unique declared positive states")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "log_mass", masses)


@dataclass(frozen=True)
class LogFactor:
    """Log likelihood only; sum_product inserts each roster prior once."""

    scope: tuple[int, ...]
    values: np.ndarray


@dataclass(frozen=True)
class PointFactors:
    rosters: tuple[Roster, ...]
    factors: tuple[LogFactor, ...]
    current_index: int
    current_q_values: tuple[str | int, ...]
    log_common: float
    elimination_order: tuple[int, ...]
    binding: dict


@dataclass(frozen=True)
class Partition:
    current_q_values: tuple[str | int, ...]
    log_joint_mass_by_q: np.ndarray
    current_log_prior_mass_by_q: np.ndarray
    log_likelihood_by_q: np.ndarray
    binding: dict


def sum_product(point, *, maximum_intermediate_entries):
    """Canonical output includes current prior once. No probability or integral."""
    if type(maximum_intermediate_entries) is not int or maximum_intermediate_entries < 1:
        raise ValueError("frontier table budget must be explicitly positive")
    if not np.isfinite(point.log_common):
        raise ValueError("pointwise common Gaussian/scalar factor must be finite")
    widths = tuple(len(roster.states) for roster in point.rosters)
    if any(width > maximum_intermediate_entries for width in widths):
        raise FrontierBudgetExceeded("a complete positive roster exceeds the declared table budget")
    if not 0 <= point.current_index < len(widths):
        raise ValueError("current variable is outside the declared roster")
    if (not point.current_q_values or len(set(point.current_q_values)) != len(point.current_q_values)
            or {state.q for state in point.rosters[point.current_index].states} != set(point.current_q_values)):
        raise ValueError("sum-product must retain the complete positive current q roster")
    factors = [LogFactor((index,), roster.log_mass) for index, roster in enumerate(point.rosters)]
    for factor in point.factors:
        if (factor.scope != tuple(sorted(set(factor.scope))) or not factor.scope
                or any(type(index) is not int or not 0 <= index < len(widths) for index in factor.scope)
                or factor.values.shape != tuple(widths[index] for index in factor.scope)
                or np.isnan(factor.values).any() or np.isposinf(factor.values).any()):
            raise ValueError("log factor does not preserve its complete declared scope")
        if factor.values.size > maximum_intermediate_entries:
            raise FrontierBudgetExceeded("declared factor already exceeds the complete-state table budget")
        factors.append(factor)
    current = point.current_index
    expected = set(range(len(widths)))-{current}
    if set(point.elimination_order) != expected or len(point.elimination_order) != len(expected):
        raise ValueError("elimination must retain current and eliminate every other variable once")
    constant = point.log_common
    maximum_entries = max(factor.values.size for factor in factors)
    maximum_width, table_evaluations = max(map(lambda factor: len(factor.scope), factors)), 0
    trace = []
    for variable in point.elimination_order:
        bucket = [factor for factor in factors if variable in factor.scope]
        if not bucket:
            raise ValueError("each positive variable requires its prior factor")
        scope = tuple(sorted(set().union(*(factor.scope for factor in bucket))))
        entries = prod(widths[index] for index in scope)
        if entries > maximum_intermediate_entries:
            raise FrontierBudgetExceeded(f"complete frontier scope {scope} requires {entries} entries")
        table = np.zeros(tuple(widths[index] for index in scope))
        for factor in bucket:
            shape = [1]*len(scope)
            for axis, index in enumerate(factor.scope):
                shape[scope.index(index)] = factor.values.shape[axis]
            table += factor.values.reshape(shape)
        reduced = logsumexp(table, axis=scope.index(variable))
        factors = [factor for factor in factors if variable not in factor.scope]
        remaining = tuple(index for index in scope if index != variable)
        if remaining:
            factors.append(LogFactor(remaining, reduced))
        else:
            constant += float(reduced)
        maximum_entries = max(maximum_entries, entries)
        maximum_width = max(maximum_width, len(scope))
        table_evaluations += entries
        trace.append({"selection_id": point.rosters[variable].selection_id,
            "scope": [point.rosters[index].selection_id for index in scope], "entries": entries})
    values = np.zeros(widths[current])+constant
    for factor in factors:
        if factor.scope != (current,):
            raise ValueError("the frontier did not close onto the complete current state")
        values += factor.values
    roster = point.rosters[current]
    joint, prior = [], []
    for q in point.current_q_values:
        selected = np.array([state.q == q for state in roster.states])
        if not selected.any():
            raise ValueError("every declared current q must retain positive states")
        joint.append(logsumexp(values[selected]))
        prior.append(logsumexp(roster.log_mass[selected]))
    joint, prior = np.asarray(joint), np.asarray(prior)
    if np.isnan(joint).any() or np.isposinf(joint).any() or not np.isfinite(joint).any():
        raise ValueError("pointwise partition arithmetic is unresolved")
    return Partition(point.current_q_values, joint, prior, joint-prior,
        {**point.binding, "canonical_output": "joint_mass_including_current_prior_once",
         "maximum_intermediate_entries": maximum_entries, "maximum_elimination_width": maximum_width,
         "total_table_entries_evaluated": table_evaluations, "elimination_trace": trace,
         "outer_continuous_integration_executed": False,
         "numerical_error_role": "float64_reference_not_production_absolute_nats_certificate"})
