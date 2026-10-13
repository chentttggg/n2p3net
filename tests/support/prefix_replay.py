"""Actual raw/trial/grouped prefix graph shared by replay behavior fixtures."""
from dataclasses import replace

from research.calibration_selection import CalibrationProgramCandidate, fit_nested_prefix_selection
from tests.support.independent_prefix import independent_prefix_case
from train.runtime import cpu_thread_budget
from transfer.ensemble import EnsembleProgram
from transfer.ensemble_views import prepare_bnci_independent_prefixes


def create_prefix_replay_graph(root):
    root, _, path, spec, experts, old_views = independent_prefix_case(root)
    with cpu_thread_budget(1):
        views = prepare_bnci_independent_prefixes(path, source_sha256=old_views.ledger["source_sha256"],
            budgets=(0, 800), profiles=(experts[1].profile,), authorization_preprocessing=spec)
        raw = replace(experts[0], n_basis=1)
        roster = EnsembleProgram.equal_weights("replay-admission", (raw.expert_id, experts[1].expert_id))
        candidates = (CalibrationProgramCandidate("fixed", (raw, experts[1]), roster),
            CalibrationProgramCandidate("grouped", (raw, experts[1]), roster, "grouped_temperature"))
        future = next(view for view in views.decisions if view.samples is not None)
        selected = fit_nested_prefix_selection(views.calibration[800], candidates, baseline_name="fixed",
            prefix_compiler=views.prefix_compiler, minimum_selection_rows=1,
            context={"test_scope": views.ledger["test_scope"], "requested_decision_ids": [future.record_id]})
        expected = selected.score(future).record()
        directory = selected.save(root / "selection")
    return directory, views, future, selected.graph_sha256, expected
