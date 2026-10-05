import pytest

from research.source_catalog import CORE_STUDIES, DEFAULT_POOLS, SourceCatalog
from research.source_matrix import (
    freeze_matched_dose,
    matched_dose_tasks,
    select_source_matrix_tasks,
    source_matrix_tasks,
)


def test_historical_matrix_is_complete_without_an_opt_in():
    assert select_source_matrix_tasks() == tuple(source_matrix_tasks())
    assert len(source_matrix_tasks()) == 153
    assert SourceCatalog().studies('B7', excluded=['BI2014a']) == tuple(s for s in DEFAULT_POOLS['B7'] if s != 'BI2014a')
    with pytest.raises(ValueError, match='frozen definition'):
        SourceCatalog({'B0': ['changed']})


def test_expanded_fold_checks_membership_not_just_count():
    catalog = SourceCatalog()
    outer = {'source_pool': 'B7', 'excluded_source_datasets': ['BI2014a', 'BI2015a'],
             'targets': [{'study': 'BI2014a'}, {'study': 'BI2015a'}]}
    actual = catalog.studies('B7', excluded=outer['excluded_source_datasets'])
    assert len(catalog.validate_fold(outer, actual)) == 7
    for wrong in (actual[:-1], (*actual[:-1], 'impostor'), (*actual, 'BI2014a')):
        with pytest.raises(ValueError, match='membership|excluded'):
            catalog.validate_fold(outer, wrong)
    with pytest.raises(ValueError, match='target studies'):
        catalog.validate_fold({**outer, 'target_study': 'Speier2017'}, actual)
    with pytest.raises(ValueError, match='explicit source_studies'):
        catalog.validate_fold({'excluded_source_datasets': ['held']}, actual)
    assert catalog.validate_fold({'source_studies': actual, 'excluded_source_datasets': ['held']}, actual) == actual


def test_new_pool_plan_and_single_probe_preserve_comparison_dose():
    plan = {'schema': 'n2p3_source_pool_plan/1', 'source_pools': {
        'base': ['a', 'b', 'held', 'family'], 'extra': ['a', 'b', 'c', 'held', 'family']},
        'targets': {'held': ['held', 'family']}, 'seeds': [7, 9], 'matched_pools': ['base', 'extra']}
    declared = source_matrix_tasks(plan)
    assert len(declared) == 8
    assert all(not set(t['studies']) & {'held', 'family'} for t in declared)
    selected = select_source_matrix_tasks(plan=plan, selected_ids=['Bpublic_extra_held_matched_s7'])
    dependencies = matched_dose_tasks(selected, declared)
    assert len(dependencies) == 4
    counts = {s: {f'{s}/{p}': [10+p, 3+p] for p in range(4)} for s in ['a', 'b', 'c']}
    whole = freeze_matched_dose(counts, tasks=declared)
    probe = freeze_matched_dose(counts, tasks=dependencies)
    assert probe == whole
    assert probe['folds']['held']['participants'] == 8
    assert freeze_matched_dose({}, tasks=[])['folds'] == {}
    changed = {**counts, 'held': {'unseen': [9999, 9999]}, 'family': {'unseen': [1, 1]}}
    assert freeze_matched_dose(changed, tasks=dependencies)['folds'] == whole['folds']
    with pytest.raises(ValueError, match='lacks study counts'):
        freeze_matched_dose({'a': counts['a']}, tasks=dependencies)


def test_new_plan_rejects_implicit_or_colliding_designs():
    plan = {'schema': 'n2p3_source_pool_plan/1', 'source_pools': {'new': list(CORE_STUDIES)},
            'seeds': [7], 'targets': {'held': ['held']}}
    for altered in ({'seeds': [7, 7]}, {'targets': {'held': ['other']}},
                    {'source_pools': {'bad/path': ['a']}}, {'matched_pools': ['absent']}):
        with pytest.raises(ValueError):
            source_matrix_tasks({**plan, **altered})


def test_deployment_is_explicit_complete_and_never_an_outer_evaluation():
    catalog = SourceCatalog()
    outer = {"role": "deployment", "source_pool": "B7", "excluded_source_datasets": []}
    assert catalog.validate_fold(outer, DEFAULT_POOLS["B7"]) == DEFAULT_POOLS["B7"]
    for changed in ({"role": "evaluation"}, {"target_study": "BI2014a"},
                    {"excluded_source_datasets": ["BI2014a"]}, {"role": "typo"}):
        with pytest.raises(ValueError):
            catalog.validate_fold({**outer, **changed}, DEFAULT_POOLS["B7"])
    with pytest.raises(ValueError, match="complete source membership"):
        catalog.validate_fold({"role": "deployment", "excluded_source_datasets": []}, ["a", "b", "c"])
