from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

from data.domain import ensure_common_channel_average_reference
from data.ea_artifact import EAFitPlan, authorization_from_dataset, load_ea_feature_artifact
from data.epochs import epoch_row_identity_digest, load_epoch_dataset, save_epoch_dataset
from data.euclidean_alignment import EuclideanAlignmentConfig
from experiments.prepare_euclidean_alignment_cache import main
from tests.support.ea import make_ea_dataset
from tests.support.io import read_json
from tests.support.preparation_cli import cli_options


def _run_ea_cli(tmp_path, dataset, fit_scope="source_rows", *,
                reference_mode="native_preserve", fit_rows=None, **config):
    source = tmp_path / "source.npz"
    output = tmp_path / "source-ea.npz"
    save_epoch_dataset(source, dataset)
    parent = load_epoch_dataset(source, validation="attested")
    authorization = authorization_from_dataset(
        parent, fit_scope=fit_scope,
        fit_rows=np.arange(parent.n_epochs) if fit_rows is None else fit_rows,
    )
    plan = {"schema": "n2p3_ea_fit_plan/1", "plan_id": "test-experiment/split-1",
            "authorization": authorization.record(), "reference_mode": reference_mode,
            "config": asdict(EuclideanAlignmentConfig(**config))}
    plan_path = tmp_path / "fit-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    main(cli_options(dataset_cache=source, output=output, fit_plan=plan_path,
                     fit_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest()))
    return load_ea_feature_artifact(output, parent=parent, expected_plan=EAFitPlan.from_record(plan)), read_json(
        output.with_suffix(".record.json"))


def test_prepare_ea_cache_cli_writes_source_rows_scope(tmp_path):
    dataset = make_ea_dataset("source", ("s01", "s02", "s03", "s04"))
    adapted, record = _run_ea_cli(tmp_path, dataset, relative_eigenvalue_floor=1e-10)
    alignment = record["alignment"]
    assert alignment["schema"] == "n2p3_euclidean_alignment_fit/3"
    assert record["fit_plan"]["authorization"]["fit_scope"] == "source_rows"
    assert record["fit_plan"]["reference_mode"] == "native_preserve"
    assert adapted.authorization.parent_row_identity_sha256 == epoch_row_identity_digest(dataset)
    np.testing.assert_array_equal(adapted.y, dataset.y)
    assert adapted.name == "source-ea-source_rows"
    covariance = np.einsum("nct,nut->cu", adapted.X, adapted.X) / (len(adapted.X) * adapted.X.shape[2])
    np.testing.assert_allclose(covariance, np.eye(3), atol=2e-7)


def test_prepare_ea_cache_cli_writes_per_domain_scope(tmp_path):
    dataset = make_ea_dataset("multidomain", tuple(f"s{i:02d}" for i in range(8)),
                              domains=("A", "A", "A", "A", "B", "B", "B", "B"))
    adapted, record = _run_ea_cli(tmp_path, dataset, "per_domain")
    assert set(record["alignment"]["states"]) == {"A", "B"}
    assert adapted.row_domain_ids == ("A", "A", "A", "A", "B", "B", "B", "B")


def test_prepare_ea_cache_cli_rejects_per_domain_without_axis(tmp_path):
    dataset = make_ea_dataset("single-domain", ("s01", "s02", "s03"))
    with pytest.raises(ValueError, match="source_domain_axis"):
        _run_ea_cli(tmp_path, dataset, "per_domain")
    assert not (tmp_path / "source-ea.npz").exists()


def test_prepare_ea_cache_cli_rejects_uncovered_output_domain(tmp_path):
    dataset = make_ea_dataset("multidomain", ("a", "b", "c", "d"), domains=("A", "A", "B", "B"))
    with pytest.raises(ValueError, match="missing domain"):
        _run_ea_cli(tmp_path, dataset, "per_domain", fit_rows=[0, 1])
    assert not (tmp_path / "source-ea.npz").exists()


def test_prepare_ea_cache_cli_rejects_existing_output(tmp_path):
    dataset = make_ea_dataset("source", ("s01", "s02", "s03"))
    output = tmp_path / "source-ea.npz"
    output.write_bytes(b"occupied")
    with pytest.raises(FileExistsError, match="new output path"):
        _run_ea_cli(tmp_path, dataset)
    assert output.read_bytes() == b"occupied"


def test_prepare_ea_cache_cli_rejects_reference_mode_mismatch(tmp_path):
    native = make_ea_dataset("native", ("s01", "s02", "s03"))
    with pytest.raises(ValueError, match="common_average"):
        _run_ea_cli(tmp_path, native, reference_mode="common_average")
    car = ensure_common_channel_average_reference(native, native.channel_names)
    with pytest.raises(ValueError, match="native_preserve"):
        _run_ea_cli(tmp_path, car)


def test_prepare_ea_cache_cli_accepts_declared_common_average(tmp_path):
    native = make_ea_dataset("native", ("s01", "s02", "s03"))
    car = ensure_common_channel_average_reference(native, native.channel_names)
    adapted, record = _run_ea_cli(tmp_path, car, reference_mode="common_average")
    assert record["fit_plan"]["reference_mode"] == "common_average"
    assert adapted.source_reference == car.provenance["source_reference"]


def test_prepare_ea_cli_rejects_unpinned_or_edited_plan(tmp_path):
    plan = tmp_path / "fit-plan.json"
    plan.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="plan bytes"):
        main(cli_options(dataset_cache=tmp_path / "absent.npz", output=tmp_path / "out.npz",
                         fit_plan=plan, fit_plan_sha256="0" * 64))
    with pytest.raises(SystemExit):
        main(cli_options(dataset_cache=tmp_path / "absent.npz", output=tmp_path / "out.npz",
                         fit_scope="source_rows", reference_mode="native_preserve"))
