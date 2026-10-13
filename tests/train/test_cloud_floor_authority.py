"""Central CUDA admission is independent of caller defaults and overrides."""

import json

import pytest

from research import source_execution


def central_floor():
    return json.loads(source_execution.CLOUD_RESOURCE_CONFIG.read_text(encoding="utf-8"))[
        "cloud_resource_floor"
    ]


def resolve(plan, *, device="cuda", cpu_count=64, group_count=8):
    return source_execution.resolve_source_resources(
        plan, group_count, device, cpu_count=cpu_count,
        free_device_bytes=256 * 2**30, free_host_bytes=512 * 2**30,
    )


@pytest.mark.parametrize("plan", [{}, {"cloud_resource_floor": None}])
def test_null_or_omitted_cuda_floor_uses_central_authority(plan):
    before = dict(plan)
    normalized = source_execution.with_cloud_resource_floor(plan, "cuda")
    assert normalized["cloud_resource_floor"] == central_floor()
    assert plan == before
    assert resolve(plan)["cloud_resource_floor"] == central_floor()
    assert resolve(normalized)["cloud_resource_floor"] == central_floor()


@pytest.mark.parametrize("entry", [source_execution.with_cloud_resource_floor, resolve])
@pytest.mark.parametrize("axis", ["cpu_threads", "gpu_workers"])
def test_explicit_cuda_override_cannot_lower_either_central_axis(entry, axis):
    floor = central_floor()
    floor[axis] -= 1
    with pytest.raises(ValueError, match="central cloud resource floor"):
        entry({"cloud_resource_floor": floor}, "cuda") if entry is source_execution.with_cloud_resource_floor else entry(
            {"cloud_resource_floor": floor}
        )


@pytest.mark.parametrize("floor", [False, 0, "25/4", [], {}, {"cpu_threads": 25},
                                   {"cpu_threads": True, "gpu_workers": 4},
                                   {"cpu_threads": 25, "gpu_workers": 0}])
def test_malformed_floor_is_typed_failure_at_both_ingresses(floor):
    for scope in ({}, {"evidence_scope": "software_correctness"}):
        plan = {**scope, "cloud_resource_floor": floor}
        with pytest.raises(ValueError, match="cloud_resource_floor requires positive"):
            source_execution.with_cloud_resource_floor(plan, "cuda")
        with pytest.raises(ValueError, match="cloud_resource_floor requires positive"):
            resolve(plan)


def test_stronger_explicit_floor_preserves_declaration_and_real_capacity():
    floor = {key: value + 1 for key, value in central_floor().items()}
    plan = {"cloud_resource_floor": floor, "consumer_identity": "unchanged"}
    assert source_execution.with_cloud_resource_floor(plan, "cuda") is plan
    assert resolve(plan)["cloud_resource_floor"] == floor
    with pytest.raises(ValueError, match="cannot meet declared"):
        resolve(plan, cpu_count=floor["cpu_threads"] - 1)


@pytest.mark.parametrize("scope", [{"resource_scope": "local_development"},
                                   {"resource_scope": "software_correctness"},
                                   {"evidence_scope": "software_correctness"}])
@pytest.mark.parametrize("declared", [None, {"cpu_threads": 1, "gpu_workers": 1}])
def test_explicit_probe_scope_can_use_small_cuda_capacity(scope, declared):
    plan = {**scope, "cpu_budget": 2, "cloud_resource_floor": declared}
    assert source_execution.with_cloud_resource_floor(plan, "cuda") is plan
    resources = resolve(plan, cpu_count=2, group_count=1)
    assert resources["source_workers"] == resources["gpu_slots"] == 1
    assert resources["cloud_resource_floor"] == declared


def test_generic_cpu_budget_and_dependency_tail_keep_their_original_roles():
    plan = {"cpu_budget": 2}
    assert source_execution.with_cloud_resource_floor(plan, "cpu") is plan
    assert resolve(plan, device="cpu", cpu_count=2, group_count=1)["source_workers"] == 1
    floor = central_floor()
    resources = resolve({"cpu_budget": floor["cpu_threads"]},
                        cpu_count=floor["cpu_threads"], group_count=1)
    assert resources["source_workers"] == 1  # One ready source task, not four invented fits.
    assert resources["gpu_slots"] >= floor["gpu_workers"]
    assert resources["target_workers"] >= floor["gpu_workers"]


def test_both_ingresses_consume_the_actual_config_instead_of_literal_floors(tmp_path, monkeypatch):
    path = tmp_path / "resources.json"
    floor = {"cpu_threads": 37, "gpu_workers": 6}
    path.write_text(json.dumps({"cloud_resource_floor": floor}), encoding="utf-8")
    monkeypatch.setattr(source_execution, "CLOUD_RESOURCE_CONFIG", path)
    assert source_execution.with_cloud_resource_floor({}, "cuda")["cloud_resource_floor"] == floor
    assert resolve({})["cloud_resource_floor"] == floor
    with pytest.raises(ValueError, match="central cloud resource floor"):
        resolve({"cloud_resource_floor": {"cpu_threads": 25, "gpu_workers": 4}})
