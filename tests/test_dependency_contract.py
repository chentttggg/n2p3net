from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.markers import Marker
from packaging.requirements import Requirement

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATED_LINUX = {
    "os_name": "posix",
    "platform_machine": "x86_64",
    "platform_system": "Linux",
    "python_full_version": "3.12.0",
    "python_version": "3.12",
    "sys_platform": "linux",
}
WINDOWS_PY314 = {
    "os_name": "nt",
    "platform_machine": "AMD64",
    "platform_system": "Windows",
    "python_full_version": "3.14.0",
    "python_version": "3.14",
    "sys_platform": "win32",
}
CUDA_128_PACKAGES = {
    "nvidia-cublas-cu12": "12.8.4.1",
    "nvidia-cuda-cupti-cu12": "12.8.90",
    "nvidia-cuda-nvrtc-cu12": "12.8.93",
    "nvidia-cuda-runtime-cu12": "12.8.90",
    "nvidia-cudnn-cu12": "9.10.2.21",
    "nvidia-cufft-cu12": "11.3.3.83",
    "nvidia-cufile-cu12": "1.13.1.3",
    "nvidia-curand-cu12": "10.3.9.90",
    "nvidia-cusolver-cu12": "11.7.3.90",
    "nvidia-cusparse-cu12": "12.5.8.93",
    "nvidia-cusparselt-cu12": "0.7.1",
    "nvidia-nccl-cu12": "2.27.3",
    "nvidia-nvjitlink-cu12": "12.8.93",
    "nvidia-nvtx-cu12": "12.8.90",
}


def _load_toml(name: str) -> dict[str, object]:
    with (PROJECT_ROOT / name).open("rb") as stream:
        return tomllib.load(stream)


def _requirement_applies(requirement: Requirement, environment: dict[str, str]) -> bool:
    return requirement.marker is None or requirement.marker.evaluate(environment)


def _active_project_requirements(environment: dict[str, str]) -> list[Requirement]:
    project = _load_toml("pyproject.toml")["project"]
    requirements = [Requirement(value) for value in project["dependencies"]]
    return [value for value in requirements if _requirement_applies(value, environment)]


def _torch_family(environment: dict[str, str]) -> list[tuple[str, str]]:
    return [
        (requirement.name, str(requirement.specifier))
        for requirement in _active_project_requirements(environment)
        if requirement.name in {"torch", "torchaudio"}
    ]


def _package_applies(package: dict[str, object], environment: dict[str, str]) -> bool:
    markers = package.get("resolution-markers")
    return markers is None or any(Marker(value).evaluate(environment) for value in markers)


def _active_packages(
    lock: dict[str, object], name: str, environment: dict[str, str]
) -> list[dict[str, object]]:
    return [
        package
        for package in lock["package"]
        if package["name"] == name and _package_applies(package, environment)
    ]


def test_validated_linux_dependency_branch_is_exact_and_non_overlapping() -> None:
    for python_version in ("3.11", "3.12", "3.13"):
        environment = VALIDATED_LINUX | {
            "python_full_version": f"{python_version}.0",
            "python_version": python_version,
        }
        assert set(_torch_family(environment)) == {
            ("torch", "==2.8.0"),
            ("torchaudio", "==2.8.0"),
        }

    linux_python314 = VALIDATED_LINUX | {
        "python_full_version": "3.14.0",
        "python_version": "3.14",
    }
    linux_aarch64 = VALIDATED_LINUX | {"platform_machine": "aarch64"}
    assert _torch_family(linux_python314) == [("torch", ">=2.4")]
    assert _torch_family(linux_aarch64) == [("torch", ">=2.4")]
    assert _torch_family(WINDOWS_PY314) == [("torch", ">=2.4")]


def test_validated_linux_lock_uses_only_cuda_128_torch_dependencies() -> None:
    lock = _load_toml("uv.lock")
    torch_packages = _active_packages(lock, "torch", VALIDATED_LINUX)
    audio_packages = _active_packages(lock, "torchaudio", VALIDATED_LINUX)

    assert [package["version"] for package in torch_packages] == ["2.8.0"]
    assert [package["version"] for package in audio_packages] == ["2.8.0"]

    torch_dependencies = {dependency["name"] for dependency in torch_packages[0]["dependencies"]}
    cuda_dependencies = {name for name in torch_dependencies if name.startswith("nvidia-")}
    assert cuda_dependencies == set(CUDA_128_PACKAGES)
    assert not any(name.endswith("-cu13") for name in torch_dependencies)

    triton = next(
        dependency
        for dependency in torch_packages[0]["dependencies"]
        if dependency["name"] == "triton"
    )
    assert triton["version"] == "3.4.0"

    audio_torch = next(
        dependency
        for dependency in audio_packages[0]["dependencies"]
        if dependency["name"] == "torch"
    )
    assert audio_torch["version"] == "2.8.0"

    locked_versions = {package["name"]: package["version"] for package in lock["package"]}
    assert {name: locked_versions[name] for name in CUDA_128_PACKAGES} == CUDA_128_PACKAGES


def test_windows_python314_uses_installable_development_branch() -> None:
    lock = _load_toml("uv.lock")
    torch_packages = _active_packages(lock, "torch", WINDOWS_PY314)
    assert len(torch_packages) == 1
    assert any("cp314-cp314-win_amd64.whl" in wheel["url"] for wheel in torch_packages[0]["wheels"])
