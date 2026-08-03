from __future__ import annotations

import importlib.util
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[3]


def _installer():
    path = _ROOT / "deployment" / "amd" / "scripts" / "install_dependencies.py"
    spec = importlib.util.spec_from_file_location("_dependency_lock_installer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checked_in_runtime_lock_matches_the_safe_installer_plan() -> None:
    lock = _ROOT / "deployment" / "amd" / "requirements.lock.txt"

    assert (
        lock.read_text(encoding="ascii").splitlines()
        == _installer().locked_requirements()
    )


def test_checked_in_runtime_lock_never_contains_torch_packages() -> None:
    lock = _ROOT / "deployment" / "amd" / "requirements.lock.txt"
    names = {
        line.split("==", 1)[0].casefold().replace("_", "-")
        for line in lock.read_text(encoding="ascii").splitlines()
    }

    assert names.isdisjoint({"torch", "torchvision", "torchaudio", "pytorch"})
