from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
import fcntl
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
from typing import Any

import pytest

from specstyle.domain.enums import RuleLevel, RuleScope, RuleStatus
from specstyle.domain.identifiers import ArtifactId, Identifier, RuleId, Sha256
from specstyle.errors import DomainError, InfrastructureError
from specstyle.generation.model_registry import ModelDescriptor
from specstyle.generation.pipeline_factory import PipelineGraph
from specstyle.observability.environment import (
    DeviceInventory,
    DeviceSnapshot,
    EnvironmentSnapshot,
    IntegerObservation,
    TextObservation,
    hash_environment,
)
from specstyle.spec.compiled_models import (
    EncoderCapability,
    ModelCapability,
    ResourcePin,
    RuntimeCapability,
    ThresholdProfileCapability,
)
from specstyle.spec.compiler import compile_style_spec
from specstyle.spec.models import StyleSpecV1


def test_public_context_config_surface_is_frozen() -> None:
    spec = importlib.util.find_spec("specstyle.production.context_config")
    assert spec is not None
    module = importlib.import_module("specstyle.production.context_config")

    assert module.__all__ == (
        "ProductionContextConfig",
        "load_production_context_config",
        "make_production_compiler_context_factory",
    )
    assert tuple(
        inspect.signature(module.load_production_context_config).parameters
    ) == (
        "config_root_fd",
        "evidence_root_fd",
    )
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        for parameter in inspect.signature(
            module.load_production_context_config
        ).parameters.values()
    )
    assert tuple(
        inspect.signature(module.make_production_compiler_context_factory).parameters
    ) == ("config", "environment", "graph")
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        for parameter in inspect.signature(
            module.make_production_compiler_context_factory
        ).parameters.values()
    )
    assert "__dict__" not in module.ProductionContextConfig.__slots__
    with pytest.raises(TypeError):
        module.ProductionContextConfig()
    issued = object.__new__(module.ProductionContextConfig)
    with pytest.raises(FrozenInstanceError):
        issued._seal = object()


def test_context_config_import_graph_excludes_workflow() -> None:
    spec = importlib.util.find_spec("specstyle.production.context_config")
    assert spec is not None
    assert spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)

    assert not any(
        name == "specstyle.workflow" or name.startswith("specstyle.workflow.")
        for name in imports
    )


def test_context_config_uses_only_canonical_canny_contract() -> None:
    contracts = importlib.import_module("specstyle.generation.canny_contracts")
    module = importlib.import_module("specstyle.production.context_config")

    assert not hasattr(module, "_Canny")
    assert module.CannyProcessorConfig is contracts.CannyProcessorConfig


def _pin(identifier: str, character: str) -> dict[str, str]:
    return {"id": identifier, "revision": "r1", "sha256": character * 64}


def _context_document(
    evidence: dict[str, str], *, status: str = "VALIDATED"
) -> dict[str, Any]:
    l1_rules = [
        {
            "rule_id": rule_id,
            "verifier_pin": _pin(f"{rule_id}-verifier", character),
            "priority": priority,
            "affected_by_actions": [],
        }
        for priority, (rule_id, character) in enumerate(
            (
                ("l1_bundle", "4"),
                ("l1_decode", "5"),
                ("l1_dimensions", "6"),
                ("l1_pixels", "7"),
            )
        )
    ]
    return {
        "schema_version": "specstyle.production.context.v1",
        "compiler_pin": _pin("compiler", "1"),
        "model_support": [
            {
                "role": role,
                "supported_pipelines": ["sdxl_turbo", "sdxl_base"],
            }
            for role in ("base", "ip_adapter", "controlnet")
        ],
        "strength_mapping": {
            "pin": _pin("strength-mapping", "2"),
            "preset_id": "preset",
            "entries": [
                {
                    "user_strength": 0.0,
                    "preview_ip_adapter_scale": 0.0,
                    "production_ip_adapter_scale": 0.0,
                },
                {
                    "user_strength": 0.7,
                    "preview_ip_adapter_scale": 0.55,
                    "production_ip_adapter_scale": 0.72,
                },
                {
                    "user_strength": 1.0,
                    "preview_ip_adapter_scale": 1.0,
                    "production_ip_adapter_scale": 1.0,
                },
            ],
        },
        "output_profile": {"pin": _pin("xhs-output", "3")},
        "rule_catalog": {
            "ruleset_version": "1",
            "pin": _pin("rules", "8"),
            "l1_rules": l1_rules,
            "l2_item_rule": {
                "rule_id": "l2_style",
                "verifier_pin": _pin("l2-style-verifier", "9"),
                "metric_id": "reference_style_statistics_similarity",
                "priority": 10,
                "affected_by_actions": ["repair_style"],
            },
            "l2_batch_rule": {
                "rule_id": "l2_batch",
                "verifier_pin": _pin("l2-batch-verifier", "a"),
                "metric_id": "batch_style_consistency",
                "priority": 11,
                "affected_by_actions": [],
            },
        },
        "l2_threshold_profile": {
            "pin": _pin("l2-profile", "b"),
            "logical_name": "l2-product-instance",
            "status": status,
            "style_pack_id": "preset",
            "metric": {
                "metric_id": "reference_style_statistics_similarity",
                "operator": ">=",
                "value": 0.5,
            },
            "evidence": evidence,
        },
        "source_preprocess": {
            "processor_pin": _pin("source-processor", "c"),
            "resize_mode": "contain_pad",
            "background": [255, 255, 255],
        },
        "canny": {
            "low_threshold": 100,
            "high_threshold": 200,
            "aperture_size": 3,
            "l2_gradient": False,
        },
    }


def _write_roots(tmp_path: Path, *, status: str = "VALIDATED") -> tuple[Path, Path]:
    config_root, evidence_root = tmp_path / "config", tmp_path / "evidence"
    config_root.mkdir(mode=0o700)
    evidence_root.mkdir(mode=0o700)
    contents = {
        "calibration_dataset_sha256": b"calibration manifest",
        "validation_dataset_sha256": b"validation manifest",
        "annotation_protocol_sha256": b"annotation protocol manifest",
    }
    evidence = {
        key: hashlib.sha256(payload).hexdigest() for key, payload in contents.items()
    }
    for key, payload in contents.items():
        digest = evidence[key]
        directory = evidence_root / "sha256" / digest[:2]
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        (evidence_root / "sha256").chmod(0o700)
        path = directory / digest
        path.write_bytes(payload)
        path.chmod(0o400)
    config_path = config_root / "context.json"
    config_path.write_text(
        json.dumps(
            _context_document(evidence, status=status),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    config_path.chmod(0o600)
    return config_root, evidence_root


def _load(config_root: Path, evidence_root: Path):
    module = importlib.import_module("specstyle.production.context_config")
    config_fd = os.open(config_root, os.O_RDONLY | os.O_DIRECTORY)
    evidence_fd = os.open(evidence_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return module.load_production_context_config(config_fd, evidence_fd)
    finally:
        os.close(evidence_fd)
        os.close(config_fd)


def test_loading_valid_context_does_not_import_workflow_service(
    tmp_path: Path,
) -> None:
    config_root, evidence_root = _write_roots(tmp_path)
    src_root = Path(__file__).parents[3] / "src"
    script = """
import os
import sys

sys.path.insert(0, sys.argv[1])
from specstyle.generation.canny_contracts import CannyProcessorConfig
from specstyle.production.context_config import load_production_context_config

config_fd = os.open(sys.argv[2], os.O_RDONLY | os.O_DIRECTORY)
evidence_fd = os.open(sys.argv[3], os.O_RDONLY | os.O_DIRECTORY)
try:
    loaded = load_production_context_config(config_fd, evidence_fd)
finally:
    os.close(evidence_fd)
    os.close(config_fd)
assert loaded.schema_version == "specstyle.production.context.v1"
assert type(loaded.canny) is CannyProcessorConfig
forbidden = (
    "cv2",
    "numpy",
    "PIL",
    "Pillow",
    "diffusers",
    "torch",
    "gradio",
    "specstyle.workflow.production_service",
)
assert not [name for name in forbidden if name in sys.modules]
"""

    completed = subprocess.run(
        (
            sys.executable,
            "-B",
            "-c",
            script,
            str(src_root),
            str(config_root),
            str(evidence_root),
        ),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _read_document(config_root: Path) -> dict[str, Any]:
    return json.loads((config_root / "context.json").read_text(encoding="utf-8"))


def _write_document(config_root: Path, document: object) -> None:
    path = config_root / "context.json"
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    path.chmod(0o600)


_SCHEMA_OBJECTS = (
    ((), "schema_version"),
    (("compiler_pin",), "sha256"),
    (("model_support", 0), "supported_pipelines"),
    (("strength_mapping",), "preset_id"),
    (("strength_mapping", "pin"), "revision"),
    (("strength_mapping", "entries", 0), "production_ip_adapter_scale"),
    (("output_profile",), "pin"),
    (("rule_catalog",), "ruleset_version"),
    (("rule_catalog", "pin"), "id"),
    (("rule_catalog", "l1_rules", 0), "priority"),
    (("rule_catalog", "l1_rules", 0, "verifier_pin"), "sha256"),
    (("rule_catalog", "l2_item_rule"), "metric_id"),
    (("rule_catalog", "l2_batch_rule"), "affected_by_actions"),
    (("l2_threshold_profile",), "logical_name"),
    (("l2_threshold_profile", "pin"), "sha256"),
    (("l2_threshold_profile", "metric"), "operator"),
    (("l2_threshold_profile", "evidence"), "validation_dataset_sha256"),
    (("source_preprocess",), "resize_mode"),
    (("source_preprocess", "processor_pin"), "revision"),
    (("canny",), "aperture_size"),
)


def _at(document: object, path: tuple[object, ...]) -> Any:
    value = document
    for component in path:
        value = value[component]  # type: ignore[index]
    return value


def test_loads_verified_context_without_retaining_paths_or_evidence_bytes(
    tmp_path: Path,
) -> None:
    config_root, evidence_root = _write_roots(tmp_path)

    loaded = _load(config_root, evidence_root)

    assert loaded.schema_version == "specstyle.production.context.v1"
    assert loaded.compiler_pin == ResourcePin("compiler", "r1", Sha256("1" * 64))
    assert tuple(item.role for item in loaded.model_support) == (
        "base",
        "ip_adapter",
        "controlnet",
    )
    assert loaded.strength_mapping.entries[1].user_strength == 0.7
    assert loaded.output_profile.profile == "xhs_grid"
    assert loaded.output_profile.supported_domains == ("product_instance",)
    assert tuple(rule.rule_id for rule in loaded.rule_catalog.rules[:4]) == tuple(
        RuleId(value)
        for value in ("l1_bundle", "l1_decode", "l1_dimensions", "l1_pixels")
    )
    l2_item, l2_batch = loaded.rule_catalog.rules[-2:]
    assert (l2_item.level, l2_item.scope, l2_item.metric_id) == (
        RuleLevel.L2,
        RuleScope.ITEM,
        Identifier("reference_style_statistics_similarity"),
    )
    assert (l2_batch.scope, l2_batch.supported_output_profiles) == (
        RuleScope.BATCH,
        ("background_sequence",),
    )
    assert loaded.l2_threshold_profile.status == "VALIDATED"
    assert loaded.source_preprocess.background == (255, 255, 255)
    assert loaded.canny.low_threshold == 100
    contracts = importlib.import_module("specstyle.generation.canny_contracts")
    assert type(loaded.canny) is contracts.CannyProcessorConfig
    assert not any(
        isinstance(value, (bytes, Path))
        for value in (
            loaded.schema_version,
            loaded.compiler_pin,
            loaded.model_support,
            loaded.strength_mapping,
            loaded.output_profile,
            loaded.rule_catalog,
            loaded.l2_threshold_profile,
            loaded.source_preprocess,
            loaded.canny,
        )
    )


def test_loaded_l1_rule_ids_match_authoritative_binding_order(tmp_path: Path) -> None:
    from specstyle.verification.l1.production_bindings import (
        production_l1_rule_bindings,
    )

    config_root, evidence_root = _write_roots(tmp_path)

    loaded = _load(config_root, evidence_root)

    actual = tuple(
        rule.rule_id for rule in loaded.rule_catalog.rules if rule.level is RuleLevel.L1
    )
    expected = tuple(binding.rule_id for binding in production_l1_rule_bindings())
    assert actual == expected
    assert all(type(rule_id) is RuleId for rule_id in actual)


@pytest.mark.parametrize(("path", "missing"), _SCHEMA_OBJECTS)
@pytest.mark.parametrize("mutation", ("unknown", "missing"))
def test_rejects_unknown_or_missing_keys_at_every_context_layer(
    tmp_path: Path,
    path: tuple[object, ...],
    missing: str,
    mutation: str,
) -> None:
    config_root, evidence_root = _write_roots(tmp_path)
    document = _read_document(config_root)
    target = _at(document, path)
    if mutation == "unknown":
        target["unexpected"] = True
    else:
        target.pop(missing)
    _write_document(config_root, document)

    with pytest.raises(DomainError):
        _load(config_root, evidence_root)


@pytest.mark.parametrize("attack", ("duplicate", "nan", "depth", "surrogate"))
def test_rejects_noncanonical_context_json(tmp_path: Path, attack: str) -> None:
    config_root, evidence_root = _write_roots(tmp_path)
    path = config_root / "context.json"
    if attack == "duplicate":
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace(
                '"schema_version":', '"schema_version":"x","schema_version":', 1
            ),
            encoding="utf-8",
        )
    elif attack == "nan":
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace('"value":0.5', '"value":NaN'), encoding="utf-8")
    elif attack == "depth":
        path.write_bytes(b"[" * 10_000 + b"0" + b"]" * 10_000)
    else:
        document = _read_document(config_root)
        document["l2_threshold_profile"]["logical_name"] = "\ud800"
        _write_document(config_root, document)
    path.chmod(0o600)

    with pytest.raises(DomainError):
        _load(config_root, evidence_root)


@pytest.mark.parametrize(
    ("path", "key"),
    (
        (("rule_catalog", "l1_rules", 0), "priority"),
        (("canny",), "low_threshold"),
        (("source_preprocess",), "background"),
    ),
)
def test_rejects_bool_where_context_number_is_required(
    tmp_path: Path, path: tuple[object, ...], key: str
) -> None:
    config_root, evidence_root = _write_roots(tmp_path)
    document = _read_document(config_root)
    target = _at(document, path)
    target[key] = True if key != "background" else [True, 255, 255]
    _write_document(config_root, document)

    with pytest.raises(DomainError):
        _load(config_root, evidence_root)


def test_rejects_context_document_over_one_mib(tmp_path: Path) -> None:
    config_root, evidence_root = _write_roots(tmp_path)
    path = config_root / "context.json"
    path.write_bytes(path.read_bytes().ljust(1024 * 1024 + 1, b" "))
    path.chmod(0o600)

    with pytest.raises(InfrastructureError):
        _load(config_root, evidence_root)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("model_support",), []),
        (("model_support", 0, "role"), "ip_adapter"),
        (("model_support", 0, "supported_pipelines"), []),
        (("model_support", 0, "supported_pipelines"), ["sdxl_base", "sdxl_turbo"]),
        (("model_support", 0, "supported_pipelines"), ["sdxl_turbo"]),
        (
            ("model_support", 0, "supported_pipelines"),
            ["sdxl_turbo", "sdxl_turbo", "sdxl_base"],
        ),
        (("model_support", 0, "supported_pipelines"), ["unknown", "sdxl_base"]),
        (
            ("model_support", 0, "supported_pipelines"),
            {"lcm": True, "sdxl_base": True},
        ),
        (("strength_mapping", "entries", 0, "user_strength"), True),
        (
            ("rule_catalog", "l2_item_rule", "metric_id"),
            "batch_style_consistency",
        ),
        (("l2_threshold_profile", "status"), "REVOKED"),
        (("l2_threshold_profile", "status"), "UNKNOWN"),
        (("l2_threshold_profile", "logical_name"), " padded"),
        (("l2_threshold_profile", "style_pack_id"), "other_preset"),
        (("l2_threshold_profile", "metric", "metric_id"), "other_metric"),
        (("l2_threshold_profile", "metric", "operator"), "<="),
        (("l2_threshold_profile", "metric", "value"), -1.1),
        (("l2_threshold_profile", "metric", "value"), 1.1),
        (("source_preprocess", "resize_mode"), "stretch"),
        (("source_preprocess", "background"), [-1, 0, 0]),
        (("source_preprocess", "background"), [0, 0, 256]),
        (("canny", "low_threshold"), -1),
        (("canny", "low_threshold"), 200),
        (("canny", "high_threshold"), 256),
        (("canny", "aperture_size"), 5),
        (("canny", "l2_gradient"), True),
    ),
)
def test_rejects_invalid_context_semantics(
    tmp_path: Path, path: tuple[object, ...], value: object
) -> None:
    config_root, evidence_root = _write_roots(tmp_path)
    document = _read_document(config_root)
    target = _at(document, path[:-1])
    target[path[-1]] = value
    _write_document(config_root, document)

    with pytest.raises(DomainError):
        _load(config_root, evidence_root)


@pytest.mark.parametrize("mutation", ("missing", "reordered", "wrong"))
def test_rejects_l1_rules_that_do_not_match_public_production_bindings(
    tmp_path: Path, mutation: str
) -> None:
    config_root, evidence_root = _write_roots(tmp_path)
    document = _read_document(config_root)
    rules = document["rule_catalog"]["l1_rules"]
    if mutation == "missing":
        rules.pop()
    elif mutation == "reordered":
        rules[0], rules[1] = rules[1], rules[0]
    else:
        rules[0]["rule_id"] = "l1_other"
    _write_document(config_root, document)

    with pytest.raises(DomainError):
        _load(config_root, evidence_root)


@pytest.mark.parametrize("status", ("DRAFT", "CALIBRATED", "VALIDATED"))
def test_accepts_nonrevoked_l2_threshold_statuses(tmp_path: Path, status: str) -> None:
    config_root, evidence_root = _write_roots(tmp_path, status=status)

    loaded = _load(config_root, evidence_root)

    assert loaded.l2_threshold_profile.status == status


def _evidence_path(
    config_root: Path,
    evidence_root: Path,
    key: str = "calibration_dataset_sha256",
) -> Path:
    digest = _read_document(config_root)["l2_threshold_profile"]["evidence"][key]
    return evidence_root / "sha256" / digest[:2] / digest


@pytest.mark.parametrize("directory", ("root", "sha256", "prefix"))
def test_rejects_untrusted_evidence_directory_mode(
    tmp_path: Path, directory: str
) -> None:
    config_root, evidence_root = _write_roots(tmp_path)
    evidence_path = _evidence_path(config_root, evidence_root)
    target = {
        "root": evidence_root,
        "sha256": evidence_root / "sha256",
        "prefix": evidence_path.parent,
    }[directory]
    target.chmod(0o722)

    with pytest.raises(InfrastructureError):
        _load(config_root, evidence_root)


@pytest.mark.parametrize("directory", ("sha256", "prefix"))
def test_rejects_symlink_in_evidence_directory_chain(
    tmp_path: Path, directory: str
) -> None:
    config_root, evidence_root = _write_roots(tmp_path)
    evidence_path = _evidence_path(config_root, evidence_root)
    target = evidence_root / "sha256" if directory == "sha256" else evidence_path.parent
    relocated = target.with_name(f"real-{target.name}")
    target.rename(relocated)
    target.symlink_to(relocated.name, target_is_directory=True)

    with pytest.raises(InfrastructureError):
        _load(config_root, evidence_root)


@pytest.mark.parametrize("mode", (0o644, 0o200))
def test_rejects_evidence_file_mode(tmp_path: Path, mode: int) -> None:
    config_root, evidence_root = _write_roots(tmp_path)
    _evidence_path(config_root, evidence_root).chmod(mode)

    with pytest.raises(InfrastructureError):
        _load(config_root, evidence_root)


def test_accepts_private_read_write_evidence_file_mode(tmp_path: Path) -> None:
    config_root, evidence_root = _write_roots(tmp_path)
    _evidence_path(config_root, evidence_root).chmod(0o600)

    assert _load(config_root, evidence_root).schema_version == (
        "specstyle.production.context.v1"
    )


def test_rejects_hardlinked_evidence_file(tmp_path: Path) -> None:
    config_root, evidence_root = _write_roots(tmp_path)
    evidence_path = _evidence_path(config_root, evidence_root)
    os.link(evidence_path, evidence_path.with_name("alias"))

    with pytest.raises(InfrastructureError):
        _load(config_root, evidence_root)


def test_detects_evidence_file_change_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import specstyle.production.context_config as context_config

    config_root, evidence_root = _write_roots(tmp_path)
    evidence_path = _evidence_path(config_root, evidence_root)
    target_inode = evidence_path.stat().st_ino
    real_read = os.read
    changed = False

    def changing_read(fd: int, size: int) -> bytes:
        nonlocal changed
        chunk = real_read(fd, size)
        if not changed and os.fstat(fd).st_ino == target_inode:
            changed = True
            os.utime(evidence_path, None)
        return chunk

    monkeypatch.setattr(context_config.os, "read", changing_read)
    with pytest.raises(InfrastructureError, match="changed"):
        _load(config_root, evidence_root)


def test_rejects_missing_evidence_even_for_draft_threshold(tmp_path: Path) -> None:
    config_root, evidence_root = _write_roots(tmp_path, status="DRAFT")
    _evidence_path(config_root, evidence_root).unlink()

    with pytest.raises(InfrastructureError):
        _load(config_root, evidence_root)


def test_borrows_context_root_fds_and_does_not_leak_internal_fds(
    tmp_path: Path,
) -> None:
    config_root, evidence_root = _write_roots(tmp_path)
    config_fd = os.open(config_root, os.O_RDONLY | os.O_DIRECTORY)
    evidence_fd = os.open(evidence_root, os.O_RDONLY | os.O_DIRECTORY)
    before = len(os.listdir("/dev/fd"))
    try:
        module = importlib.import_module("specstyle.production.context_config")
        module.load_production_context_config(config_fd, evidence_fd)
        os.fstat(config_fd)
        os.fstat(evidence_fd)
        assert len(os.listdir("/dev/fd")) == before

        _evidence_path(config_root, evidence_root).chmod(0o644)
        with pytest.raises(InfrastructureError):
            module.load_production_context_config(config_fd, evidence_fd)
        os.fstat(config_fd)
        os.fstat(evidence_fd)
        assert len(os.listdir("/dev/fd")) == before
    finally:
        os.close(evidence_fd)
        os.close(config_fd)


@pytest.mark.parametrize("root", ("config", "evidence"))
def test_rejects_non_directory_context_root_fd(tmp_path: Path, root: str) -> None:
    config_root, evidence_root = _write_roots(tmp_path)
    file_fd = os.open(config_root / "context.json", os.O_RDONLY)
    config_fd = os.open(config_root, os.O_RDONLY | os.O_DIRECTORY)
    evidence_fd = os.open(evidence_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        module = importlib.import_module("specstyle.production.context_config")
        arguments = (file_fd, evidence_fd) if root == "config" else (config_fd, file_fd)
        with pytest.raises(InfrastructureError):
            module.load_production_context_config(*arguments)
    finally:
        os.close(evidence_fd)
        os.close(config_fd)
        os.close(file_fd)


@pytest.mark.parametrize("attack", ("mode", "symlink", "fifo", "changed"))
def test_rejects_untrusted_or_changed_context_file(
    tmp_path: Path, attack: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import specstyle.production.config_io as config_io

    config_root, evidence_root = _write_roots(tmp_path)
    path = config_root / "context.json"
    if attack == "mode":
        path.chmod(0o640)
    elif attack == "symlink":
        target = config_root / "target.json"
        path.rename(target)
        path.symlink_to(target.name)
    elif attack == "fifo":
        path.unlink()
        os.mkfifo(path, 0o600)
    else:
        target_inode = path.stat().st_ino
        real_read = os.read
        changed = False

        def changing_read(fd: int, size: int) -> bytes:
            nonlocal changed
            chunk = real_read(fd, size)
            if not changed and os.fstat(fd).st_ino == target_inode:
                changed = True
                with path.open("ab") as stream:
                    stream.write(b" ")
            return chunk

        monkeypatch.setattr(config_io.os, "read", changing_read)

    with pytest.raises(InfrastructureError):
        _load(config_root, evidence_root)


def test_accepts_context_document_at_exactly_one_mib(tmp_path: Path) -> None:
    config_root, evidence_root = _write_roots(tmp_path)
    path = config_root / "context.json"
    path.write_bytes(path.read_bytes().ljust(1024 * 1024, b" "))
    path.chmod(0o600)

    assert _load(config_root, evidence_root).schema_version == (
        "specstyle.production.context.v1"
    )


@pytest.mark.parametrize("attack", ("symlink", "fifo", "empty", "oversize", "digest"))
def test_rejects_unsafe_evidence_file_content_or_type(
    tmp_path: Path, attack: str
) -> None:
    config_root, evidence_root = _write_roots(tmp_path)
    path = _evidence_path(config_root, evidence_root)
    if attack == "symlink":
        target = path.with_name("target")
        path.rename(target)
        path.symlink_to(target.name)
    elif attack == "fifo":
        path.unlink()
        os.mkfifo(path, 0o400)
    elif attack == "empty":
        path.chmod(0o600)
        path.write_bytes(b"")
        path.chmod(0o400)
    elif attack == "oversize":
        path.chmod(0o600)
        with path.open("wb") as stream:
            stream.truncate(16 * 1024 * 1024 + 1)
        path.chmod(0o400)
    else:
        path.chmod(0o600)
        path.write_bytes(b"wrong evidence")
        path.chmod(0o400)

    with pytest.raises(InfrastructureError):
        _load(config_root, evidence_root)


@pytest.mark.parametrize("target", ("root", "sha256", "prefix", "file"))
def test_rejects_evidence_owned_by_another_user(
    tmp_path: Path, target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import specstyle.production.context_config as context_config

    config_root, evidence_root = _write_roots(tmp_path)
    path = _evidence_path(config_root, evidence_root)
    labels = {
        "root": "root",
        "sha256": "sha256",
        "prefix": path.parent.name,
        "file": path.name,
    }
    real_stat = context_config._evidence_stat

    def wrong_owner(fd: int, label: str) -> object:
        value = real_stat(fd, label)
        if label != labels[target]:
            return value
        fields = {
            name: getattr(value, name)
            for name in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_uid",
                "st_gid",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        }
        fields["st_uid"] += 1
        return type("EvidenceStat", (), fields)()

    monkeypatch.setattr(context_config, "_evidence_stat", wrong_owner)
    with pytest.raises(InfrastructureError):
        _load(config_root, evidence_root)


def test_accepts_three_evidence_reads_at_exact_size_limits(tmp_path: Path) -> None:
    config_root, evidence_root = _write_roots(tmp_path)
    payload = b"x" * (16 * 1024 * 1024)
    digest = hashlib.sha256(payload).hexdigest()
    document = _read_document(config_root)
    evidence = document["l2_threshold_profile"]["evidence"]
    for key in evidence:
        evidence[key] = digest
    _write_document(config_root, document)
    directory = evidence_root / "sha256" / digest[:2]
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / digest
    path.write_bytes(payload)
    path.chmod(0o400)

    loaded = _load(config_root, evidence_root)
    verified = loaded.l2_threshold_profile.evidence
    assert {
        verified.calibration_dataset_sha256,
        verified.validation_dataset_sha256,
        verified.annotation_protocol_sha256,
    } == {Sha256(digest)}


def _swap_directory_fd(
    caller_fd: int,
    attacker_fd: int,
    entered: threading.Event,
    swapped: threading.Event,
    errors: list[BaseException],
) -> None:
    try:
        assert entered.wait(5)
        os.close(caller_fd)
        os.dup2(attacker_fd, caller_fd)
    except BaseException as exc:
        errors.append(exc)
    finally:
        swapped.set()


@pytest.mark.parametrize("entry", ("fixed", "single"))
def test_config_entry_binds_root_before_concurrent_fd_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    import specstyle.production.config_io as config_io
    from tests.unit.production.test_supply_config import _documents, _write_documents

    original, attacker = tmp_path / "original", tmp_path / "attacker"
    original.mkdir(mode=0o700)
    attacker.mkdir(mode=0o700)
    original_documents = _documents()
    attacker_documents = _documents()
    attacker_documents["models.json"]["models"][0]["family"] = "attacker"
    _write_documents(original, original_documents)
    _write_documents(attacker, attacker_documents)
    caller_fd = os.open(original, os.O_RDONLY | os.O_DIRECTORY)
    attacker_fd = os.open(attacker, os.O_RDONLY | os.O_DIRECTORY)
    entered, swapped = threading.Event(), threading.Event()
    errors: list[BaseException] = []
    real_validate = config_io._validate_root_fd

    def gated_validate(fd: int) -> int:
        entered.set()
        assert swapped.wait(5)
        return real_validate(fd)

    monkeypatch.setattr(config_io, "_validate_root_fd", gated_validate)
    attack = threading.Thread(
        target=_swap_directory_fd,
        args=(caller_fd, attacker_fd, entered, swapped, errors),
    )
    attack.start()
    try:
        if entry == "fixed":
            loaded = config_io.load_fixed_json_documents(caller_fd)["models.json"]
        else:
            loaded = config_io._load_json_document(
                caller_fd, "models.json", 1024 * 1024
            )
    finally:
        attack.join(5)
        os.close(attacker_fd)
        os.close(caller_fd)

    assert not errors
    assert loaded["models"][0]["family"] == "sdxl-production"


def test_evidence_entry_binds_root_before_concurrent_fd_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import specstyle.production.context_config as context_config

    config_root, evidence_root = _write_roots(tmp_path)
    attacker = tmp_path / "attacker-evidence"
    attacker.mkdir(mode=0o700)
    config_fd = os.open(config_root, os.O_RDONLY | os.O_DIRECTORY)
    caller_fd = os.open(evidence_root, os.O_RDONLY | os.O_DIRECTORY)
    attacker_fd = os.open(attacker, os.O_RDONLY | os.O_DIRECTORY)
    entered, swapped = threading.Event(), threading.Event()
    errors: list[BaseException] = []
    real_validate = context_config._validate_evidence_directory

    def gated_validate(fd: int, label: str) -> int:
        if label == "root" and not entered.is_set():
            entered.set()
            assert swapped.wait(5)
        return real_validate(fd, label)

    monkeypatch.setattr(context_config, "_validate_evidence_directory", gated_validate)
    attack = threading.Thread(
        target=_swap_directory_fd,
        args=(caller_fd, attacker_fd, entered, swapped, errors),
    )
    attack.start()
    try:
        loaded = context_config.load_production_context_config(config_fd, caller_fd)
    finally:
        attack.join(5)
        os.close(attacker_fd)
        os.close(caller_fd)
        os.close(config_fd)

    assert not errors
    assert loaded.schema_version == "specstyle.production.context.v1"


def test_context_binds_config_root_before_evidence_validation_fd_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import specstyle.production.context_config as context_config

    original_parent = tmp_path / "original"
    attacker_parent = tmp_path / "attacker"
    original_parent.mkdir()
    attacker_parent.mkdir()
    config_root, evidence_root = _write_roots(original_parent)
    attacker_root, _ = _write_roots(attacker_parent)
    attacker_document = _read_document(attacker_root)
    attacker_document["compiler_pin"]["id"] = "attacker-compiler"
    _write_document(attacker_root, attacker_document)
    config_fd = os.open(config_root, os.O_RDONLY | os.O_DIRECTORY)
    evidence_fd = os.open(evidence_root, os.O_RDONLY | os.O_DIRECTORY)
    attacker_fd = os.open(attacker_root, os.O_RDONLY | os.O_DIRECTORY)
    entered, swapped = threading.Event(), threading.Event()
    errors: list[BaseException] = []
    real_validate = context_config._validate_evidence_directory

    def gated_validate(fd: int, label: str) -> int:
        if label == "root" and not entered.is_set():
            entered.set()
            assert swapped.wait(5)
        return real_validate(fd, label)

    monkeypatch.setattr(context_config, "_validate_evidence_directory", gated_validate)
    attack = threading.Thread(
        target=_swap_directory_fd,
        args=(config_fd, attacker_fd, entered, swapped, errors),
    )
    attack.start()
    try:
        loaded = context_config.load_production_context_config(config_fd, evidence_fd)
    finally:
        attack.join(5)
        os.close(attacker_fd)
        os.close(evidence_fd)
        os.close(config_fd)

    assert not errors
    assert loaded.compiler_pin == ResourcePin("compiler", "r1", Sha256("1" * 64))


def test_context_snapshots_borrowed_roots_consecutively_before_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Callers keep both fds stable through this consecutive, non-atomic snapshot."""
    import specstyle.production._fd_ownership as fd_ownership
    import specstyle.production.config_io as config_io
    import specstyle.production.context_config as context_config

    config_root, evidence_root = _write_roots(tmp_path)
    config_fd = os.open(config_root, os.O_RDONLY | os.O_DIRECTORY)
    evidence_fd = os.open(evidence_root, os.O_RDONLY | os.O_DIRECTORY)
    real_duplicate = fcntl.fcntl
    real_config_validate = config_io._validate_root_fd
    real_evidence_validate = context_config._validate_evidence_directory
    real_open = os.open
    events: list[tuple[str, int]] = []

    def duplicate(fd: int, command: int, minimum: int) -> int:
        events.append(("duplicate", fd))
        return real_duplicate(fd, command, minimum)

    def validate_config(fd: int) -> int:
        events.append(("validate", fd))
        return real_config_validate(fd)

    def validate_evidence(fd: int, label: str) -> int:
        events.append(("validate", fd))
        return real_evidence_validate(fd, label)

    def open_relative(name: str, flags: int, *args: object, **kwargs: object) -> int:
        if kwargs.get("dir_fd") is not None:
            events.append(("open", int(kwargs["dir_fd"])))
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(fd_ownership.fcntl, "fcntl", duplicate)
    monkeypatch.setattr(config_io, "_validate_root_fd", validate_config)
    monkeypatch.setattr(
        context_config, "_validate_evidence_directory", validate_evidence
    )
    monkeypatch.setattr(config_io.os, "open", open_relative)
    try:
        context_config.load_production_context_config(config_fd, evidence_fd)
        os.fstat(config_fd)
        os.fstat(evidence_fd)
    finally:
        os.close(evidence_fd)
        os.close(config_fd)

    assert events[:2] == [
        ("duplicate", config_fd),
        ("duplicate", evidence_fd),
    ]
    assert [event for event in events if event[0] == "duplicate"] == events[:2]


def test_context_second_root_dup_failure_closes_first_and_preserves_callers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import specstyle.production._fd_ownership as fd_ownership
    import specstyle.production.context_config as context_config

    config_root, evidence_root = _write_roots(tmp_path)
    config_fd = os.open(config_root, os.O_RDONLY | os.O_DIRECTORY)
    evidence_fd = os.open(evidence_root, os.O_RDONLY | os.O_DIRECTORY)
    real_duplicate = fcntl.fcntl
    calls: list[int] = []
    first_owned: list[int] = []

    def fail_second(fd: int, command: int, minimum: int) -> int:
        calls.append(fd)
        if len(calls) == 2:
            raise OSError("injected second duplicate failure")
        duplicated = real_duplicate(fd, command, minimum)
        first_owned.append(duplicated)
        return duplicated

    monkeypatch.setattr(fd_ownership.fcntl, "fcntl", fail_second)
    try:
        with pytest.raises(
            InfrastructureError,
            match="production context evidence root unavailable",
        ):
            context_config.load_production_context_config(config_fd, evidence_fd)
        os.fstat(config_fd)
        os.fstat(evidence_fd)
        with pytest.raises(OSError):
            os.fstat(first_owned[0])
    finally:
        os.close(evidence_fd)
        os.close(config_fd)

    assert calls == [config_fd, evidence_fd]


@pytest.mark.parametrize("entry", ("fixed", "single", "evidence"))
def test_root_fd_duplication_failure_is_infrastructure_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    import specstyle.production._fd_ownership as fd_ownership
    import specstyle.production.config_io as config_io
    import specstyle.production.context_config as context_config
    from tests.unit.production.test_supply_config import _documents, _write_documents

    config_root, evidence_root = _write_roots(tmp_path)
    supply_root = tmp_path / "supply"
    supply_root.mkdir(mode=0o700)
    _write_documents(supply_root, _documents())
    supply_fd = os.open(supply_root, os.O_RDONLY | os.O_DIRECTORY)
    config_fd = os.open(config_root, os.O_RDONLY | os.O_DIRECTORY)
    evidence_fd = os.open(evidence_root, os.O_RDONLY | os.O_DIRECTORY)

    def refused_duplicate(_fd: int, _command: int, _minimum: int) -> int:
        raise OSError("injected duplicate failure")

    monkeypatch.setattr(fd_ownership.fcntl, "fcntl", refused_duplicate)
    try:
        with pytest.raises(InfrastructureError):
            if entry == "fixed":
                config_io.load_fixed_json_documents(supply_fd)
            elif entry == "single":
                config_io._load_json_document(supply_fd, "models.json", 1024 * 1024)
            else:
                context_config.load_production_context_config(config_fd, evidence_fd)
    finally:
        os.close(evidence_fd)
        os.close(config_fd)
        os.close(supply_fd)


@pytest.mark.parametrize(
    ("entry", "expected_message"),
    (
        ("fixed", "production config root unavailable"),
        ("single", "production config root unavailable"),
        ("evidence", "production context evidence root unavailable"),
    ),
)
def test_oversized_exact_root_fd_is_unavailable(
    tmp_path: Path, entry: str, expected_message: str
) -> None:
    import specstyle.production.config_io as config_io
    import specstyle.production.context_config as context_config

    config_root = tmp_path / "config"
    config_root.mkdir(mode=0o700)
    config_fd = os.open(config_root, os.O_RDONLY | os.O_DIRECTORY)
    oversized_fd = 10**100
    try:
        with pytest.raises(InfrastructureError, match=expected_message):
            if entry == "fixed":
                config_io.load_fixed_json_documents(oversized_fd)
            elif entry == "single":
                config_io._load_json_document(oversized_fd, "models.json", 1024 * 1024)
            else:
                context_config.load_production_context_config(config_fd, oversized_fd)
    finally:
        os.close(config_fd)


@pytest.mark.parametrize("entry", ("fixed", "single", "evidence"))
def test_root_fd_inputs_are_exact_and_closed_descriptors_are_unavailable(
    tmp_path: Path, entry: str
) -> None:
    import specstyle.production.config_io as config_io
    import specstyle.production.context_config as context_config
    from tests.unit.production.test_supply_config import _documents, _write_documents

    supply_root = tmp_path / "supply"
    supply_root.mkdir(mode=0o700)
    _write_documents(supply_root, _documents())
    context_root = tmp_path / "context"
    context_root.mkdir(mode=0o700)
    config_root, _ = _write_roots(context_root)
    config_fd = os.open(config_root, os.O_RDONLY | os.O_DIRECTORY)
    closed_fd = os.open(supply_root, os.O_RDONLY | os.O_DIRECTORY)
    os.close(closed_fd)
    try:
        with pytest.raises(DomainError):
            if entry == "fixed":
                config_io.load_fixed_json_documents(True)
            elif entry == "single":
                config_io._load_json_document(True, "models.json", 1024 * 1024)
            else:
                context_config.load_production_context_config(config_fd, True)
        with pytest.raises(InfrastructureError):
            if entry == "fixed":
                config_io.load_fixed_json_documents(closed_fd)
            elif entry == "single":
                config_io._load_json_document(closed_fd, "models.json", 1024 * 1024)
            else:
                context_config.load_production_context_config(config_fd, closed_fd)
    finally:
        os.close(config_fd)


@pytest.mark.parametrize("entry", ("fixed", "single"))
def test_config_entry_keeps_caller_fd_and_rejects_untrusted_duplicate(
    tmp_path: Path, entry: str
) -> None:
    import specstyle.production.config_io as config_io
    from tests.unit.production.test_supply_config import _documents, _write_documents

    root = tmp_path / "supply"
    root.mkdir(mode=0o700)
    _write_documents(root, _documents())
    caller_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        if entry == "fixed":
            config_io.load_fixed_json_documents(caller_fd)
        else:
            config_io._load_json_document(caller_fd, "models.json", 1024 * 1024)
        os.fstat(caller_fd)

        root.chmod(0o722)
        with pytest.raises(InfrastructureError):
            if entry == "fixed":
                config_io.load_fixed_json_documents(caller_fd)
            else:
                config_io._load_json_document(caller_fd, "models.json", 1024 * 1024)
        os.fstat(caller_fd)
    finally:
        os.close(caller_fd)


class _CloseFaults:
    def __init__(self, target_caller_fd: int) -> None:
        self.target_caller_fd = target_caller_fd
        self.target_owned: list[int] = []
        self.target_directories: set[int] = set()
        self.close_attempts: list[int] = []
        self.real_fcntl = fcntl.fcntl
        self.real_open = os.open
        self.real_close = os.close

    def duplicate(self, fd: int, command: int, minimum: int) -> int:
        duplicated = self.real_fcntl(fd, command, minimum)
        if fd == self.target_caller_fd:
            self.target_owned.append(duplicated)
            self.target_directories.add(duplicated)
        return duplicated

    def open(self, name: str, flags: int, *args: object, **kwargs: object) -> int:
        opened = self.real_open(name, flags, *args, **kwargs)
        if kwargs.get("dir_fd") in self.target_directories:
            self.target_owned.append(opened)
            if stat.S_ISDIR(os.fstat(opened).st_mode):
                self.target_directories.add(opened)
        return opened

    def close(self, fd: int) -> None:
        targeted = fd in self.target_owned
        if targeted:
            self.close_attempts.append(fd)
        self.real_close(fd)
        if targeted:
            raise OSError(f"injected close failure for fd {fd}")

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import specstyle.production._fd_ownership as fd_ownership

        monkeypatch.setattr(fd_ownership.fcntl, "fcntl", self.duplicate)
        monkeypatch.setattr(fd_ownership.os, "open", self.open)
        monkeypatch.setattr(fd_ownership.os, "close", self.close)


class _CleanupCancellation(BaseException):
    pass


class _SequencedCloseFaults(_CloseFaults):
    def __init__(
        self,
        target_caller_fd: int,
        failures: tuple[BaseException | None, ...],
    ) -> None:
        super().__init__(target_caller_fd)
        self.failures = failures

    def close(self, fd: int) -> None:
        targeted = fd in self.target_owned
        failure: BaseException | None = None
        if targeted:
            attempt = len(self.close_attempts)
            self.close_attempts.append(fd)
            failure = self.failures[attempt]
        self.real_close(fd)
        if failure is not None:
            raise failure


@pytest.mark.parametrize("registration_type", (MemoryError, _CleanupCancellation))
def test_owner_acquisition_closes_fd_and_preserves_registration_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registration_type: type[BaseException],
) -> None:
    import specstyle.production._fd_ownership as fd_ownership

    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    real_duplicate = fcntl.fcntl
    real_close = os.close
    acquired: list[int] = []
    primary = registration_type("injected registration failure")

    def opener() -> int:
        opened = real_duplicate(root_fd, fcntl.F_DUPFD_CLOEXEC, 0)
        acquired.append(opened)
        return opened

    def fail_registration(_owner: object, _fd: int, _label: str) -> int:
        raise primary

    monkeypatch.setattr(
        fd_ownership._OwnedFileDescriptors,
        "_register",
        fail_registration,
        raising=False,
    )
    try:
        with pytest.raises(registration_type) as caught:
            with fd_ownership._OwnedFileDescriptors("test descriptors") as owned:
                owned.acquire(opener, "test root")
        os.fstat(root_fd)
        with pytest.raises(OSError):
            os.fstat(acquired[0])
    finally:
        for fd in acquired:
            try:
                real_close(fd)
            except OSError:
                pass
        real_close(root_fd)

    assert caught.value is primary


def test_owner_acquisition_preserves_registration_error_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import specstyle.production._fd_ownership as fd_ownership

    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    real_duplicate = fcntl.fcntl
    real_close = os.close
    acquired: list[int] = []
    primary = MemoryError("injected registration failure")

    def opener() -> int:
        opened = real_duplicate(root_fd, fcntl.F_DUPFD_CLOEXEC, 0)
        acquired.append(opened)
        return opened

    def fail_registration(_owner: object, _fd: int, _label: str) -> int:
        raise primary

    def fail_cleanup(fd: int) -> None:
        real_close(fd)
        if acquired and fd == acquired[0]:
            raise _CleanupCancellation("injected acquisition cleanup failure")

    monkeypatch.setattr(
        fd_ownership._OwnedFileDescriptors,
        "_register",
        fail_registration,
        raising=False,
    )
    monkeypatch.setattr(fd_ownership.os, "close", fail_cleanup)
    try:
        with pytest.raises(MemoryError) as caught:
            with fd_ownership._OwnedFileDescriptors("test descriptors") as owned:
                owned.acquire(opener, "test root")
        os.fstat(root_fd)
        with pytest.raises(OSError):
            os.fstat(acquired[0])
    finally:
        for fd in acquired:
            try:
                real_close(fd)
            except OSError:
                pass
        real_close(root_fd)

    assert caught.value is primary
    assert len(caught.value.__notes__) == 1
    assert "close" in caught.value.__notes__[0]


@pytest.mark.parametrize(
    ("site", "failure_label"),
    (
        ("fixed_root", "config root"),
        ("single_root", "config root"),
        ("context_config_root", "config root"),
        ("context_evidence_root", "evidence root"),
        ("fixed_config_file", "config file models.json"),
        ("single_config_file", "config file models.json"),
        ("context_config_file", "config file context.json"),
        ("evidence_component", "evidence directory sha256"),
        ("evidence_file", "evidence file"),
    ),
)
def test_loader_acquisition_closes_fd_when_registration_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    site: str,
    failure_label: str,
) -> None:
    import specstyle.production._fd_ownership as fd_ownership
    import specstyle.production.config_io as config_io
    import specstyle.production.context_config as context_config
    from tests.unit.production.test_supply_config import _documents, _write_documents

    config_root, evidence_root = _write_roots(tmp_path)
    supply_root = tmp_path / "supply"
    supply_root.mkdir(mode=0o700)
    _write_documents(supply_root, _documents())
    config_fd = os.open(config_root, os.O_RDONLY | os.O_DIRECTORY)
    evidence_fd = os.open(evidence_root, os.O_RDONLY | os.O_DIRECTORY)
    supply_fd = os.open(supply_root, os.O_RDONLY | os.O_DIRECTORY)
    owner_type = fd_ownership._OwnedFileDescriptors
    real_register = owner_type._register
    failed_fd: list[int] = []
    primary = MemoryError(f"injected registration failure for {failure_label}")

    def fail_registration(owner: object, fd: int, label: str) -> int:
        if label == failure_label and not failed_fd:
            failed_fd.append(fd)
            raise primary
        return real_register(owner, fd, label)

    monkeypatch.setattr(owner_type, "_register", fail_registration)
    failed_fd_closed = False
    try:
        with pytest.raises(MemoryError) as caught:
            if site.startswith("fixed"):
                config_io.load_fixed_json_documents(supply_fd)
            elif site.startswith("single"):
                config_io._load_json_document(supply_fd, "models.json", 1024 * 1024)
            else:
                context_config.load_production_context_config(config_fd, evidence_fd)
        os.fstat(config_fd)
        os.fstat(evidence_fd)
        os.fstat(supply_fd)
        assert caught.value is primary
        assert len(failed_fd) == 1
        try:
            os.fstat(failed_fd[0])
        except OSError:
            failed_fd_closed = True
    finally:
        for fd in failed_fd:
            try:
                os.close(fd)
            except OSError:
                pass
        os.close(supply_fd)
        os.close(evidence_fd)
        os.close(config_fd)

    assert failed_fd_closed


@pytest.mark.parametrize(
    ("entry", "expected_owned"), (("fixed", 4), ("single", 2), ("evidence", 10))
)
def test_success_close_faults_attempt_every_owned_fd_in_reverse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    expected_owned: int,
) -> None:
    import specstyle.production.config_io as config_io
    import specstyle.production.context_config as context_config
    from tests.unit.production.test_supply_config import _documents, _write_documents

    config_root, evidence_root = _write_roots(tmp_path)
    supply_root = tmp_path / "supply"
    supply_root.mkdir(mode=0o700)
    _write_documents(supply_root, _documents())
    supply_fd = os.open(supply_root, os.O_RDONLY | os.O_DIRECTORY)
    config_fd = os.open(config_root, os.O_RDONLY | os.O_DIRECTORY)
    evidence_fd = os.open(evidence_root, os.O_RDONLY | os.O_DIRECTORY)
    target_fd = evidence_fd if entry == "evidence" else supply_fd
    faults = _CloseFaults(target_fd)
    before = len(os.listdir("/dev/fd"))
    faults.install(monkeypatch)
    try:
        with pytest.raises(InfrastructureError, match="close"):
            if entry == "fixed":
                config_io.load_fixed_json_documents(supply_fd)
            elif entry == "single":
                config_io._load_json_document(supply_fd, "models.json", 1024 * 1024)
            else:
                context_config.load_production_context_config(config_fd, evidence_fd)
        os.fstat(target_fd)
        assert len(os.listdir("/dev/fd")) == before
    finally:
        faults.real_close(evidence_fd)
        faults.real_close(config_fd)
        faults.real_close(supply_fd)

    assert len(faults.target_owned) == expected_owned
    assert faults.close_attempts == list(reversed(faults.target_owned))


@pytest.mark.parametrize("primary", ("domain", "infrastructure"))
def test_config_primary_error_survives_root_and_file_close_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, primary: str
) -> None:
    import specstyle.production.config_io as config_io
    from tests.unit.production.test_supply_config import _documents, _write_documents

    root = tmp_path / "supply"
    root.mkdir(mode=0o700)
    _write_documents(root, _documents())
    path = root / "models.json"
    if primary == "domain":
        path.write_bytes(b"{")
        path.chmod(0o600)
        expected_type, expected_text = DomainError, "JSON"
    else:
        path.chmod(0o640)
        expected_type, expected_text = InfrastructureError, "mode refused"
    caller_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    faults = _CloseFaults(caller_fd)
    faults.install(monkeypatch)
    try:
        with pytest.raises(expected_type, match=expected_text) as caught:
            config_io._load_json_document(caller_fd, "models.json", 1024 * 1024)
        os.fstat(caller_fd)
    finally:
        faults.real_close(caller_fd)

    assert faults.close_attempts == list(reversed(faults.target_owned))
    assert len(faults.close_attempts) == 2
    assert len(caught.value.__notes__) == 2
    assert all("close" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("primary", ("domain", "infrastructure"))
def test_context_primary_error_survives_evidence_close_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, primary: str
) -> None:
    import specstyle.production.context_config as context_config

    config_root, evidence_root = _write_roots(tmp_path)
    if primary == "domain":
        document = _read_document(config_root)
        document["unexpected"] = True
        _write_document(config_root, document)
        expected_type, expected_text, expected_owned = DomainError, "document", 1
    else:
        evidence_path = _evidence_path(config_root, evidence_root)
        evidence_path.chmod(0o600)
        evidence_path.write_bytes(b"wrong evidence")
        evidence_path.chmod(0o400)
        expected_type = InfrastructureError
        expected_text, expected_owned = "digest mismatch", 4
    config_fd = os.open(config_root, os.O_RDONLY | os.O_DIRECTORY)
    evidence_fd = os.open(evidence_root, os.O_RDONLY | os.O_DIRECTORY)
    faults = _CloseFaults(evidence_fd)
    faults.install(monkeypatch)
    try:
        with pytest.raises(expected_type, match=expected_text) as caught:
            context_config.load_production_context_config(config_fd, evidence_fd)
        os.fstat(evidence_fd)
    finally:
        faults.real_close(evidence_fd)
        faults.real_close(config_fd)

    assert len(faults.target_owned) == expected_owned
    assert faults.close_attempts == list(reversed(faults.target_owned))
    assert len(caught.value.__notes__) == expected_owned


@pytest.mark.parametrize("cleanup_type", (MemoryError, _CleanupCancellation))
def test_cleanup_non_os_base_exception_propagates_after_all_close_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_type: type[BaseException],
) -> None:
    import specstyle.production.config_io as config_io
    from tests.unit.production.test_supply_config import _documents, _write_documents

    root = tmp_path / "supply"
    root.mkdir(mode=0o700)
    _write_documents(root, _documents())
    caller_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    cleanup_error = cleanup_type("injected cleanup failure")
    faults = _SequencedCloseFaults(caller_fd, (cleanup_error, None))
    faults.install(monkeypatch)
    try:
        with pytest.raises(cleanup_type) as caught:
            config_io._load_json_document(caller_fd, "models.json", 1024 * 1024)
        os.fstat(caller_fd)
    finally:
        faults.real_close(caller_fd)

    assert caught.value is cleanup_error
    assert faults.close_attempts == list(reversed(faults.target_owned))
    assert len(faults.close_attempts) == 2


def test_primary_domain_error_survives_cleanup_memory_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import specstyle.production.config_io as config_io
    from tests.unit.production.test_supply_config import _documents, _write_documents

    root = tmp_path / "supply"
    root.mkdir(mode=0o700)
    _write_documents(root, _documents())
    path = root / "models.json"
    path.write_bytes(b"{")
    path.chmod(0o600)
    caller_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    faults = _SequencedCloseFaults(
        caller_fd,
        (MemoryError("injected cleanup memory error"), OSError("cleanup close")),
    )
    faults.install(monkeypatch)
    try:
        with pytest.raises(DomainError, match="JSON") as caught:
            config_io._load_json_document(caller_fd, "models.json", 1024 * 1024)
        os.fstat(caller_fd)
    finally:
        faults.real_close(caller_fd)

    assert faults.close_attempts == list(reversed(faults.target_owned))
    assert len(faults.close_attempts) == 2
    assert len(caught.value.__notes__) == 2


@pytest.mark.parametrize("primary_type", (MemoryError, _CleanupCancellation))
def test_primary_base_exception_survives_every_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
) -> None:
    import specstyle.production.config_io as config_io
    from tests.unit.production.test_supply_config import _documents, _write_documents

    root = tmp_path / "supply"
    root.mkdir(mode=0o700)
    _write_documents(root, _documents())
    caller_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    primary = primary_type("injected primary failure")
    faults = _SequencedCloseFaults(
        caller_fd,
        (
            MemoryError("injected cleanup memory error"),
            _CleanupCancellation("injected cleanup cancellation"),
        ),
    )

    def raise_primary(_payload: bytes, _filename: str) -> dict[str, Any]:
        raise primary

    faults.install(monkeypatch)
    monkeypatch.setattr(config_io, "_parse_json", raise_primary)
    try:
        with pytest.raises(primary_type) as caught:
            config_io._load_json_document(caller_fd, "models.json", 1024 * 1024)
        os.fstat(caller_fd)
    finally:
        faults.real_close(caller_fd)

    assert caught.value is primary
    assert faults.close_attempts == list(reversed(faults.target_owned))
    assert len(faults.close_attempts) == 2
    assert len(caught.value.__notes__) == 2


def test_repeated_public_loads_keep_fd_count_stable(tmp_path: Path) -> None:
    import specstyle.production.config_io as config_io
    import specstyle.production.context_config as context_config
    from tests.unit.production.test_supply_config import _documents, _write_documents

    config_root, evidence_root = _write_roots(tmp_path)
    supply_root = tmp_path / "supply"
    supply_root.mkdir(mode=0o700)
    _write_documents(supply_root, _documents())
    supply_fd = os.open(supply_root, os.O_RDONLY | os.O_DIRECTORY)
    config_fd = os.open(config_root, os.O_RDONLY | os.O_DIRECTORY)
    evidence_fd = os.open(evidence_root, os.O_RDONLY | os.O_DIRECTORY)
    before = len(os.listdir("/dev/fd"))
    try:
        for _ in range(10):
            config_io.load_fixed_json_documents(supply_fd)
            config_io._load_json_document(supply_fd, "models.json", 1024 * 1024)
            context_config.load_production_context_config(config_fd, evidence_fd)
            assert len(os.listdir("/dev/fd")) == before
    finally:
        os.close(evidence_fd)
        os.close(config_fd)
        os.close(supply_fd)


def _available(value: str) -> TextObservation:
    return TextObservation("AVAILABLE", value, None)


def _factory_environment() -> EnvironmentSnapshot:
    device = DeviceSnapshot(
        0,
        _available("AMD test"),
        IntegerObservation("AVAILABLE", 16 * 1024**3, None),
        _available("gfx1100"),
    )
    return EnvironmentSnapshot(
        "1.0",
        *(_available("host") for _ in range(6)),
        _available("7.2.1"),
        _available("7.2.1"),
        _available("2.8.0"),
        _available("0.39.0"),
        DeviceInventory("AVAILABLE", None, (device,)),
    )


def _descriptor(model_id: str, role: str, character: str) -> ModelDescriptor:
    return ModelDescriptor(
        model_id,
        role,
        character * 40,
        Sha256(character * 64),
        "MIT",
        "APPROVED",
        "sdxl",
    )


def _factory_graph() -> PipelineGraph:
    return PipelineGraph(
        "production",
        _descriptor("base", "base", "d"),
        _descriptor("ip", "ip_adapter", "e"),
        _descriptor("control", "controlnet", "f"),
        None,
        "models",
    )


def test_factory_returns_fresh_exact_compiler_contexts(tmp_path: Path) -> None:
    module = importlib.import_module("specstyle.production.context_config")
    config_root, evidence_root = _write_roots(tmp_path)
    config = _load(config_root, evidence_root)
    environment, graph = _factory_environment(), _factory_graph()
    runtime_hash = hash_environment(environment)
    runtime_pin = ResourcePin("production-runtime", "environment-v1", runtime_hash)

    factory = module.make_production_compiler_context_factory(
        config, environment, graph
    )
    first = factory("clip-preprocess-v1")
    second = factory("clip-preprocess-v1")

    assert first == second
    assert first is not second
    assert first.compiler_pin == config.compiler_pin
    assert first.runtime_capabilities == (
        RuntimeCapability(runtime_pin, "rocm", "7.2.1", "2.8.0", "0.39.0", "float16"),
    )
    assert first.model_capabilities == tuple(
        ModelCapability(
            role,
            ResourcePin(model.model_id, model.revision, model.expected_sha256),
            "canny" if role == "controlnet" else None,
            support.supported_pipelines,
            ("float16",),
            (runtime_hash,),
        )
        for role, model, support in zip(
            ("base", "ip_adapter", "controlnet"),
            (graph.base, graph.ip_adapter, graph.controlnet),
            config.model_support,
            strict=True,
        )
    )
    encoder_pin = ResourcePin(
        graph.ip_adapter.model_id,
        graph.ip_adapter.revision,
        graph.ip_adapter.expected_sha256,
    )
    assert first.encoder_capabilities == (
        EncoderCapability(
            encoder_pin,
            "clip-preprocess-v1",
            "hidden_states[-2]",
            "median_cosine_patch_mean_std_v1",
            (runtime_hash,),
        ),
    )
    profile = config.l2_threshold_profile
    assert first.threshold_profiles == (
        ThresholdProfileCapability(
            profile.pin,
            profile.logical_name,
            "l2",
            profile.status,
            profile.style_pack_id,
            "product_instance",
            encoder_pin,
            None,
            (profile.metric,),
            profile.evidence.calibration_dataset_sha256,
            profile.evidence.validation_dataset_sha256,
            profile.evidence.annotation_protocol_sha256,
        ),
    )
    assert first.strength_mappings == (config.strength_mapping,)
    assert first.output_profile_capabilities == (config.output_profile,)
    assert first.rule_catalogs == (config.rule_catalog,)
    assert first.l3_plugins == ()
    for attribute in (
        "runtime_capabilities",
        "model_capabilities",
        "encoder_capabilities",
        "strength_mappings",
        "output_profile_capabilities",
        "rule_catalogs",
        "threshold_profiles",
    ):
        first_values = getattr(first, attribute)
        second_values = getattr(second, attribute)
        assert first_values is not second_values
        assert first_values[0] is not second_values[0]
    assert first.model_capabilities[0].pin.sha256 is not (
        second.model_capabilities[0].pin.sha256
    )
    assert first.model_capabilities[0].supported_pipelines is not (
        second.model_capabilities[0].supported_pipelines
    )
    assert (
        first.strength_mappings[0].entries[0]
        is not (second.strength_mappings[0].entries[0])
    )
    assert first.rule_catalogs[0].rules[0] is not second.rule_catalogs[0].rules[0]
    assert (
        first.threshold_profiles[0].metrics[0]
        is not (second.threshold_profiles[0].metrics[0])
    )


@pytest.mark.parametrize(
    "preprocessing_version", (None, 1, True, " padded", "line\nbreak")
)
def test_factory_rejects_nonexact_or_unsafe_preprocessing_version(
    tmp_path: Path, preprocessing_version: object
) -> None:
    module = importlib.import_module("specstyle.production.context_config")
    config_root, evidence_root = _write_roots(tmp_path)
    factory = module.make_production_compiler_context_factory(
        _load(config_root, evidence_root), _factory_environment(), _factory_graph()
    )

    with pytest.raises(DomainError):
        factory(preprocessing_version)


@pytest.mark.parametrize("invalid", ("config", "environment", "graph"))
def test_factory_rejects_invalid_exact_input_contracts(
    tmp_path: Path, invalid: str
) -> None:
    module = importlib.import_module("specstyle.production.context_config")
    config_root, evidence_root = _write_roots(tmp_path)
    values = [
        _load(config_root, evidence_root),
        _factory_environment(),
        _factory_graph(),
    ]
    values[("config", "environment", "graph").index(invalid)] = object()

    with pytest.raises(DomainError):
        module.make_production_compiler_context_factory(*values)


@pytest.mark.parametrize(
    "mutation",
    (
        "profile",
        "preview",
        "base_role",
        "rocm",
        "hip",
        "pytorch",
        "diffusers",
        "devices",
        "device_detail",
    ),
)
def test_factory_rejects_incompatible_runtime_or_graph(
    tmp_path: Path, mutation: str
) -> None:
    module = importlib.import_module("specstyle.production.context_config")
    config_root, evidence_root = _write_roots(tmp_path)
    environment, graph = _factory_environment(), _factory_graph()
    unavailable = TextObservation("UNAVAILABLE", None, "NOT_REPORTED")
    if mutation == "profile":
        graph = replace(graph, profile="preview")
    elif mutation == "preview":
        graph = replace(graph, preview_adapter=graph.base)
    elif mutation == "base_role":
        graph = replace(graph, base=graph.ip_adapter)
    elif mutation == "rocm":
        environment = replace(environment, rocm_version=unavailable)
    elif mutation == "hip":
        environment = replace(environment, hip_version=unavailable)
    elif mutation == "pytorch":
        environment = replace(environment, pytorch_version=unavailable)
    elif mutation == "diffusers":
        environment = replace(environment, diffusers_version=unavailable)
    elif mutation == "devices":
        environment = replace(
            environment,
            hip_devices=DeviceInventory("UNAVAILABLE", "NO_DEVICE", ()),
        )
    else:
        device = environment.hip_devices.devices[0]
        environment = replace(
            environment,
            hip_devices=replace(
                environment.hip_devices,
                devices=(replace(device, name=unavailable),),
            ),
        )

    with pytest.raises(DomainError):
        module.make_production_compiler_context_factory(
            _load(config_root, evidence_root), environment, graph
        )


def _raw_for_factory(
    context: object,
    environment: EnvironmentSnapshot,
    graph: PipelineGraph,
    preprocessing_version: str,
) -> StyleSpecV1:
    from tests.unit.spec.test_compiler import raw_spec

    raw = raw_spec().model_dump(mode="python")
    raw["runtime"] = {
        "backend": "rocm",
        "rocm_version": environment.rocm_version.value,
        "torch_version": environment.pytorch_version.value,
        "diffusers_version": environment.diffusers_version.value,
        "dtype": "float16",
    }
    raw["models"] = {
        "base": {
            "id": graph.base.model_id,
            "revision": graph.base.revision,
            "sha256": graph.base.expected_sha256.value,
        },
        "ip_adapter": {
            "id": graph.ip_adapter.model_id,
            "revision": graph.ip_adapter.revision,
            "sha256": graph.ip_adapter.expected_sha256.value,
        },
        "controlnet": {
            "type": "canny",
            "id": graph.controlnet.model_id,
            "revision": graph.controlnet.revision,
            "sha256": graph.controlnet.expected_sha256.value,
        },
    }
    raw["verification"]["ruleset_version"] = context.rule_catalogs[0].ruleset_version
    raw["verification"]["l2"] = {
        "encoder_id": graph.ip_adapter.model_id,
        "encoder_revision": graph.ip_adapter.revision,
        "preprocessing_version": preprocessing_version,
        "threshold_profile": {
            "id": context.threshold_profiles[0].pin.id,
            "revision": context.threshold_profiles[0].pin.revision,
            "sha256": context.threshold_profiles[0].pin.sha256.value,
        },
    }
    return StyleSpecV1.model_validate(raw)


def test_factory_context_compiles_only_exact_runtime_graph_and_pins(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("specstyle.production.context_config")
    config_root, evidence_root = _write_roots(tmp_path)
    environment, graph = _factory_environment(), _factory_graph()
    preprocessing_version = "clip-preprocess-v1"
    context = module.make_production_compiler_context_factory(
        _load(config_root, evidence_root), environment, graph
    )(preprocessing_version)
    raw = _raw_for_factory(context, environment, graph, preprocessing_version)

    compiled = compile_style_spec(raw, context)

    assert compiled.preview_graphs[0].base_model.pin.id == graph.base.model_id
    assert compiled.production_graphs[0].controlnet.pin.id == graph.controlnet.model_id
    assert (
        compiled.l2_encoder.pin,
        compiled.l2_encoder.preprocessing_version,
        compiled.l2_encoder.layer,
        compiled.l2_encoder.distance_function,
    ) == (
        context.encoder_capabilities[0].pin,
        context.encoder_capabilities[0].preprocessing_version,
        context.encoder_capabilities[0].layer,
        context.encoder_capabilities[0].distance_function,
    )
    assert compiled.verification_plans[0].l3_status == "NOT_APPLICABLE"
    assert compiled.verification_plans[0].l3_reason == "NO_L3_CONFIG"


@pytest.mark.parametrize("mismatch", ("runtime", "model", "threshold"))
def test_factory_context_has_no_pin_or_version_fallback(
    tmp_path: Path, mismatch: str
) -> None:
    module = importlib.import_module("specstyle.production.context_config")
    config_root, evidence_root = _write_roots(tmp_path)
    environment, graph = _factory_environment(), _factory_graph()
    context = module.make_production_compiler_context_factory(
        _load(config_root, evidence_root), environment, graph
    )("clip-preprocess-v1")
    raw = _raw_for_factory(context, environment, graph, "clip-preprocess-v1")
    primitive = raw.model_dump(mode="python")
    if mismatch == "runtime":
        primitive["runtime"]["rocm_version"] = "other"
    elif mismatch == "model":
        primitive["models"]["base"]["sha256"] = "0" * 64
    else:
        primitive["verification"]["l2"]["threshold_profile"]["sha256"] = "0" * 64

    with pytest.raises(DomainError):
        compile_style_spec(StyleSpecV1.model_validate(primitive), context)


@pytest.mark.parametrize(
    ("status", "expected"),
    (
        ("DRAFT", RuleStatus.UNVERIFIABLE),
        ("CALIBRATED", RuleStatus.UNVERIFIABLE),
        ("VALIDATED", None),
    ),
)
def test_threshold_status_controls_l2_execution_without_encoder_call(
    tmp_path: Path, status: str, expected: RuleStatus | None
) -> None:
    from specstyle.verification.production import _metric_without_execution

    module = importlib.import_module("specstyle.production.context_config")
    config_root, evidence_root = _write_roots(tmp_path, status=status)
    environment, graph = _factory_environment(), _factory_graph()
    context = module.make_production_compiler_context_factory(
        _load(config_root, evidence_root), environment, graph
    )("clip-preprocess-v1")
    compiled = compile_style_spec(
        _raw_for_factory(context, environment, graph, "clip-preprocess-v1"), context
    )
    rule = next(
        item
        for item in compiled.verification_plans[0].rules
        if item.metric_id == Identifier("reference_style_statistics_similarity")
    )

    result = _metric_without_execution(rule, ArtifactId("artifact"))

    assert (None if result is None else result.status) is expected
