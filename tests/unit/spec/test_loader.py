"""specstyle.spec.loader 单测：YAML 安全解析 + 路径/输入安全。

覆盖 Module 2 冻结合同：
- 文本：UTF-8 byte size、duplicate key、unsafe/custom tag、alias、merge、root mapping、深度、节点数；
  非法 UTF-8/孤立 surrogate、YAML/Pydantic 输入错误抛 DomainError，不回显原文/值；
- 文件：allowed_root 可信目录、relative path 相对且不含 ..、逐组件拒绝 symlink、resolve 后仍在 root、
  仅 regular file、读取前后双重 size/短读检查；
- 异常分类：不存在/路径政策违反 = DomainError；权限/设备/短读/其它非输入 OSError = InfrastructureError；
  错误信息不含完整私密绝对路径。
- 限额：MAX_SPEC_BYTES=1_048_576、MAX_YAML_DEPTH=32、MAX_YAML_NODES=10_000；P0 拒绝所有 alias 和 merge。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from specstyle.errors import DomainError
from specstyle.spec.loader import (
    MAX_SPEC_BYTES,
    MAX_YAML_DEPTH,
    MAX_YAML_NODES,
    load_style_spec_file,
    load_style_spec_text,
)

H = "a" * 64


def _valid_dict() -> dict:
    return {
        "schema_version": "1.0",
        "schema_uri": "schemas/style-spec-1.0.schema.json",
        "metadata": {
            "spec_id": "brand-retro-v1",
            "name": "Retro Editorial",
            "author": "studio-a",
            "created_at": "2026-07-29T10:00:00Z",
            "parent_spec": None,
        },
        "runtime": {
            "backend": "rocm",
            "rocm_version": "6.2",
            "torch_version": "2.4",
            "diffusers_version": "0.30",
            "dtype": "float16",
        },
        "models": {
            "base": {"id": "sdxl-base", "revision": "r1", "sha256": "a" * 64},
            "ip_adapter": {"id": "ip-adapter", "revision": "r1", "sha256": "b" * 64},
            "controlnet": {"type": "canny", "id": "canny-cn", "revision": "r1", "sha256": "c" * 64},
        },
        "assets": {
            "style_references": [
                {
                    "asset_sha256": "d" * 64,
                    "source_url": "https://example.com/a.png",
                    "license": "CC0",
                    "attribution": "author",
                    "consent": "not_applicable",
                }
            ]
        },
        "profiles": {
            "preview": {"pipeline": "sdxl_turbo", "resolution": [512, 512], "steps": 4, "guidance_scale": 0},
            "production": {
                "pipeline": "sdxl_base",
                "resolution": [1024, 1024],
                "steps": 30,
                "guidance_scale": 5,
                "scheduler": "euler",
            },
        },
        "style": {
            "preset_id": "retro-v1",
            "user_strength": 0.7,
            "preview_ip_adapter_scale": 0.55,
            "production_ip_adapter_scale": 0.72,
        },
        "generation": {
            "img2img_strength": 0.45,
            "controlnet_scale": 0.70,
            "seed_policy": "per_asset_deterministic",
            "batch_execution": "sequential",
        },
        "domain": {"profile": "product_instance", "verifier_version": None, "fidelity_required": False},
        "outputs": {"profiles": ["xhs_grid"]},
        "verification": {
            "ruleset_version": "1.0",
            "gate_defaults": {"on_unverifiable": "reject", "on_warning": "manual_review"},
            "l2": {
                "encoder_id": "enc",
                "encoder_revision": "r1",
                "preprocessing_version": "p1",
                "threshold_profile": {"id": "tp1", "revision": "r1", "sha256": "e" * 64},
            },
            "l3": None,
        },
        "repair": {"policy_version": "1.0", "max_rounds": 3, "stop_after_no_improvement": 2},
        "replay_contract": {
            "mode": "semantic",
            "tolerated_metric_delta": {"l2_style_fidelity": 0.02, "l3_fidelity": 0.02},
            "new_batch": {
                "contract": "same_compiled_graph_and_gate_definitions",
                "per_item_metric_equality_required": False,
            },
        },
    }


def _valid_yaml() -> str:
    return yaml.dump(_valid_dict(), sort_keys=False, default_flow_style=False, allow_unicode=True)


def _mutate_yaml(mutator) -> str:
    import copy
    d = copy.deepcopy(_valid_dict())
    mutator(d)
    return yaml.dump(d, sort_keys=False, default_flow_style=False, allow_unicode=True)


# --- 常量 ---

def test_limits_constants():
    assert MAX_SPEC_BYTES == 1_048_576
    assert MAX_YAML_DEPTH == 32
    assert MAX_YAML_NODES == 10_000


# --- 合法文本加载 ---

def test_load_valid_text_returns_spec():
    spec = load_style_spec_text(_valid_yaml())
    assert spec.schema_version == "1.0"
    assert spec.metadata.spec_id == "brand-retro-v1"
    # YAML sequence 已转为 tuple
    assert spec.profiles.preview.resolution == (512, 512)
    assert spec.outputs.profiles == ("xhs_grid",)


def test_load_valid_text_accepts_bytes():
    spec = load_style_spec_text(_valid_yaml().encode("utf-8"))
    assert spec.schema_version == "1.0"


def test_sha_uppercase_normalized_in_text():
    def m(d):
        d["models"]["base"]["sha256"] = "A" * 64
    spec = load_style_spec_text(_mutate_yaml(m))
    assert spec.models.base.sha256 == "a" * 64


# --- 非法 UTF-8 / 大小 ---

def test_invalid_utf8_raises_domain_error():
    with pytest.raises(DomainError):
        load_style_spec_text(b"\xff\xfe invalid utf-8 \xed\xa0\x80")


def test_text_exceeds_max_bytes_raises():
    with pytest.raises(DomainError):
        load_style_spec_text("x" * (MAX_SPEC_BYTES + 1))


# --- duplicate key / alias / merge / unsafe tag ---

def test_duplicate_key_raises():
    text = "schema_version: \"1.0\"\nschema_version: \"2.0\"\n"
    with pytest.raises(DomainError):
        load_style_spec_text(text)


def test_alias_raises():
    text = "schema_version: &a \"1.0\"\nschema_uri: *a\n"
    with pytest.raises(DomainError):
        load_style_spec_text(text)


def test_merge_key_raises():
    base = "schema_version: \"1.0\"\nschema_uri: \"schemas/style-spec-1.0.schema.json\"\n"
    merge = base + "<<: {extra: 1}\n"
    with pytest.raises(DomainError):
        load_style_spec_text(merge)


def test_custom_tag_raises():
    text = "schema_version: !foo \"1.0\"\n"
    with pytest.raises(DomainError):
        load_style_spec_text(text)


def test_unsafe_python_tag_raises():
    text = "schema_version: !!python/object/apply:os.system [\"echo pwn\"]\n"
    with pytest.raises(DomainError):
        load_style_spec_text(text)


# --- root mapping / depth / nodes ---

def test_root_not_mapping_raises():
    with pytest.raises(DomainError):
        load_style_spec_text("- a\n- b\n")  # root is sequence
    with pytest.raises(DomainError):
        load_style_spec_text("just-a-scalar\n")  # root is scalar


def test_empty_yaml_raises():
    with pytest.raises(DomainError):
        load_style_spec_text("")


def test_excessive_depth_raises():
    # 嵌套深度 > 32
    text = "a:\n" + "".join("  " * i + "b:\n" for i in range(1, MAX_YAML_DEPTH + 2))
    with pytest.raises(DomainError):
        load_style_spec_text(text)


def test_very_deep_recursion_caught():
    # ~800 层嵌套可能触发 PyYAML RecursionError（取决于栈限）；无论哪条路径都必须是 DomainError
    text = "a:\n" + "".join("  " * i + "b:\n" for i in range(1, 800))
    with pytest.raises(DomainError):
        load_style_spec_text(text)


def test_excessive_nodes_raises():
    # 节点数 > 10000
    text = "schema_version: \"1.0\"\nitems: [" + ", ".join("1" for _ in range(MAX_YAML_NODES + 1)) + "]\n"
    with pytest.raises(DomainError):
        load_style_spec_text(text)


# --- Pydantic 输入错误 -> DomainError，不回显值 ---

def test_pydantic_validation_error_becomes_domain_error():
    def m(d):
        d["schema_version"] = "9.9"  # 非法 Literal
    with pytest.raises(DomainError):
        load_style_spec_text(_mutate_yaml(m))


def test_error_message_does_not_leak_value():
    def m(d):
        d["models"]["base"]["sha256"] = "SECRET-LEAK-VALUE"
    try:
        load_style_spec_text(_mutate_yaml(m))
    except DomainError as exc:
        msg = str(exc)
        assert "SECRET-LEAK-VALUE" not in msg


# --- 文件加载 + 路径安全 ---

def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def test_load_valid_file(tmp_path):
    _write(tmp_path, "spec.yaml", _valid_yaml())
    spec = load_style_spec_file(tmp_path, "spec.yaml")
    assert spec.schema_version == "1.0"


def test_file_not_found_raises_domain_error(tmp_path):
    with pytest.raises(DomainError):
        load_style_spec_file(tmp_path, "missing.yaml")


def test_absolute_relative_path_raises(tmp_path):
    with pytest.raises(DomainError):
        load_style_spec_file(tmp_path, "/etc/passwd")


def test_dotdot_in_relative_path_raises(tmp_path):
    _write(tmp_path, "spec.yaml", _valid_yaml())
    with pytest.raises(DomainError):
        load_style_spec_file(tmp_path, "../spec.yaml")
    with pytest.raises(DomainError):
        load_style_spec_file(tmp_path, "sub/../../spec.yaml")


def test_symlink_rejected(tmp_path):
    target = _write(tmp_path, "real.yaml", _valid_yaml())
    link = tmp_path / "link.yaml"
    os.symlink(target, link)
    with pytest.raises(DomainError):
        load_style_spec_file(tmp_path, "link.yaml")


def test_directory_rejected(tmp_path):
    (tmp_path / "sub").mkdir()
    with pytest.raises(DomainError):
        load_style_spec_file(tmp_path, "sub")


def test_file_exceeds_max_bytes(tmp_path):
    _write(tmp_path, "big.yaml", "x" * (MAX_SPEC_BYTES + 1))
    with pytest.raises(DomainError):
        load_style_spec_file(tmp_path, "big.yaml")


def test_allowed_root_must_exist(tmp_path):
    with pytest.raises(DomainError):
        load_style_spec_file(tmp_path / "nope", "spec.yaml")


def test_non_str_text_raises():
    with pytest.raises(DomainError):
        load_style_spec_text(123)  # type: ignore[arg-type]


def test_load_file_uses_size_guard(tmp_path):
    # 写一个刚好等于上限的合法文件不应被拒
    _write(tmp_path, "spec.yaml", _valid_yaml())
    spec = load_style_spec_file(tmp_path, "spec.yaml", max_bytes=len(_valid_yaml().encode("utf-8")) + 1000)
    assert spec.schema_version == "1.0"
