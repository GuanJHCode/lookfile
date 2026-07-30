"""specstyle.spec.schema 单测：JSON Schema Draft 2020-12 生成。

覆盖 Module 2 冻结合同：
- build_style_spec_schema 返回 Draft 2020-12 dict，$schema 官方 URI，$id = schema_uri；
- 14 顶层字段全部在 required 与 properties；
- 每次 build 返回互不污染的新对象；
- SHA/ID/resolution/scale 约束出现在 Schema（Pydantic 用 $defs，故用 dump 子串校验）；
- dump_style_spec_schema 返回 canonical JSON（sort_keys、紧凑、ensure_ascii=False）。
"""
from __future__ import annotations

import json

from specstyle.spec.schema import build_style_spec_schema, dump_style_spec_schema

DRAFT_URI = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_ID = "schemas/style-spec-1.0.schema.json"

TOP_LEVEL = {
    "schema_version", "schema_uri", "metadata", "runtime", "models", "assets",
    "profiles", "style", "generation", "domain", "outputs", "verification",
    "repair", "replay_contract",
}


def test_schema_draft_and_id():
    schema = build_style_spec_schema()
    assert schema["$schema"] == DRAFT_URI
    assert schema["$id"] == SCHEMA_ID


def test_schema_required_all_top_level():
    schema = build_style_spec_schema()
    assert set(schema["required"]) == TOP_LEVEL


def test_schema_has_properties_for_all_top_level():
    schema = build_style_spec_schema()
    assert set(schema["properties"].keys()) == TOP_LEVEL


def test_schema_build_returns_independent_objects():
    a = build_style_spec_schema()
    b = build_style_spec_schema()
    assert a is not b
    assert a["properties"] is not b["properties"]
    a["properties"]["bogus"] = {}
    assert "bogus" not in b["properties"]


def test_schema_sha256_pattern_and_length():
    dumped = dump_style_spec_schema()
    # SHA 字段：64-hex pattern + minLength/maxLength 64
    assert "[0-9a-fA-F]" in dumped
    assert '"minLength":64' in dumped
    assert '"maxLength":64' in dumped


def test_schema_idlike_pattern_and_length():
    dumped = dump_style_spec_schema()
    # ID-like：minLength 1, maxLength 128
    assert '"minLength":1' in dumped
    assert '"maxLength":128' in dumped


def test_schema_resolution_multiple_of_and_range():
    dumped = dump_style_spec_schema()
    # resolution member: ge=64, le=4096, multiple_of=8
    assert '"minimum":64' in dumped
    assert '"maximum":4096' in dumped
    assert '"multipleOf":8' in dumped


def test_schema_scale_range():
    dumped = dump_style_spec_schema()
    # scale [0,1]: minimum 0, maximum 1
    assert '"minimum":0' in dumped
    assert '"maximum":1' in dumped


def test_schema_per_item_metric_equality_required_const_false():
    dumped = dump_style_spec_schema()
    # Literal[False] -> const:false（收紧，非宽松 boolean）
    assert '"const":false' in dumped


def test_schema_extra_forbid():
    schema = build_style_spec_schema()
    assert schema.get("additionalProperties") is False


def test_dump_canonical_json_string():
    dumped = dump_style_spec_schema()
    assert isinstance(dumped, str)
    reparsed = json.loads(dumped)
    assert reparsed["$schema"] == DRAFT_URI
    assert json.dumps(reparsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False) == dumped


def test_dump_is_idempotent():
    assert dump_style_spec_schema() == dump_style_spec_schema()
