"""specstyle.spec.schema — Style Spec JSON Schema Draft 2020-12 生成。

build_style_spec_schema 每次返回互不污染的新对象；$schema 固定 Draft 2020-12 官方 URI，
$id 等于实例 schema_uri；dump 使用 sort_keys + 紧凑分隔 + ensure_ascii=False 的 canonical JSON。
"""

from __future__ import annotations

import json

from pydantic import TypeAdapter

from specstyle.spec.models import StyleSpecV1

_DRAFT_URI = "https://json-schema.org/draft/2020-12/schema"
_SCHEMA_ID = "schemas/style-spec-1.0.schema.json"


def build_style_spec_schema() -> dict[str, object]:
    schema: dict[str, object] = TypeAdapter(StyleSpecV1).json_schema()
    schema["$schema"] = _DRAFT_URI
    schema["$id"] = _SCHEMA_ID
    return schema


def dump_style_spec_schema() -> str:
    return json.dumps(
        build_style_spec_schema(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
