"""SPEC-004 StyleSpec migration registry（contracts §14）。

显式有向 edge；copy-on-write；失败不修改 source；无 identity / 无降级 / 无跨 major。
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import ValidationError

from specstyle.errors import DomainError
from specstyle.spec.diff import SpecDiff, semantic_diff
from specstyle.spec.models import (
    DEFAULT_ENVIRONMENT_POLICY_V11,
    LEGACY_STRENGTH_MAPPING_VERSION,
    SCHEMA_URI_V11,
    StyleSpec,
    StyleSpecV1,
    StyleSpecV11,
)


@dataclass(frozen=True, slots=True)
class MigrationResult:
    source_version: str
    target_version: str
    path: tuple[str, ...]
    target_spec: StyleSpecV11
    diff: SpecDiff


def _version_of(spec: StyleSpec) -> str:
    if type(spec) is StyleSpecV1:
        return "1.0"
    if type(spec) is StyleSpecV11:
        return "1.1"
    raise DomainError("unsupported style spec migration") from None


def _list_to_tuple(obj: object) -> object:
    """model_dump JSON 模式产出 list；strict model 要求 tuple（与 loader 一致）。"""
    if type(obj) is list:
        return tuple(_list_to_tuple(x) for x in obj)
    if type(obj) is dict:
        return {k: _list_to_tuple(v) for k, v in obj.items()}
    return obj


def _migrate_1_0_to_1_1(source: StyleSpecV1) -> StyleSpecV11:
    primitive = _list_to_tuple(copy.deepcopy(source.model_dump(mode="json")))
    if type(primitive) is not dict:
        raise DomainError("unsupported style spec migration") from None
    primitive["schema_version"] = "1.1"
    primitive["schema_uri"] = SCHEMA_URI_V11
    style = primitive.get("style")
    if type(style) is not dict:
        raise DomainError("unsupported style spec migration") from None
    style = dict(style)
    style["strength_mapping_version"] = LEGACY_STRENGTH_MAPPING_VERSION
    primitive["style"] = style
    replay = primitive.get("replay_contract")
    if type(replay) is not dict:
        raise DomainError("unsupported style spec migration") from None
    replay = dict(replay)
    replay["environment_policy"] = DEFAULT_ENVIRONMENT_POLICY_V11
    primitive["replay_contract"] = replay
    try:
        return StyleSpecV11(**primitive)
    except ValidationError:
        raise DomainError("style spec migration validation failed") from None


_EDGE: dict[tuple[str, str], Callable[[StyleSpec], StyleSpec]] = {
    ("1.0", "1.1"): lambda s: _migrate_1_0_to_1_1(s),  # type: ignore[arg-type]
}


def _shortest_path(source: str, target: str) -> tuple[str, ...] | None:
    if source == target:
        return None
    # 单层 BFS 覆盖未来多 edge；当前仅 1.0→1.1。
    frontier: list[tuple[str, ...]] = [(source,)]
    seen = {source}
    while frontier:
        path = frontier.pop(0)
        last = path[-1]
        for (src, dst), _ in _EDGE.items():
            if src != last or dst in seen:
                continue
            nxt = path + (dst,)
            if dst == target:
                return nxt
            seen.add(dst)
            frontier.append(nxt)
    return None


def migrate_style_spec(source: StyleSpec, target_version: str, /) -> MigrationResult:
    if type(source) not in (StyleSpecV1, StyleSpecV11):
        raise DomainError("unsupported style spec migration") from None
    if type(target_version) is not str:
        raise DomainError("unsupported style spec migration") from None
    source_version = _version_of(source)
    if source_version == target_version:
        raise DomainError("unsupported style spec migration") from None
    path = _shortest_path(source_version, target_version)
    if path is None or len(path) < 2:
        raise DomainError("unsupported style spec migration") from None

    current: StyleSpec = source
    # 保留源引用；每一步只消费 dump 副本
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        fn = _EDGE.get(edge)
        if fn is None:
            raise DomainError("unsupported style spec migration") from None
        try:
            current = fn(current)
        except DomainError:
            raise
        except Exception:
            raise DomainError("style spec migration validation failed") from None
        if _version_of(current) != path[i + 1]:
            raise DomainError("style spec migration validation failed") from None

    if type(current) is not StyleSpecV11:
        raise DomainError("style spec migration validation failed") from None
    diff = semantic_diff(source, current)
    return MigrationResult(
        source_version,
        target_version,
        path,
        current,
        diff,
    )
