"""specstyle 领域标识符与哈希值对象。

frozen+slots、hashable、无 __dict__；具体类型隔离；to/from_primitive 精确 round-trip。

- Identifier / JobId / AssetId / AttemptId / ArtifactId / DecisionId / RuleId：
  仅接受 re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value, re.ASCII)，
  非法值抛 DomainError，不 strip、不改大小写。
- Sha256：严格 64 hex，规范为小写；不做 hash 计算，仅校验与存储。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from specstyle.errors import DomainError

_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", re.ASCII)
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def _validate_id(value: object) -> str:
    """校验 ID：必须是匹配模式的 str；不 strip、不改大小写。"""
    if not isinstance(value, str) or _ID_PATTERN.fullmatch(value) is None:
        raise DomainError(f"invalid identifier: {value!r}")
    return value


def _validate_sha256(value: object) -> str:
    """校验 64 hex 并规范为小写。不做 hash 计算。"""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise DomainError(f"invalid sha256: {value!r}")
    return value.lower()


@dataclass(frozen=True, slots=True)
class Identifier:
    """受控本地资产 ID 基类（非原文件名公开标识）。"""

    value: str

    def __post_init__(self) -> None:
        _validate_id(self.value)

    def __str__(self) -> str:
        return self.value

    def to_primitive(self) -> str:
        return self.value

    @classmethod
    def from_primitive(cls, value: object) -> Identifier:
        if not isinstance(value, str):
            raise DomainError(f"identifier primitive must be str: {value!r}")
        return cls(value)


class JobId(Identifier):
    """批量任务 ID。"""

    __slots__ = ()


class AssetId(Identifier):
    """输入资产 ID。"""

    __slots__ = ()


class AttemptId(Identifier):
    """单次生成 attempt ID。"""

    __slots__ = ()


class ArtifactId(Identifier):
    """生成产物 ID。"""

    __slots__ = ()


class DecisionId(Identifier):
    """Repair 决策 ID。"""

    __slots__ = ()


class RuleId(Identifier):
    """验证规则 ID。"""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class Sha256:
    """内容哈希值对象：严格 64 hex，小写规范。不做 hash 计算，仅校验与存储。"""

    value: str

    def __post_init__(self) -> None:
        normalized = _validate_sha256(self.value)
        # frozen 实例需用 object.__setattr__ 规范化
        object.__setattr__(self, "value", normalized)

    def __str__(self) -> str:
        return self.value

    def to_primitive(self) -> str:
        return self.value

    @classmethod
    def from_primitive(cls, value: object) -> Sha256:
        if not isinstance(value, str):
            raise DomainError(f"sha256 primitive must be str: {value!r}")
        return cls(value)
