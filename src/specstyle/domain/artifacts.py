"""specstyle 领域资产/产物引用值对象。

AssetRef(asset_id, sha256) 与 ArtifactRef(artifact_id, sha256)：immutable、严格类型、
精确 mapping round-trip。不含 path，不做 IO、解码或 hash 计算（hash 由外部计算后传入）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from specstyle.errors import DomainError
from specstyle.domain.identifiers import AssetId, ArtifactId, Sha256

_ASSET_KEYS = frozenset({"asset_id", "sha256"})
_ARTIFACT_KEYS = frozenset({"artifact_id", "sha256"})


@dataclass(frozen=True, slots=True)
class AssetRef:
    """对一项输入资产的不可变引用（资产 ID + 内容哈希）。不含 path。"""

    asset_id: AssetId
    sha256: Sha256

    def __post_init__(self) -> None:
        if not isinstance(self.asset_id, AssetId):
            raise DomainError(f"asset_id must be AssetId: {self.asset_id!r}")
        if not isinstance(self.sha256, Sha256):
            raise DomainError(f"sha256 must be Sha256: {self.sha256!r}")

    def to_primitive(self) -> dict:
        return {
            "asset_id": self.asset_id.to_primitive(),
            "sha256": self.sha256.to_primitive(),
        }

    @classmethod
    def from_primitive(cls, data: object) -> AssetRef:
        if not isinstance(data, Mapping) or set(data) != _ASSET_KEYS:
            raise DomainError(
                f"AssetRef primitive must be mapping with keys {sorted(_ASSET_KEYS)}: {data!r}"
            )
        return cls(
            AssetId.from_primitive(data["asset_id"]),
            Sha256.from_primitive(data["sha256"]),
        )


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """对一项生成产物的不可变引用（产物 ID + 内容哈希）。不含 path。"""

    artifact_id: ArtifactId
    sha256: Sha256

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, ArtifactId):
            raise DomainError(f"artifact_id must be ArtifactId: {self.artifact_id!r}")
        if not isinstance(self.sha256, Sha256):
            raise DomainError(f"sha256 must be Sha256: {self.sha256!r}")

    def to_primitive(self) -> dict:
        return {
            "artifact_id": self.artifact_id.to_primitive(),
            "sha256": self.sha256.to_primitive(),
        }

    @classmethod
    def from_primitive(cls, data: object) -> ArtifactRef:
        if not isinstance(data, Mapping) or set(data) != _ARTIFACT_KEYS:
            raise DomainError(
                f"ArtifactRef primitive must be mapping with keys {sorted(_ARTIFACT_KEYS)}: {data!r}"
            )
        return cls(
            ArtifactId.from_primitive(data["artifact_id"]),
            Sha256.from_primitive(data["sha256"]),
        )
