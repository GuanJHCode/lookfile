"""Model registry with license/revision/SHA gate (SEC-001A consumable)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from specstyle.domain.identifiers import Sha256
from specstyle.errors import DomainError

LicenseStatus = Literal["APPROVED", "BLOCKED", "UNKNOWN"]
ModelRole = Literal[
    "base", "ip_adapter", "controlnet", "preview_adapter", "style_encoder", "l3_feature"
]


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    model_id: str
    role: ModelRole
    revision: str
    expected_sha256: Sha256
    license_spdx: str
    license_status: LicenseStatus
    family: str

    def __post_init__(self) -> None:
        if type(self.model_id) is not str or not self.model_id:
            raise DomainError("invalid model descriptor")
        if self.role not in (
            "base",
            "ip_adapter",
            "controlnet",
            "preview_adapter",
            "style_encoder",
            "l3_feature",
        ):
            raise DomainError("invalid model role")
        if (
            type(self.revision) is not str
            or not self.revision
            or self.revision
            in (
                "main",
                "latest",
            )
        ):
            raise DomainError("floating revision forbidden")
        if type(self.expected_sha256) is not Sha256:
            raise DomainError("invalid model sha")
        if type(self.license_spdx) is not str or not self.license_spdx:
            raise DomainError("invalid license")
        if self.license_status not in ("APPROVED", "BLOCKED", "UNKNOWN"):
            raise DomainError("invalid license status")
        if type(self.family) is not str or not self.family:
            raise DomainError("invalid family")


class ModelRegistry:
    def __init__(self, descriptors: tuple[ModelDescriptor, ...]) -> None:
        if type(descriptors) is not tuple:
            raise DomainError("invalid registry")
        seen: set[str] = set()
        by_id: dict[str, ModelDescriptor] = {}
        for d in descriptors:
            if type(d) is not ModelDescriptor:
                raise DomainError("invalid registry")
            if d.model_id in seen:
                raise DomainError("duplicate model id")
            seen.add(d.model_id)
            by_id[d.model_id] = d
        self._by_id = by_id

    def get(self, model_id: str) -> ModelDescriptor:
        if type(model_id) is not str or model_id not in self._by_id:
            raise DomainError("model not registered")
        return self._by_id[model_id]

    def require_production(self, model_id: str) -> ModelDescriptor:
        desc = self.get(model_id)
        if desc.license_status == "UNKNOWN":
            raise DomainError("model license UNKNOWN blocked")
        if desc.license_status == "BLOCKED":
            raise DomainError("model license BLOCKED")
        if desc.license_status != "APPROVED":
            raise DomainError("model license not approved")
        return desc
