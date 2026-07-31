"""UI view models — pure data for presenters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpecEditorView:
    schema_version: str
    spec_id: str
    compile_ok: bool
    errors: tuple[str, ...]
    compiled_hash: str | None


@dataclass(frozen=True, slots=True)
class JobStatusView:
    job_id: str
    status: str
    message: str


@dataclass(frozen=True, slots=True)
class QaRuleView:
    rule_id: str
    status: str
    score: float | None


@dataclass(frozen=True, slots=True)
class ExportView:
    bundle_name: str
    approved_count: int
    rejected_count: int
    manual_review_count: int
    bundle_sha256: str | None
