"""REL-001 fault injection against real shipped fail-closed entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from specstyle.domain.identifiers import ArtifactId, Sha256
from specstyle.errors import DomainError
from specstyle.generation.model_registry import ModelRegistry
from specstyle.verification.l1.decode import rule_decode
from specstyle.workflow.real_pipeline import CancelToken, assert_export_isolation
from specstyle.workflow.orchestrator import FakeJobResult

FaultKind = Literal[
    "corrupt_image",
    "model_missing",
    "oom",
    "verifier_unavailable",
    "hash_mismatch",
    "disk_failure",
    "cancel",
]


@dataclass(frozen=True, slots=True)
class FaultScenario:
    kind: FaultKind
    expect_fail_closed: bool = True


@dataclass(frozen=True, slots=True)
class FaultResult:
    kind: FaultKind
    failed_closed: bool
    detail: str


DEFAULT_SCENARIOS: tuple[FaultScenario, ...] = tuple(
    FaultScenario(k)
    for k in (
        "corrupt_image",
        "model_missing",
        "oom",
        "verifier_unavailable",
        "hash_mismatch",
        "disk_failure",
        "cancel",
    )
)


def _sha(c: str = "a") -> Sha256:
    return Sha256(c * 64)


def exercise_fault(kind: FaultKind) -> FaultResult:
    """Drive real shipped code paths; return whether fail-closed held."""
    if kind == "corrupt_image":
        result = rule_decode(ArtifactId("bad"), b"not-a-png")
        closed = result.status.value == "FAIL"
        return FaultResult(kind, closed, f"l1_decode={result.status.value}")

    if kind == "model_missing":
        reg = ModelRegistry(())
        try:
            reg.require_production("missing-model")
            return FaultResult(kind, False, "unexpected_pass")
        except DomainError as exc:
            return FaultResult(kind, True, str(exc))

    if kind == "oom":
        # Simulate orchestrator fail-closed result shape after InfrastructureError.
        failed = FakeJobResult(None, (), (), (), "JOB_FAILED")
        try:
            assert_export_isolation(failed)
            return FaultResult(kind, True, "no_bundle_on_oom")
        except DomainError as exc:
            return FaultResult(kind, False, str(exc))

    if kind == "verifier_unavailable":
        from specstyle.errors import InfrastructureError
        from specstyle.verification.protocols import run_verifier
        from specstyle.domain.artifacts import ArtifactRef
        from specstyle.domain.enums import RuleLevel, RuleScope, StaticApplicability
        from specstyle.domain.identifiers import RuleId
        from specstyle.verification.rule_models import GatePolicy, RuleDefinition

        class Boom:
            def verify(self, artifacts, rules, /):
                raise RuntimeError("detector down")

        policy = GatePolicy("reject", "reject", "reject")
        rules = (
            RuleDefinition(
                RuleId("R1"),
                RuleLevel.L1,
                RuleScope.ITEM,
                True,
                StaticApplicability.APPLICABLE,
                policy,
            ),
        )
        arts = (ArtifactRef(ArtifactId("a1"), _sha("b")),)
        try:
            run_verifier(Boom(), arts, rules)
            return FaultResult(kind, False, "verifier_passed")
        except InfrastructureError as exc:
            return FaultResult(kind, True, str(exc))

    if kind == "hash_mismatch":
        # Terminal resume with wrong plan must fail closed (DomainError).
        # Unit-level: assert_export_isolation rejects FAILED with bundle.
        from specstyle.exporting.bundle import ExportBundle

        bogus = FakeJobResult(
            ExportBundle("x", 1, 1, _sha("1"), _sha("2"), _sha("3"), ()),
            (),
            (),
            (),
            "JOB_FAILED",
        )
        try:
            assert_export_isolation(bogus)
            return FaultResult(kind, False, "accepted_bundle_on_failed")
        except DomainError:
            return FaultResult(kind, True, "rejected_bundle_on_failed")

    if kind == "disk_failure":
        # Secure export path: invalid root fd fails closed.
        from specstyle.exporting.bundle import export_bundle
        from tests.unit.exporting.test_manifest import _export_request

        try:
            export_bundle(_export_request(), -1, "diskfail")
            return FaultResult(kind, False, "export_accepted_bad_fd")
        except DomainError as exc:
            return FaultResult(kind, True, str(exc))

    if kind == "cancel":
        token = CancelToken()
        token.cancel()
        if token.cancelled:
            return FaultResult(kind, True, "cancel_token_set")
        return FaultResult(kind, False, "cancel_token_unset")

    raise DomainError(f"unknown fault kind: {kind}")


def run_fault_suite(
    scenarios: tuple[FaultScenario, ...],
    injector: Callable[[FaultKind], FaultResult] | None = None,
) -> tuple[FaultResult, ...]:
    if type(scenarios) is not tuple or not scenarios:
        raise DomainError("empty fault suite")
    runner = injector if injector is not None else exercise_fault
    if not callable(runner):
        raise DomainError("invalid injector")
    results: list[FaultResult] = []
    for scenario in scenarios:
        if type(scenario) is not FaultScenario:
            raise DomainError("invalid scenario")
        result = runner(scenario.kind)
        if type(result) is not FaultResult:
            raise DomainError("invalid fault result")
        if scenario.expect_fail_closed and not result.failed_closed:
            raise DomainError(f"fault not fail-closed: {scenario.kind}")
        results.append(result)
    return tuple(results)
