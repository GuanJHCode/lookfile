"""REL-001 fault injection scenarios for CPU suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from specstyle.errors import DomainError

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


def run_fault_suite(
    scenarios: tuple[FaultScenario, ...],
    injector: Callable[[FaultKind], FaultResult],
) -> tuple[FaultResult, ...]:
    if type(scenarios) is not tuple or not scenarios:
        raise DomainError("empty fault suite")
    if not callable(injector):
        raise DomainError("invalid injector")
    results: list[FaultResult] = []
    for scenario in scenarios:
        if type(scenario) is not FaultScenario:
            raise DomainError("invalid scenario")
        result = injector(scenario.kind)
        if type(result) is not FaultResult:
            raise DomainError("invalid fault result")
        if scenario.expect_fail_closed and not result.failed_closed:
            raise DomainError(f"fault not fail-closed: {scenario.kind}")
        results.append(result)
    return tuple(results)


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
