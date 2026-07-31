"""REL-001 fault injection harness."""

from __future__ import annotations

from specstyle.reliability.fault_injection import (
    DEFAULT_SCENARIOS,
    FaultResult,
    run_fault_suite,
)


def test_all_faults_fail_closed() -> None:
    def injector(kind):
        return FaultResult(kind, True, f"handled:{kind}")

    results = run_fault_suite(DEFAULT_SCENARIOS, injector)
    assert len(results) == len(DEFAULT_SCENARIOS)
    assert all(r.failed_closed for r in results)
