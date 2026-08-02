"""APP-COMPOSE-001C production generation→verification→repair vertical slice."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from specstyle.domain.artifacts import ArtifactRef
from specstyle.domain.enums import (
    ArtifactStatus,
    DecisionReason,
    RepairStopReason,
    RuleScope,
    RuleStatus,
    StaticApplicability,
)
from specstyle.domain.identifiers import ArtifactId, AttemptId, JobId, RuleId
from specstyle.generation.protocols import GeneratedArtifact
from specstyle.generation.requests import GenerationRequest
from specstyle.observability.hashing import hash_bytes
from specstyle.reliability.fixtures import sample_production_request
from specstyle.repair.actions import (
    DECREASE_STYLE_SCALE,
    INCREASE_STRUCTURE,
    INCREASE_STYLE_SCALE,
    REDUCE_DENOISE,
    RENDER_OUTPUT_PROFILE,
    RETRY_SAMPLING,
)
from specstyle.verification.rule_models import GatePolicy, RuleResult
from specstyle.workflow import production_service
from specstyle.workflow.job_models import Event, EventType, JobStatus
from specstyle.workflow.job_store import JobStore
from specstyle.workflow.production_artifacts import _open_production_artifact_store
from specstyle.workflow.production_reports import _open_production_report_store

_TS = "2026-08-02T00:00:00.000Z"


@dataclass(frozen=True)
class _RuleSpec:
    rule_id: str
    actions: tuple[object, ...]
    priority: int = 1
    policy: GatePolicy = GatePolicy("reject", "reject", "reject")


def _contract(rule_specs: tuple[_RuleSpec, ...]):
    base = sample_production_request()
    template = next(
        rule
        for rule in base.compiled_spec.verification_plans[0].rules
        if rule.definition.scope is RuleScope.ITEM and rule.metric_id is not None
    )
    rules = tuple(
        replace(
            template,
            definition=replace(
                template.definition,
                rule_id=RuleId(spec.rule_id),
                required=True,
                applicability=StaticApplicability.APPLICABLE,
                gate_policy=spec.policy,
            ),
            priority=spec.priority,
            affected_by_actions=spec.actions,
        )
        for spec in rule_specs
    )
    plan = replace(base.compiled_spec.verification_plans[0], rules=rules)
    compiled = replace(base.compiled_spec, verification_plans=(plan,))
    request = GenerationRequest(
        JobId("job"),
        AttemptId("job-a0-xhs_grid-0"),
        None,
        compiled,
        "production",
        "xhs_grid",
        base.source,
        base.style_references,
        base.prompt,
        base.control_input,
        0,
        base.environment_hash,
    )
    return compiled, compiled.production_graphs[0], plan, request


class _Verifier:
    def __init__(self, statuses: dict[str, RuleStatus]) -> None:
        self.statuses = statuses

    def verify(self, artifacts, rules, /):
        artifact_id = artifacts[0].artifact_id
        return tuple(
            RuleResult(
                rule.rule_id,
                self.statuses[rule.rule_id.value],
                (artifact_id,),
                None,
            )
            for rule in rules
        )


class _Factory:
    def __init__(self, outcomes: dict[str, dict[str, RuleStatus]]) -> None:
        self.outcomes = outcomes
        self.created: list[tuple[GenerationRequest, object, _Verifier]] = []
        self.closed = False

    def create(self, request, _plan, repository, _style_assets):
        verifier = _Verifier(self.outcomes[request.attempt_id.value])
        self.created.append((request, repository, verifier))
        return verifier

    def close(self) -> None:
        self.closed = True


class _TrackingRepository:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.puts = 0
        self.reads = 0
        self.closed = False

    def put(self, *args) -> None:
        self.puts += 1
        self.inner.put(*args)

    def __call__(self, *args):
        self.reads += 1
        return self.inner(*args)

    def close(self) -> None:
        self.closed = True
        self.inner.close()


class _TrackingStore:
    def __init__(self, inner, method: str) -> None:
        self.inner = inner
        self.method = method
        self.repositories: list[_TrackingRepository] = []

    def for_job(self, job_id):
        assert self.method == "for_job"
        repository = _TrackingRepository(self.inner.for_job(job_id))
        self.repositories.append(repository)
        return repository

    def for_attempt(self, job_id, attempt_id):
        assert self.method == "for_attempt"
        repository = _TrackingRepository(self.inner.for_attempt(job_id, attempt_id))
        self.repositories.append(repository)
        return repository

    def close(self) -> None:
        self.inner.close()


class _Loaded:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _DurabilityCheckingJobStore(JobStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.root = root

    def append_event(self, job_id: JobId, event: Event, /) -> Event:
        if event.event_type is EventType.ATTEMPT_FINISHED:
            artifact = event.payload.artifact_id.value
            directory = self.root / "jobs" / job_id.value / "artifacts" / artifact
            assert (directory / "artifact.png").is_file()
            assert (directory / "metadata.json").is_file()
        if event.event_type is EventType.VERIFIER_FINISHED:
            attempt = self.load(job_id).attempt_ids[-1].value
            directory = self.root / "jobs" / job_id.value / "reports" / attempt
            assert (directory / "report.json").is_file()
            assert (directory / "metadata.json").is_file()
        return super().append_event(job_id, event)


def _artifact(request: GenerationRequest) -> GeneratedArtifact:
    suffix = "a0" if request.parent_attempt_id is None else "a1"
    content = f"artifact-{suffix}".encode()
    return GeneratedArtifact(
        ArtifactRef(ArtifactId(f"artifact-{suffix}"), hash_bytes(content)),
        content,
        request.request_hash,
        request.generation_fingerprint,
    )


def _runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rule_specs: tuple[_RuleSpec, ...],
    outcomes: dict[str, dict[str, RuleStatus]],
):
    compiled, graph, plan, initial = _contract(rule_specs)
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        artifacts = _TrackingStore(_open_production_artifact_store(root_fd), "for_job")
        reports = _TrackingStore(_open_production_report_store(root_fd), "for_attempt")
    finally:
        os.close(root_fd)
    factory, loaded = _Factory(outcomes), _Loaded()
    runtime = object.__new__(production_service._ProductionGenerationRuntime)
    values = {
        "_loaded": loaded,
        "_verifier_factory": factory,
        "_report_store": reports,
        "_artifact_store": artifacts,
        "_environment": object(),
        "_compiler_context": object(),
        "_style_assets": lambda _reference: b"",
        "_control_builder": object(),
        "_job_store": _DurabilityCheckingJobStore(tmp_path),
        "_clock": production_service._NondecreasingAuditClock(lambda: _TS),
        "_closed": False,
    }
    for name, value in values.items():
        setattr(runtime, name, value)
    monkeypatch.setattr(
        production_service,
        "_select_initial_contract",
        lambda *_args: (compiled, graph, plan),
    )
    monkeypatch.setattr(production_service, "_preflight_bindings", lambda *_args: None)
    monkeypatch.setattr(
        production_service, "_initial_generation_request", lambda *_args: initial
    )
    monkeypatch.setattr(
        production_service,
        "_run_initial_generation",
        lambda *args: _artifact(
            next(item for item in args if type(item) is GenerationRequest)
        ),
    )
    request = production_service.ProductionJobRequest(
        initial.job_id,
        "spec",
        initial.source,
        initial.style_references,
        initial.prompt,
        initial.output_profile,
        initial.variation_index,
        "bundle",
    )
    return runtime, request, factory, artifacts, reports


def _event_summary(store: JobStore) -> list[tuple[str, str, str]]:
    return [
        (event.event_type.value, event.from_state.value, event.to_state.value)
        for event in store.list_events(JobId("job"))
    ]


@pytest.mark.parametrize(
    ("status", "policy", "expected_state", "reason", "stop"),
    [
        (
            RuleStatus.PASS,
            GatePolicy("reject", "reject", "reject"),
            JobStatus.APPROVED,
            DecisionReason.ALL_REQUIRED_PASS,
            RepairStopReason.PASS_ALL_REQUIRED,
        ),
        (
            RuleStatus.UNVERIFIABLE,
            GatePolicy("reject", "reject", "reject"),
            JobStatus.REJECTED,
            DecisionReason.REQUIRED_GATE_UNVERIFIABLE,
            RepairStopReason.UNVERIFIABLE,
        ),
        (
            RuleStatus.WARNING,
            GatePolicy("reject", "reject", "manual_review"),
            JobStatus.MANUAL_REVIEW,
            DecisionReason.MANUAL_POLICY,
            RepairStopReason.MANUAL_REQUEST,
        ),
    ],
)
def test_initial_terminal_routes_are_durable_before_exact_final_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: RuleStatus,
    policy: GatePolicy,
    expected_state: JobStatus,
    reason: DecisionReason,
    stop: RepairStopReason,
) -> None:
    attempt = "job-a0-xhs_grid-0"
    runtime, request, factory, artifacts, reports = _runtime(
        tmp_path,
        monkeypatch,
        (_RuleSpec("STYLE_LOW", (INCREASE_STYLE_SCALE,), policy=policy),),
        {attempt: {"STYLE_LOW": status}},
    )
    try:
        result = runtime._execute_initial_attempt(request)
        events = runtime._job_store.list_events(JobId("job"))
        assert _event_summary(runtime._job_store) == [
            ("JOB_STARTED", "CREATED", "SPEC_VALIDATED"),
            ("SPEC_COMPILED", "SPEC_VALIDATED", "SPEC_COMPILED"),
            ("ATTEMPT_STARTED", "SPEC_COMPILED", "GENERATING"),
            ("ATTEMPT_FINISHED", "GENERATING", "VERIFYING"),
            ("VERIFIER_FINISHED", "VERIFYING", expected_state.value),
        ]
        payload = events[-1].payload
        assert (payload.artifact_status, payload.decision_reason) == (
            ArtifactStatus(expected_state.value),
            reason,
        )
        assert payload.repair_stop_reason is stop
        assert result.request.attempt_id.value == attempt
        assert result.terminal.artifact_decision.repair_stop_reason is stop
        assert result.job_state.job.status is expected_state
        assert len(factory.created) == 1
        assert artifacts.repositories[0].reads >= 1
        assert reports.repositories[0].reads >= 1
        assert all(
            repo.closed for repo in artifacts.repositories + reports.repositories
        )
    finally:
        runtime.close()


def test_no_action_records_selecting_then_repair_exhausted(
    tmp_path, monkeypatch
) -> None:
    attempt = "job-a0-xhs_grid-0"
    runtime, request, factory, artifacts, reports = _runtime(
        tmp_path,
        monkeypatch,
        (_RuleSpec("STYLE_LOW", ()),),
        {attempt: {"STYLE_LOW": RuleStatus.FAIL}},
    )
    try:
        result = runtime._execute_initial_attempt(request)
        events = runtime._job_store.list_events(JobId("job"))
        assert _event_summary(runtime._job_store)[-2:] == [
            ("VERIFIER_FINISHED", "VERIFYING", "REPAIR_SELECTING"),
            ("VERIFIER_FINISHED", "REPAIR_SELECTING", "REJECTED"),
        ]
        assert events[-2].payload.repair_stop_reason is None
        assert events[-1].payload.decision_reason is DecisionReason.REPAIR_EXHAUSTED
        assert events[-1].payload.repair_stop_reason is RepairStopReason.NO_ACTION
        assert result.terminal.no_action is not None
        assert result.job_state.job.status is JobStatus.REJECTED
        assert len(factory.created) == 1
        assert all(
            repo.closed for repo in artifacts.repositories + reports.repositories
        )
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("child", "stop"),
    [
        ((RuleStatus.PASS,), RepairStopReason.PASS_ALL_REQUIRED),
        ((RuleStatus.FAIL,), RepairStopReason.NO_IMPROVEMENT),
    ],
)
def test_repair_round_pass_or_no_improvement_has_exact_events(
    tmp_path, monkeypatch, child: tuple[RuleStatus, ...], stop: RepairStopReason
) -> None:
    a0, a1 = "job-a0-xhs_grid-0", "job-a1-xhs_grid-0"
    runtime, request, factory, artifacts, reports = _runtime(
        tmp_path,
        monkeypatch,
        (_RuleSpec("STYLE_LOW", (INCREASE_STYLE_SCALE,)),),
        {
            a0: {"STYLE_LOW": RuleStatus.FAIL},
            a1: {"STYLE_LOW": child[0]},
        },
    )
    try:
        result = runtime._execute_initial_attempt(request)
        summaries = _event_summary(runtime._job_store)
        assert summaries[4:] == [
            ("VERIFIER_FINISHED", "VERIFYING", "REPAIR_SELECTING"),
            ("REPAIR_STEP", "REPAIR_SELECTING", "REPAIRING"),
            ("ATTEMPT_FINISHED", "REPAIRING", "VERIFYING"),
            (
                "VERIFIER_FINISHED",
                "VERIFYING",
                "APPROVED"
                if stop is RepairStopReason.PASS_ALL_REQUIRED
                else "REJECTED",
            ),
        ]
        repair_payload = runtime._job_store.list_events(JobId("job"))[5].payload
        assert (
            repair_payload.decision_id.value,
            repair_payload.action_id,
            repair_payload.parent_attempt_id.value,
            repair_payload.child_attempt_id.value,
        ) == (
            "job-d1-xhs_grid-0",
            INCREASE_STYLE_SCALE,
            a0,
            a1,
        )
        assert result.request.attempt_id.value == a1
        assert result.history.rounds == 1
        assert result.terminal.artifact_decision.repair_stop_reason is stop
        assert len(factory.created) == 2
        assert factory.created[0][2] is not factory.created[1][2]
        assert all(
            repo.closed for repo in artifacts.repositories + reports.repositories
        )
    finally:
        runtime.close()


def test_improved_but_still_failing_round_stops_at_max_rounds(
    tmp_path, monkeypatch
) -> None:
    a0, a1 = "job-a0-xhs_grid-0", "job-a1-xhs_grid-0"
    rules = (
        _RuleSpec("STYLE_LOW", (INCREASE_STYLE_SCALE,), 1),
        _RuleSpec("CONTENT_DRIFT", (REDUCE_DENOISE,), 2),
    )
    runtime, request, _factory, _artifacts, _reports = _runtime(
        tmp_path,
        monkeypatch,
        rules,
        {
            a0: {"STYLE_LOW": RuleStatus.FAIL, "CONTENT_DRIFT": RuleStatus.FAIL},
            a1: {"STYLE_LOW": RuleStatus.PASS, "CONTENT_DRIFT": RuleStatus.FAIL},
        },
    )
    try:
        result = runtime._execute_initial_attempt(request)
        payload = runtime._job_store.list_events(JobId("job"))[-1].payload
        assert payload.decision_reason is DecisionReason.REPAIR_EXHAUSTED
        assert payload.repair_stop_reason is RepairStopReason.MAX_ROUNDS
        assert result.history.consecutive_no_improvement == 0
        assert result.terminal.artifact_decision.repair_stop_reason is (
            RepairStopReason.MAX_ROUNDS
        )
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("rule_id", "actions", "expected"),
    [
        ("STYLE_LOW", (INCREASE_STYLE_SCALE,), INCREASE_STYLE_SCALE),
        (
            "STYLE_OVERPOWERED",
            (DECREASE_STYLE_SCALE, REDUCE_DENOISE),
            DECREASE_STYLE_SCALE,
        ),
        ("STYLE_OVERPOWERED", (REDUCE_DENOISE,), REDUCE_DENOISE),
        ("CONTENT_DRIFT", (REDUCE_DENOISE, INCREASE_STRUCTURE), REDUCE_DENOISE),
        ("CONTENT_DRIFT", (INCREASE_STRUCTURE,), INCREASE_STRUCTURE),
        ("FACE_ID_LOW", (REDUCE_DENOISE,), REDUCE_DENOISE),
        ("SAMPLING_DEFECT", (RETRY_SAMPLING,), RETRY_SAMPLING),
        ("OUTPUT_PROFILE_INVALID", (RENDER_OUTPUT_PROFILE,), None),
        ("UNKNOWN_BLOCKER", (INCREASE_STYLE_SCALE,), None),
        ("STYLE_LOW", (REDUCE_DENOISE,), None),
    ],
)
def test_repair_action_uses_core_policy_and_compiled_intersection(
    tmp_path, monkeypatch, rule_id: str, actions: tuple[object, ...], expected
) -> None:
    a0, a1 = "job-a0-xhs_grid-0", "job-a1-xhs_grid-0"
    outcomes = {a0: {rule_id: RuleStatus.FAIL}}
    if expected is not None:
        outcomes[a1] = {rule_id: RuleStatus.PASS}
    runtime, request, _factory, _artifacts, _reports = _runtime(
        tmp_path, monkeypatch, (_RuleSpec(rule_id, actions),), outcomes
    )
    try:
        runtime._execute_initial_attempt(request)
        events = runtime._job_store.list_events(JobId("job"))
        repair = next(
            (event for event in events if event.event_type is EventType.REPAIR_STEP),
            None,
        )
        if expected is None:
            assert repair is None
            assert events[-1].payload.repair_stop_reason is RepairStopReason.NO_ACTION
        else:
            assert repair.payload.action_id == expected
    finally:
        runtime.close()
