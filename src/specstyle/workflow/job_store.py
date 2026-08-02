"""WF-001 本地 JobStore：append-only NDJSON 事件 + 原子 snapshot + 崩溃恢复。

单进程、本地、禁止多进程/分布式锁/SQLite/后台 GPU。canonical JSON 同 §13.9。
parser 拒绝 duplicate key/NaN/Inf/unknown-missing key；partial/截断/损坏 fail closed。
所有错误文本固定类别，脱敏（不含 path/credential/堆栈）。
"""

from __future__ import annotations

import json
import os
import stat
import threading
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path

from specstyle.domain.enums import (
    ArtifactStatus,
    DecisionReason,
    RepairStopReason,
)
from specstyle.domain.identifiers import (
    ArtifactId,
    AttemptId,
    DecisionId,
    Identifier,
    JobId,
    RuleId,
    Sha256,
)
from specstyle.errors import DomainError, InfrastructureError

from specstyle.workflow.job_models import (
    AttemptFinishedPayload,
    AttemptStartedPayload,
    CancelRequestedPayload,
    Event,
    EventType,
    ExportPublishedPayload,
    ExportStartedPayload,
    FatalPayload,
    Job,
    JobBudget,
    JobSnapshot,
    JobStartedPayload,
    JobState,
    JobStatus,
    RecoverablePayload,
    RepairStepPayload,
    SpecCompiledPayload,
    VerifierFinishedPayload,
)
from specstyle.workflow.state_machine import replay_events, validate_transition

_SNAPSHOT_VERSION = "specstyle.workflow.snapshot.v1"
_JOB_LOCKS_GUARD = threading.Lock()
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_READ_FLAGS = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
_MAX_SNAPSHOT_BYTES = 1024 * 1024
_MAX_EVENTS_BYTES = 64 * 1024 * 1024
_READ_CHUNK = 64 * 1024
_STATE_FILE_LIMITS = {
    "snapshot.json": (1, _MAX_SNAPSHOT_BYTES),
    "events.ndjson": (0, _MAX_EVENTS_BYTES),
}
_DirectoryIdentity = tuple[int, int, int, int, int, int, int, int]
_FileIdentity = tuple[int, int, int, int, int, int, int, int]


class _JobLockHolder:
    def __init__(self) -> None:
        self.lock = threading.RLock()


_JOB_LOCKS: weakref.WeakValueDictionary[tuple[str, str], _JobLockHolder] = (
    weakref.WeakValueDictionary()
)
_ID_FIELDS = {
    "job_id": JobId,
    "attempt_id": AttemptId,
    "parent_attempt_id": AttemptId,
    "artifact_id": ArtifactId,
    "decision_id": DecisionId,
    "action_id": Identifier,
    "trigger_rule_id": RuleId,
    "child_attempt_id": AttemptId,
}


def _canonical(value: object) -> object:
    if type(value) is float:
        return 0.0 if value == 0.0 else value
    if type(value) is dict:
        return {k: _canonical(v) for k, v in value.items()}
    if type(value) is list:
        return [_canonical(v) for v in value]
    return value


def _canonical_json(primitive: object) -> bytes:
    return json.dumps(
        _canonical(primitive),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    seen: set[str] = set()
    out: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in seen:
            raise DomainError("invalid job snapshot")
        seen.add(key)
        out[key] = value
    return out


def _parse(data: bytes) -> object:
    if type(data) is not bytes:
        raise DomainError("invalid job snapshot")
    try:
        return json.loads(
            data,
            object_pairs_hook=_reject_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                DomainError("invalid job snapshot")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as cause:
        raise DomainError("invalid job snapshot") from cause


def _exact_keys(data: object, keys: set[str]) -> dict[str, object]:
    if not isinstance(data, Mapping) or set(data) != keys:
        raise DomainError("invalid job snapshot")
    return dict(data)


def _id_value(data: dict[str, object], key: str, kind: type[Identifier]) -> Identifier:
    value = data[key]
    if not isinstance(value, str):
        raise DomainError("invalid job snapshot")
    return kind(value)


def _sha_value(data: dict[str, object], key: str) -> Sha256:
    value = data[key]
    if not isinstance(value, str):
        raise DomainError("invalid job snapshot")
    return Sha256(value)


def _int_value(data: dict[str, object], key: str) -> int:
    value = data[key]
    if type(value) is bool or type(value) is not int:
        raise DomainError("invalid job snapshot")
    return value


def _str_value(data: dict[str, object], key: str) -> str:
    value = data[key]
    if type(value) is not str:
        raise DomainError("invalid job snapshot")
    return value


def _enum_value(data: dict[str, object], key: str, cls: type) -> object:
    value = data[key]
    if type(value) is not str:
        raise DomainError("invalid job snapshot")
    try:
        return cls(value)
    except ValueError:
        raise DomainError("invalid job snapshot") from None


def _str_list(value: object, name: str) -> list[str]:
    if type(value) is not list or any(type(v) is not str for v in value):
        raise DomainError(f"invalid {name}")
    return value


def _job_to_primitive(job: Job) -> dict[str, object]:
    return {
        "job_id": job.job_id.value,
        "compiled_spec_hash": job.compiled_spec_hash.value,
        "cohort_profiles": list(job.cohort_profiles),
        "budget": {"max_attempts_per_item": job.budget.max_attempts_per_item},
        "status": job.status.value,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def _job_from_primitive(data: object) -> Job:
    d = _exact_keys(
        data,
        {
            "job_id",
            "compiled_spec_hash",
            "cohort_profiles",
            "budget",
            "status",
            "created_at",
            "updated_at",
        },
    )
    budget = _exact_keys(d["budget"], {"max_attempts_per_item"})
    return Job(
        _id_value(d, "job_id", JobId),
        _sha_value(d, "compiled_spec_hash"),
        tuple(_str_list(d["cohort_profiles"], "cohort profiles")),
        JobBudget(_int_value(budget, "max_attempts_per_item")),
        JobStatus(_str_value(d, "status")),
        _str_value(d, "created_at"),
        _str_value(d, "updated_at"),
    )


_PAYLOAD_BUILDERS = {
    EventType.JOB_STARTED: lambda d: JobStartedPayload(
        _sha_value(d, "compiled_spec_hash"),
        tuple(_str_list(d["cohort_profiles"], "cohort profiles")),
        JobBudget(
            _int_value(
                _exact_keys(d["budget"], {"max_attempts_per_item"}),
                "max_attempts_per_item",
            )
        ),
    ),
    EventType.SPEC_COMPILED: lambda d: SpecCompiledPayload(
        _sha_value(d, "compiled_spec_hash")
    ),
    EventType.ATTEMPT_STARTED: lambda d: AttemptStartedPayload(
        _int_value(d, "cohort_index"),
        _int_value(d, "item_index"),
        _id_value(d, "attempt_id", AttemptId),
        _id_value(d, "parent_attempt_id", AttemptId)
        if d["parent_attempt_id"] is not None
        else None,
    ),
    EventType.ATTEMPT_FINISHED: lambda d: AttemptFinishedPayload(
        _int_value(d, "cohort_index"),
        _int_value(d, "item_index"),
        _id_value(d, "attempt_id", AttemptId),
        _id_value(d, "artifact_id", ArtifactId),
        _sha_value(d, "request_hash"),
    ),
    EventType.VERIFIER_FINISHED: lambda d: VerifierFinishedPayload(
        _int_value(d, "cohort_index"),
        _int_value(d, "item_index"),
        _id_value(d, "artifact_id", ArtifactId),
        ArtifactStatus(d["artifact_status"])
        if isinstance(d["artifact_status"], ArtifactStatus)
        else _enum_value(d, "artifact_status", ArtifactStatus),
        DecisionReason(d["decision_reason"])
        if isinstance(d["decision_reason"], DecisionReason)
        else _enum_value(d, "decision_reason", DecisionReason),
        d["repair_stop_reason"]
        if d["repair_stop_reason"] is None
        or type(d["repair_stop_reason"]) is RepairStopReason
        else _enum_value(d, "repair_stop_reason", RepairStopReason),
    ),
    EventType.REPAIR_STEP: lambda d: RepairStepPayload(
        _int_value(d, "cohort_index"),
        _int_value(d, "item_index"),
        _id_value(d, "decision_id", DecisionId),
        _id_value(d, "action_id", Identifier),
        _id_value(d, "trigger_rule_id", RuleId),
        _id_value(d, "parent_attempt_id", AttemptId),
        _id_value(d, "child_attempt_id", AttemptId),
    ),
    EventType.EXPORT_PUBLISHED: lambda d: ExportPublishedPayload(
        _str_value(d, "bundle_name"),
        _sha_value(d, "manifest_sha256"),
        _sha_value(d, "payload_sha256"),
        _sha_value(d, "bundle_sha256"),
    ),
    EventType.EXPORT_STARTED: lambda d: ExportStartedPayload(
        _str_value(d, "bundle_name")
    ),
    EventType.RECOVERABLE: lambda d: RecoverablePayload(_str_value(d, "reason")),
    EventType.CANCEL_REQUESTED: lambda d: CancelRequestedPayload(
        _str_value(d, "reason")
    ),
    EventType.FATAL: lambda d: FatalPayload(
        _str_value(d, "error_family"), _str_value(d, "reason")
    ),
}


def _payload_keys(event_type: EventType) -> set[str]:
    sample = {
        EventType.JOB_STARTED: {"compiled_spec_hash", "cohort_profiles", "budget"},
        EventType.ATTEMPT_STARTED: {
            "cohort_index",
            "item_index",
            "attempt_id",
            "parent_attempt_id",
        },
        EventType.ATTEMPT_FINISHED: {
            "cohort_index",
            "item_index",
            "attempt_id",
            "artifact_id",
            "request_hash",
        },
        EventType.VERIFIER_FINISHED: {
            "cohort_index",
            "item_index",
            "artifact_id",
            "artifact_status",
            "decision_reason",
            "repair_stop_reason",
        },
        EventType.REPAIR_STEP: {
            "cohort_index",
            "item_index",
            "decision_id",
            "action_id",
            "trigger_rule_id",
            "parent_attempt_id",
            "child_attempt_id",
        },
        EventType.EXPORT_PUBLISHED: {
            "bundle_name",
            "manifest_sha256",
            "payload_sha256",
            "bundle_sha256",
        },
        EventType.EXPORT_STARTED: {"bundle_name"},
        EventType.RECOVERABLE: {"reason"},
        EventType.SPEC_COMPILED: {"compiled_spec_hash"},
        EventType.CANCEL_REQUESTED: {"reason"},
        EventType.FATAL: {"error_family", "reason"},
    }
    return sample[event_type]


def _event_to_primitive(event: Event) -> dict[str, object]:
    return {
        "sequence": event.sequence,
        "job_id": event.job_id.value,
        "event_type": event.event_type.value,
        "from_state": event.from_state.value,
        "to_state": event.to_state.value,
        "timestamp": event.timestamp,
        "payload": _payload_to_primitive(event.event_type, event.payload),
    }


def _payload_to_primitive(event_type: EventType, payload: object) -> dict[str, object]:
    import enum as _enum

    from specstyle.workflow.job_models import JobBudget

    keys = _payload_keys(event_type)
    result: dict[str, object] = {}
    for key in keys:
        value = getattr(payload, key)
        if isinstance(value, (Identifier, Sha256)):
            result[key] = value.value
        elif isinstance(value, _enum.Enum):
            result[key] = value.value
        elif isinstance(value, JobBudget):
            result[key] = {"max_attempts_per_item": value.max_attempts_per_item}
        elif type(value) is tuple:
            result[key] = list(value)
        else:
            result[key] = value
    return result


def _event_from_primitive(data: object) -> Event:
    d = _exact_keys(
        data,
        {
            "sequence",
            "job_id",
            "event_type",
            "from_state",
            "to_state",
            "timestamp",
            "payload",
        },
    )
    event_type = _enum_value(d, "event_type", EventType)  # type: ignore[assignment]
    payload_data = _exact_keys(d["payload"], _payload_keys(event_type))  # type: ignore[arg-type]
    payload = _PAYLOAD_BUILDERS[event_type](payload_data)  # type: ignore[index]
    return Event(
        _int_value(d, "sequence"),
        JobId(_str_value(d, "job_id")),
        event_type,  # type: ignore[arg-type]
        _enum_value(d, "from_state", JobStatus),  # type: ignore[arg-type]
        _enum_value(d, "to_state", JobStatus),  # type: ignore[arg-type]
        _str_value(d, "timestamp"),
        payload,
    )


def _canonical_genesis(job: Job) -> JobSnapshot:
    """从 snapshot 所携带的不可变 Job 材料重建唯一的 seq=0 genesis。"""
    if type(job) is not Job:
        raise DomainError("invalid job snapshot") from None
    genesis_job = Job(
        job.job_id,
        job.compiled_spec_hash,
        job.cohort_profiles,
        JobBudget(job.budget.max_attempts_per_item),
        JobStatus.CREATED,
        job.created_at,
        job.created_at,
    )
    return JobSnapshot(_SNAPSHOT_VERSION, genesis_job, 0, (), ())


def _snapshot_to_primitive(snapshot: JobSnapshot) -> dict[str, object]:
    return {
        "schema_version": _SNAPSHOT_VERSION,
        "job": _job_to_primitive(snapshot.job),
        "last_sequence": snapshot.last_sequence,
        "attempt_ids": [attempt.value for attempt in snapshot.attempt_ids],
        "bundle_names": list(snapshot.bundle_names),
    }


def _snapshot_from_primitive(primitive: object) -> JobSnapshot:
    d = _exact_keys(
        primitive,
        {
            "schema_version",
            "job",
            "last_sequence",
            "attempt_ids",
            "bundle_names",
        },
    )
    if type(d["schema_version"]) is not str or d["schema_version"] != _SNAPSHOT_VERSION:
        raise DomainError("invalid job snapshot")
    return JobSnapshot(
        _SNAPSHOT_VERSION,
        _job_from_primitive(d["job"]),
        _int_value(d, "last_sequence"),
        tuple(AttemptId(value) for value in _str_list(d["attempt_ids"], "attempt ids")),
        tuple(_str_list(d["bundle_names"], "bundle names")),
    )


def _rebuild_snapshot(snapshot: object, /) -> JobSnapshot:
    if type(snapshot) is not JobSnapshot:
        raise DomainError("invalid job snapshot") from None
    try:
        rebuilt = _snapshot_from_primitive(_snapshot_to_primitive(snapshot))
    except Exception:
        raise DomainError("invalid job snapshot") from None
    if not _same_exact_structure(snapshot, rebuilt):
        raise DomainError("invalid job snapshot") from None
    return rebuilt


def _validated_snapshot(
    job_id: JobId, snapshot: JobSnapshot, events: tuple[Event, ...], /
) -> JobState:
    """验证完整、不截断的事件流，并证明 snapshot 是其精确 prefix。"""
    if (
        type(job_id) is not JobId
        or type(snapshot) is not JobSnapshot
        or type(events) is not tuple
        or type(snapshot.last_sequence) is not int
        or snapshot.job.job_id != job_id
        or not 0 <= snapshot.last_sequence <= len(events)
    ):
        raise DomainError("invalid job snapshot") from None
    genesis = _canonical_genesis(snapshot.job)
    prefix = replay_events(genesis, events[: snapshot.last_sequence])
    expected = JobSnapshot(
        _SNAPSHOT_VERSION,
        prefix.job,
        prefix.last_sequence,
        prefix.attempt_ids,
        prefix.bundle_names,
    )
    if snapshot != expected:
        raise DomainError("invalid job snapshot") from None
    return replay_events(expected, events[snapshot.last_sequence :])


def _safe_job_id(job_id: object, message: str, /) -> JobId:
    try:
        if type(job_id) is not JobId or type(job_id.value) is not str:
            raise DomainError(message)
        return JobId(str.__str__(job_id.value))
    except (AttributeError, TypeError, ValueError, DomainError):
        raise DomainError(message) from None


def _rebuild_event(event: object, /) -> Event:
    if type(event) is not Event:
        raise DomainError("invalid job event") from None
    try:
        rebuilt = _event_from_primitive(_event_to_primitive(event))
    except Exception:
        raise DomainError("invalid job event") from None
    if not _same_exact_structure(event, rebuilt):
        raise DomainError("invalid job event") from None
    return rebuilt


def _same_exact_structure(left: object, right: object, /) -> bool:
    if type(left) is not type(right):
        return False
    if is_dataclass(left):
        return all(
            _same_exact_structure(getattr(left, field.name), getattr(right, field.name))
            for field in fields(left)
        )
    if type(left) is tuple:
        return len(left) == len(right) and all(
            _same_exact_structure(item, other) for item, other in zip(left, right)
        )
    return left == right


def _job_store_corrupted() -> InfrastructureError:
    return InfrastructureError("job store corrupted")


def _directory_identity(result: os.stat_result) -> _DirectoryIdentity:
    if not stat.S_ISDIR(result.st_mode):
        raise _job_store_corrupted()
    return (
        result.st_dev,
        result.st_ino,
        result.st_uid,
        result.st_mode,
        result.st_nlink,
        result.st_size,
        result.st_mtime_ns,
        result.st_ctime_ns,
    )


def _close_quietly(fd: int) -> None:
    if fd < 0:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _open_checked_directory(
    name: os.PathLike[str] | str,
    /,
    *,
    parent_fd: int | None = None,
    missing_ok: bool = False,
) -> tuple[int, _DirectoryIdentity] | None:
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise _job_store_corrupted() from None
    except OSError:
        raise _job_store_corrupted() from None
    fd = -1
    try:
        fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        opened = os.fstat(fd)
        after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        identity = _directory_identity(before)
        if identity != _directory_identity(opened) or identity != _directory_identity(
            after
        ):
            raise _job_store_corrupted()
        return fd, identity
    except InfrastructureError:
        _close_quietly(fd)
        raise
    except OSError:
        _close_quietly(fd)
        raise _job_store_corrupted() from None


@contextmanager
def _closing_fd(fd: int, /):
    try:
        yield
    except BaseException:
        _close_quietly(fd)
        raise
    else:
        try:
            os.close(fd)
        except OSError:
            raise InfrastructureError("job store io failed") from None


def _directory_names(fd: int, /) -> tuple[str, ...]:
    try:
        with os.scandir(fd) as entries:
            names = tuple(entry.name for entry in entries)
    except OSError as cause:
        raise InfrastructureError("job store io failed") from cause
    if any(type(name) is not str for name in names):
        raise _job_store_corrupted()
    return names


def _named_directory_identity(
    parent_fd: int | None, name: os.PathLike[str] | str, /
) -> _DirectoryIdentity:
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        raise _job_store_corrupted() from None
    return _directory_identity(named)


def _require_directory_identity(
    fd: int,
    parent_fd: int | None,
    name: os.PathLike[str] | str,
    identity: _DirectoryIdentity,
) -> None:
    try:
        opened = os.fstat(fd)
    except OSError:
        raise _job_store_corrupted() from None
    named = _named_directory_identity(parent_fd, name)
    if identity != _directory_identity(opened) or identity != named:
        raise _job_store_corrupted()


def _file_identity(
    result: os.stat_result,
    directory_identity: _DirectoryIdentity,
    minimum: int,
    maximum: int,
) -> _FileIdentity:
    if (
        not stat.S_ISREG(result.st_mode)
        or result.st_dev != directory_identity[0]
        or result.st_uid != directory_identity[2]
        or result.st_nlink != 1
        or not minimum <= result.st_size <= maximum
    ):
        raise _job_store_corrupted()
    return (
        result.st_dev,
        result.st_ino,
        result.st_uid,
        result.st_mode,
        result.st_nlink,
        result.st_size,
        result.st_mtime_ns,
        result.st_ctime_ns,
    )


def _named_file_identity(
    directory_fd: int,
    name: str,
    directory_identity: _DirectoryIdentity,
    minimum: int,
    maximum: int,
) -> _FileIdentity:
    try:
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        raise _job_store_corrupted() from None
    return _file_identity(named, directory_identity, minimum, maximum)


def _opened_file_identity(
    fd: int,
    directory_fd: int,
    name: str,
    directory_identity: _DirectoryIdentity,
    minimum: int,
    maximum: int,
) -> _FileIdentity:
    try:
        opened = os.fstat(fd)
    except OSError:
        raise _job_store_corrupted() from None
    opened_identity = _file_identity(opened, directory_identity, minimum, maximum)
    named_identity = _named_file_identity(
        directory_fd, name, directory_identity, minimum, maximum
    )
    if opened_identity != named_identity:
        raise _job_store_corrupted()
    return opened_identity


def _state_namespace(
    directory_fd: int, directory_identity: _DirectoryIdentity, /
) -> dict[str, _FileIdentity]:
    names = _directory_names(directory_fd)
    if set(names) not in (
        {"snapshot.json"},
        {"snapshot.json", "events.ndjson"},
    ):
        raise _job_store_corrupted()
    return {
        name: _named_file_identity(
            directory_fd,
            name,
            directory_identity,
            *_STATE_FILE_LIMITS[name],
        )
        for name in names
    }


def _read_exact(fd: int, size: int, /) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        amount = min(remaining, _READ_CHUNK)
        try:
            chunk = os.read(fd, amount)
        except OSError as cause:
            raise InfrastructureError("job store io failed") from cause
        if type(chunk) is not bytes or not chunk or len(chunk) > amount:
            raise _job_store_corrupted()
        chunks.append(chunk)
        remaining -= len(chunk)
    try:
        extra = os.read(fd, 1)
    except OSError as cause:
        raise InfrastructureError("job store io failed") from cause
    if type(extra) is not bytes or extra:
        raise _job_store_corrupted()
    return b"".join(chunks)


def _read_state_file(
    directory_fd: int,
    directory_identity: _DirectoryIdentity,
    name: str,
    expected_identity: _FileIdentity,
    /,
) -> bytes:
    minimum, maximum = _STATE_FILE_LIMITS[name]
    try:
        fd = os.open(name, _READ_FLAGS, dir_fd=directory_fd)
    except OSError:
        raise _job_store_corrupted() from None
    with _closing_fd(fd):
        before = _opened_file_identity(
            fd, directory_fd, name, directory_identity, minimum, maximum
        )
        if before != expected_identity:
            raise _job_store_corrupted()
        content = _read_exact(fd, before[5])
        after = _opened_file_identity(
            fd, directory_fd, name, directory_identity, minimum, maximum
        )
        if before != after:
            raise _job_store_corrupted()
        return content


def _snapshot_from_bytes(data: bytes, /) -> JobSnapshot:
    try:
        return _snapshot_from_primitive(_parse(data))
    except Exception:
        raise _job_store_corrupted() from None


def _events_from_bytes(job_id: JobId, data: bytes, /) -> tuple[Event, ...]:
    if data and not data.endswith(b"\n"):
        raise _job_store_corrupted()
    events: list[Event] = []
    for line in data.splitlines():
        if not line:
            raise _job_store_corrupted()
        try:
            event = _event_from_primitive(_parse(line))
            if event.job_id != job_id:
                raise DomainError("invalid job event")
        except Exception:
            raise _job_store_corrupted() from None
        events.append(event)
    return tuple(events)


def _validated_state_from_fd(
    job_fd: int,
    directory_identity: _DirectoryIdentity,
    job_id: JobId,
    /,
) -> JobState:
    namespace = _state_namespace(job_fd, directory_identity)
    snapshot_data = _read_state_file(
        job_fd,
        directory_identity,
        "snapshot.json",
        namespace["snapshot.json"],
    )
    events_data = b""
    if "events.ndjson" in namespace:
        events_data = _read_state_file(
            job_fd,
            directory_identity,
            "events.ndjson",
            namespace["events.ndjson"],
        )
    snapshot = _snapshot_from_bytes(snapshot_data)
    events = _events_from_bytes(job_id, events_data)
    try:
        state = _validated_snapshot(job_id, snapshot, events)
    except Exception:
        raise _job_store_corrupted() from None
    if _state_namespace(job_fd, directory_identity) != namespace:
        raise _job_store_corrupted()
    return state


class JobStore:
    def __init__(self, root: Path, /) -> None:
        if not isinstance(root, Path):
            raise DomainError("invalid job store root")
        if not root.is_dir():
            raise DomainError("invalid job store root")
        self._root = root
        self._lock_root = os.path.realpath(os.fspath(root))

    @contextmanager
    def _job_lock(self, job_id: JobId, /):
        key = (self._lock_root, job_id.value)
        with _JOB_LOCKS_GUARD:
            holder = _JOB_LOCKS.get(key)
            if holder is None:
                holder = _JobLockHolder()
                _JOB_LOCKS[key] = holder
        with holder.lock:
            yield

    def _job_dir(self, job_id: JobId) -> Path:
        return self._root / "jobs" / job_id.value

    def list_job_ids(self, /) -> tuple[JobId, ...]:
        jobs_path = self._root / "jobs"
        opened = _open_checked_directory(jobs_path, missing_ok=True)
        if opened is None:
            return ()
        jobs_fd, jobs_identity = opened
        with _closing_fd(jobs_fd):
            names = _directory_names(jobs_fd)
            listed = tuple(self._validated_listed_job(jobs_fd, name) for name in names)
            if set(_directory_names(jobs_fd)) != set(names):
                raise _job_store_corrupted()
            if any(
                _named_directory_identity(jobs_fd, name) != identity
                for name, (_, identity) in zip(names, listed, strict=True)
            ):
                raise _job_store_corrupted()
            _require_directory_identity(jobs_fd, None, jobs_path, jobs_identity)
        return tuple(
            sorted((item[0] for item in listed), key=lambda job_id: job_id.value)
        )

    def _validated_listed_job(
        self, jobs_fd: int, name: str, /
    ) -> tuple[JobId, _DirectoryIdentity]:
        try:
            job_id = JobId(name)
        except DomainError:
            raise _job_store_corrupted() from None
        opened = _open_checked_directory(name, parent_fd=jobs_fd)
        if opened is None:
            raise _job_store_corrupted()
        job_fd, identity = opened
        with _closing_fd(job_fd):
            _validated_state_from_fd(job_fd, identity, job_id)
            _require_directory_identity(job_fd, jobs_fd, name, identity)
        return job_id, identity

    def load(self, job_id: JobId, /) -> JobState:
        job_id = _safe_job_id(job_id, "invalid job event")
        snapshot, events = self._read_disk(job_id)
        if snapshot is None:
            raise DomainError("job not found") from None
        try:
            return _validated_snapshot(job_id, snapshot, events)
        except Exception:
            raise InfrastructureError("job store corrupted") from None

    def get_snapshot(self, job_id: JobId, /) -> JobSnapshot | None:
        job_id = _safe_job_id(job_id, "invalid job event")
        snapshot, events = self._read_disk(job_id)
        if snapshot is None:
            return None
        try:
            _validated_snapshot(job_id, snapshot, events)
        except Exception:
            raise InfrastructureError("job store corrupted") from None
        return snapshot

    def _read_disk(
        self, job_id: JobId, /
    ) -> tuple[JobSnapshot | None, tuple[Event, ...]]:
        job_id = _safe_job_id(job_id, "invalid job event")
        snapshot = self._read_snapshot(job_id)
        events = self._read_events(job_id)
        if snapshot is None and events:
            raise InfrastructureError("job store corrupted") from None
        return snapshot, events

    def _read_snapshot(self, job_id: JobId, /) -> JobSnapshot | None:
        try:
            path = self._job_dir(job_id) / "snapshot.json"
            if not path.exists():
                return None
            data = path.read_bytes()
        except OSError as cause:
            raise InfrastructureError("job store io failed") from cause
        return _snapshot_from_bytes(data)

    def save_snapshot(self, job_id: JobId, snapshot: JobSnapshot, /) -> None:
        try:
            snapshot = _rebuild_snapshot(snapshot)
            valid = (
                type(job_id) is JobId
                and type(job_id.value) is str
                and type(snapshot.job) is Job
                and snapshot.job.job_id == job_id
            )
        except (AttributeError, TypeError):
            valid = False
        if not valid:
            raise DomainError("invalid job snapshot") from None
        job_id = _safe_job_id(job_id, "invalid job snapshot")
        existing, events = self._read_disk(job_id)
        try:
            candidate_state = _validated_snapshot(job_id, snapshot, events)
            if existing is None:
                if snapshot != _canonical_genesis(snapshot.job):
                    raise DomainError("invalid job snapshot")
            else:
                _validated_snapshot(job_id, existing, events)
                if _canonical_genesis(existing.job) != _canonical_genesis(snapshot.job):
                    raise DomainError("invalid job snapshot")
                if snapshot.last_sequence < existing.last_sequence:
                    raise DomainError("invalid job snapshot")
                if (
                    snapshot.last_sequence == existing.last_sequence
                    and snapshot != existing
                ):
                    raise DomainError("invalid job snapshot")
            del candidate_state
        except (DomainError, TypeError, AttributeError):
            raise DomainError("invalid job snapshot") from None
        data = _canonical_json(_snapshot_to_primitive(snapshot))
        directory = self._job_dir(job_id)
        tmp = directory / "snapshot.tmp"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, directory / "snapshot.json")
            _fsync_dir(directory, require=True)
        except OSError as cause:
            raise InfrastructureError("job store io failed") from cause

    def append_event(self, job_id: JobId, event: Event, /) -> Event:
        if type(job_id) is not JobId or type(event) is not Event:
            raise DomainError("invalid job event") from None
        job_id = _safe_job_id(job_id, "invalid job event")
        event = _rebuild_event(event)
        with self._job_lock(job_id):
            return self._append_event_locked(job_id, event)

    def _append_event_locked(self, job_id: JobId, event: Event, /) -> Event:
        try:
            return self._append_event_locked_impl(job_id, event)
        except (AttributeError, TypeError, ValueError):
            raise DomainError("invalid job event") from None

    def _append_event_locked_impl(self, job_id: JobId, event: Event, /) -> Event:
        state = self.load(job_id)
        if state.job.terminal:
            raise DomainError("job is terminal") from None
        attempts = {a.value for a in state.attempt_ids}
        bundles = set(state.bundle_names)
        if event.event_type is EventType.ATTEMPT_STARTED:
            if event.payload.attempt_id.value in attempts:
                raise DomainError("duplicate job attempt") from None
        if event.event_type is EventType.REPAIR_STEP:
            if event.payload.child_attempt_id.value in attempts:
                raise DomainError("duplicate job attempt") from None
        if event.event_type is EventType.EXPORT_PUBLISHED:
            if event.payload.bundle_name in bundles:
                raise DomainError("duplicate job export") from None
        if event.job_id != job_id:
            raise DomainError("invalid job event") from None
        if event.from_state is not state.job.status:
            raise DomainError("invalid job transition") from None
        validate_transition(state.job.status, event.to_state, event.event_type)
        sequence = state.last_sequence + 1
        finalized = Event(
            sequence,
            event.job_id,
            event.event_type,
            event.from_state,
            event.to_state,
            event.timestamp,
            event.payload,
        )
        try:
            replay_events(
                JobSnapshot(
                    _SNAPSHOT_VERSION,
                    state.job,
                    state.last_sequence,
                    state.attempt_ids,
                    state.bundle_names,
                ),
                (finalized,),
            )
        except (DomainError, TypeError, AttributeError):
            raise DomainError("invalid job event") from None
        directory = self._job_dir(job_id)
        path = directory / "events.ndjson"
        line = _canonical_json(_event_to_primitive(finalized)) + b"\n"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with open(path, "ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            # 目录项落盘：崩溃后 attempt/export 幂等集不得回退（恢复 at-most-once）
            _fsync_dir(directory, require=True)
        except OSError as cause:
            raise InfrastructureError("job store io failed") from cause
        return finalized

    def list_events(self, job_id: JobId, /) -> tuple[Event, ...]:
        job_id = _safe_job_id(job_id, "invalid job event")
        snapshot, events = self._read_disk(job_id)
        if snapshot is not None:
            try:
                _validated_snapshot(job_id, snapshot, events)
            except Exception:
                raise InfrastructureError("job store corrupted") from None
        return events

    def _read_events(self, job_id: JobId, /) -> tuple[Event, ...]:
        try:
            path = self._job_dir(job_id) / "events.ndjson"
            if not path.exists():
                return ()
            data = path.read_bytes()
        except OSError as cause:
            raise InfrastructureError("job store io failed") from cause
        return _events_from_bytes(job_id, data)


def _fsync_dir(directory: Path, *, require: bool = False) -> None:
    """fsync 目录项。

    所有当前调用均以 ``require=True`` 要求目录项落盘；保留参数仅用于明确
    表达调用方的耐久性要求。
    """
    try:
        fd = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        if require:
            raise
        return
    try:
        os.fsync(fd)
    except OSError:
        if require:
            raise
    finally:
        os.close(fd)
