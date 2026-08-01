"""Production evidence dispatch, caching, policy, and error taxonomy tests."""

from __future__ import annotations

import importlib
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from specstyle.domain.artifacts import ArtifactRef
from specstyle.domain.enums import RuleStatus
from specstyle.errors import DomainError, InfrastructureError
from specstyle.generation.protocols import GeneratedArtifact
from specstyle.observability.hashing import hash_bytes
from tests.unit.verification._production_fixtures import (
    _OOM,
    _ProductionCase,
    _first_pixel,
    _make_production_case,
    _png,
)


def _bind(case: _ProductionCase) -> tuple[object, tuple[object, ...]]:
    production = importlib.import_module("specstyle.verification.production")
    factory = production._create_production_verifier_factory(
        case.loaded, case.allowlist(production)
    )
    verifier = factory.create(
        case.request,
        case.plan,
        case.artifact_resolver,
        case.style_resolver,
    )
    return verifier, case.plan.applicable_rule_definitions


def _result(case: _ProductionCase, rule_id: str) -> object:
    verifier, rules = _bind(case)
    results = verifier.verify((case.artifact.ref,), rules)
    return next(result for result in results if result.rule_id.value == rule_id)


def _bind_with(
    case: _ProductionCase,
    artifact_resolver: object,
    style_resolver: object,
) -> tuple[object, tuple[object, ...]]:
    production = importlib.import_module("specstyle.verification.production")
    factory = production._create_production_verifier_factory(
        case.loaded, case.allowlist(production)
    )
    verifier = factory.create(
        case.request,
        case.plan,
        artifact_resolver,
        style_resolver,
    )
    return verifier, case.plan.applicable_rule_definitions


def _drift_l2_threshold(case: _ProductionCase) -> None:
    rule = next(
        rule for rule in case.plan.rules if rule.definition.rule_id.value == "l2_style"
    )
    object.__setattr__(rule.threshold_binding, "value", 0.75)


def _callback_result(case: _ProductionCase, failure: bool) -> GeneratedArtifact:
    _drift_l2_threshold(case)
    if failure:
        raise RuntimeError("private callback failure")
    return case.artifact


def _animated_rgb_webp() -> bytes:
    frames = [
        Image.new("RGB", (64, 64), color) for color in ((10, 20, 30), (40, 50, 60))
    ]
    output = BytesIO()
    frames[0].save(
        output,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        lossless=True,
    )
    return output.getvalue()


@pytest.mark.parametrize(
    ("status", "expected"),
    (
        ("DRAFT", RuleStatus.UNVERIFIABLE),
        ("CALIBRATED", RuleStatus.UNVERIFIABLE),
    ),
)
def test_l2_nonvalidated_status_never_resolves_or_encodes(
    tmp_path: Path, status: str, expected: RuleStatus
) -> None:
    case = _make_production_case(tmp_path, l2_status=status, l3_status="DRAFT")
    try:
        result = _result(case, "l2_style")
        assert result.status is expected
        assert result.score is None
        assert case.style_resolver.calls == []
        assert case.evidence_calls == {}
    finally:
        case.close()


@pytest.mark.parametrize(
    ("status", "expected"),
    (
        ("DRAFT", RuleStatus.UNVERIFIABLE),
        ("CALIBRATED", RuleStatus.UNVERIFIABLE),
    ),
)
def test_l3_nonvalidated_status_never_encodes(
    tmp_path: Path, status: str, expected: RuleStatus
) -> None:
    case = _make_production_case(tmp_path, l2_status="DRAFT", l3_status=status)
    try:
        result = _result(case, "l3_diagnostic")
        assert result.status is expected
        assert result.score is None
        assert case.style_resolver.calls == []
        assert case.evidence_calls == {}
    finally:
        case.close()


@pytest.mark.parametrize(
    ("reference_vectors", "expected", "status"),
    (
        (((-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)), 0.0, RuleStatus.FAIL),
        (
            ((-1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 0.0)),
            0.5,
            RuleStatus.PASS,
        ),
    ),
)
def test_l2_runtime_odd_even_median_and_equality_threshold(
    tmp_path: Path,
    reference_vectors: tuple[tuple[float, float], ...],
    expected: float,
    status: RuleStatus,
) -> None:
    styles = tuple(
        _png((20 + index, 30 + index, 100 + index))
        for index in range(len(reference_vectors))
    )
    case = _make_production_case(
        tmp_path,
        style_contents=styles,
        l2_status="VALIDATED",
        l3_status="DRAFT",
    )
    try:
        for content, vector in zip(styles, reference_vectors, strict=True):
            case.evidence_vectors[_first_pixel(content)] = (
                [list(vector), list(vector)],
                [1.0, 0.0],
            )
        result = _result(case, "l2_style")
        assert result.score == pytest.approx(expected)
        assert result.status is status
    finally:
        case.close()


def test_l2_duplicate_references_keep_weight_but_encode_unique_sha_once(
    tmp_path: Path,
) -> None:
    opposite = _png((91, 41, 11))
    same = _png((92, 42, 12))
    case = _make_production_case(
        tmp_path,
        style_contents=(opposite, same, same),
        l2_status="VALIDATED",
        l3_status="DRAFT",
    )
    try:
        case.evidence_vectors[(91, 41, 11)] = (
            [[-1.0, 0.0], [-1.0, 0.0]],
            [1.0, 0.0],
        )
        case.evidence_vectors[(92, 42, 12)] = (
            [[1.0, 0.0], [1.0, 0.0]],
            [1.0, 0.0],
        )
        result = _result(case, "l2_style")
        assert result.score == pytest.approx(1.0)
        assert result.status is RuleStatus.PASS
        assert case.evidence_calls[(91, 41, 11)] == 1
        assert case.evidence_calls[(92, 42, 12)] == 1
        assert len(case.style_resolver.calls) == 3
    finally:
        case.close()


@pytest.mark.parametrize(
    ("output", "source", "expected"),
    (((1.0, 0.0), (0.0, 1.0), 0.0), ((3.0, 4.0), (6.0, 8.0), 1.0)),
)
def test_l3_uses_normalized_source_and_never_style(
    tmp_path: Path,
    output: tuple[float, float],
    source: tuple[float, float],
    expected: float,
) -> None:
    case = _make_production_case(tmp_path, l2_status="DRAFT", l3_status="VALIDATED")
    try:
        case.evidence_vectors[(10, 200, 10)] = (
            [[1.0, 0.0]],
            list(output),
        )
        case.evidence_vectors[(200, 10, 10)] = (
            [[1.0, 0.0]],
            list(source),
        )
        case.evidence_vectors[(10, 10, 200)] = (
            [[1.0, 0.0]],
            list(output),
        )
        result = _result(case, "l3_diagnostic")
        assert result.score == pytest.approx(expected)
        assert case.style_resolver.calls == []
        assert case.evidence_calls.get((10, 10, 200), 0) == 0
        assert case.evidence_calls[(10, 200, 10)] == 1
        assert case.evidence_calls[(200, 10, 10)] == 1
    finally:
        case.close()


def test_l2_and_l3_share_one_output_encoding(tmp_path: Path) -> None:
    case = _make_production_case(tmp_path)
    try:
        _result(case, "l3_diagnostic")
        assert case.evidence_calls[(10, 200, 10)] == 1
        assert case.evidence_calls[(10, 10, 200)] == 1
        assert case.evidence_calls[(200, 10, 10)] == 1
    finally:
        case.close()


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        (_OOM(), "verification OOM"),
        (RuntimeError("private"), "production verification encoder failed"),
    ),
)
def test_b0_runtime_failures_map_to_frozen_verification_errors(
    tmp_path: Path, failure: Exception, message: str
) -> None:
    case = _make_production_case(tmp_path, l3_status="DRAFT")
    pipeline = case.loaded.borrow_pipeline()
    pipeline.feature_extractor.call_impl = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(failure)
    try:
        with pytest.raises(InfrastructureError, match=f"^{message}$"):
            _result(case, "l2_style")
    finally:
        case.close()


def test_b0_contract_failure_maps_without_partial_results(tmp_path: Path) -> None:
    case = _make_production_case(tmp_path, l3_status="DRAFT")
    case.evidence_vectors[(10, 200, 10)] = (
        [[0.0, 0.0], [0.0, 0.0]],
        [1.0, 0.0],
    )
    try:
        with pytest.raises(
            InfrastructureError, match="^production verification contract violation$"
        ):
            _result(case, "l2_style")
    finally:
        case.close()


def test_hash_correct_but_unencodable_output_has_no_metric_scores(
    tmp_path: Path,
) -> None:
    case = _make_production_case(tmp_path)
    content = b"hash-correct-not-an-image"
    artifact = GeneratedArtifact(
        ArtifactRef(case.artifact.ref.artifact_id, hash_bytes(content)),
        content,
        case.request.request_hash,
        case.request.generation_fingerprint,
    )
    case.artifact_resolver.value = artifact
    try:
        verifier, rules = _bind(case)
        results = verifier.verify((artifact.ref,), rules)
        metrics = tuple(
            result for result in results if result.rule_id.value.startswith("l")
        )
        by_id = {result.rule_id.value: result for result in metrics}
        for rule_id in ("l2_style", "l3_diagnostic"):
            assert by_id[rule_id].status is RuleStatus.UNVERIFIABLE
            assert by_id[rule_id].score is None
    finally:
        case.close()


def test_hash_correct_but_unencodable_style_is_l2_unverifiable(tmp_path: Path) -> None:
    content = _animated_rgb_webp()
    with Image.open(BytesIO(content)) as image:
        assert image.mode == "RGB"
        assert image.n_frames == 2
    case = _make_production_case(
        tmp_path,
        style_contents=(content,),
        l2_status="VALIDATED",
        l3_status="DRAFT",
    )
    try:
        reference = case.request.style_references[0]
        assert case.style_resolver.values[reference] == content
        assert reference.sha256 == hash_bytes(content)
        result = _result(case, "l2_style")
        assert result.status is RuleStatus.UNVERIFIABLE
        assert result.score is None
    finally:
        case.close()


def test_create_rejects_forged_prepared_source_before_any_io(
    tmp_path: Path,
) -> None:
    case = _make_production_case(tmp_path, l2_status="DRAFT", l3_status="VALIDATED")
    object.__setattr__(case.request.source, "content", b"tampered")
    try:
        with pytest.raises(
            DomainError, match="^invalid production verifier dependency$"
        ):
            _result(case, "l3_diagnostic")
        assert case.artifact_resolver.calls == []
        assert case.style_resolver.calls == []
        assert case.evidence_calls == {}
    finally:
        case.close()


def test_owner_close_during_bound_lifetime_is_contract_violation(
    tmp_path: Path,
) -> None:
    case = _make_production_case(tmp_path)
    verifier, rules = _bind(case)
    case.loaded.close()
    try:
        with pytest.raises(
            InfrastructureError, match="^production verification contract violation$"
        ):
            verifier.verify((case.artifact.ref,), rules)
    finally:
        case.close()


def test_owner_close_in_artifact_resolver_is_contract_violation_without_partial(
    tmp_path: Path,
) -> None:
    case = _make_production_case(tmp_path)
    production = importlib.import_module("specstyle.verification.production")
    factory = production._create_production_verifier_factory(
        case.loaded, case.allowlist(production)
    )

    def close_owner(reference: ArtifactRef, /) -> GeneratedArtifact:
        assert reference == case.artifact.ref
        case.loaded.close()
        return case.artifact

    verifier = factory.create(
        case.request,
        case.plan,
        close_owner,
        case.style_resolver,
    )
    try:
        with pytest.raises(
            InfrastructureError, match="^production verification contract violation$"
        ):
            verifier.verify((case.artifact.ref,), case.plan.applicable_rule_definitions)
        assert case.evidence_calls == {}
    finally:
        case.close()


@pytest.mark.parametrize("failure", (False, True))
def test_artifact_callback_threshold_drift_has_contract_priority(
    tmp_path: Path, failure: bool
) -> None:
    case = _make_production_case(tmp_path)

    def resolver(reference: ArtifactRef, /) -> GeneratedArtifact:
        assert reference == case.artifact.ref
        return _callback_result(case, failure)

    verifier, rules = _bind_with(case, resolver, case.style_resolver)
    try:
        with pytest.raises(
            InfrastructureError, match="^production verification contract violation$"
        ):
            verifier.verify((case.artifact.ref,), rules)
    finally:
        case.close()


@pytest.mark.parametrize("failure", (False, True))
def test_style_callback_threshold_drift_has_contract_priority(
    tmp_path: Path, failure: bool
) -> None:
    case = _make_production_case(tmp_path, l3_status="DRAFT")

    def resolver(reference: object, /) -> bytes:
        _drift_l2_threshold(case)
        if failure:
            raise RuntimeError("private callback failure")
        return case.style_resolver.values[reference]

    verifier, rules = _bind_with(case, case.artifact_resolver, resolver)
    try:
        with pytest.raises(
            InfrastructureError, match="^production verification contract violation$"
        ):
            verifier.verify((case.artifact.ref,), rules)
    finally:
        case.close()


@pytest.mark.parametrize("failure", (False, True))
def test_encoder_callback_threshold_drift_has_contract_priority(
    tmp_path: Path, failure: bool
) -> None:
    case = _make_production_case(tmp_path, l3_status="DRAFT")
    pipeline = case.loaded.borrow_pipeline()
    original = pipeline.image_encoder.call_impl

    def encode(*args: object, **kwargs: object) -> object:
        _drift_l2_threshold(case)
        if failure:
            raise RuntimeError("private callback failure")
        return original(*args, **kwargs)

    pipeline.image_encoder.call_impl = encode
    try:
        with pytest.raises(
            InfrastructureError, match="^production verification contract violation$"
        ):
            _result(case, "l2_style")
    finally:
        case.close()


@pytest.mark.parametrize("failure", (False, True))
def test_metric_callback_threshold_drift_has_contract_priority(
    tmp_path: Path, failure: bool
) -> None:
    case = _make_production_case(tmp_path, l3_status="DRAFT")
    original = case.torch.dot

    def dot(left: object, right: object) -> object:
        _drift_l2_threshold(case)
        if failure:
            raise RuntimeError("private callback failure")
        return original(left, right)

    case.torch.dot = dot
    try:
        with pytest.raises(
            InfrastructureError, match="^production verification contract violation$"
        ):
            _result(case, "l2_style")
    finally:
        case.close()


@pytest.mark.parametrize(
    ("mode", "l2_status", "l3_status", "missing"),
    (
        ("only_l1_execution", "DRAFT", "DRAFT", False),
        ("calibrated_metrics", "CALIBRATED", "CALIBRATED", False),
        ("missing_artifact", "DRAFT", "DRAFT", True),
    ),
)
def test_owner_close_after_resolver_fails_on_every_early_return_path(
    tmp_path: Path,
    mode: str,
    l2_status: str,
    l3_status: str,
    missing: bool,
) -> None:
    case = _make_production_case(tmp_path, l2_status=l2_status, l3_status=l3_status)

    def resolver(_reference: ArtifactRef, /) -> GeneratedArtifact | None:
        case.loaded.close()
        return None if missing else case.artifact

    verifier, rules = _bind_with(case, resolver, case.style_resolver)
    try:
        with pytest.raises(
            InfrastructureError, match="^production verification contract violation$"
        ):
            verifier.verify((case.artifact.ref,), rules)
        assert case.evidence_calls == {}
    finally:
        case.close()


@pytest.mark.parametrize("failure", (False, True))
def test_provenance_drift_in_artifact_callback_has_contract_priority(
    tmp_path: Path, failure: bool
) -> None:
    case = _make_production_case(tmp_path)
    production = importlib.import_module("specstyle.verification.production")
    allowlist = case.allowlist(production)
    factory = production._create_production_verifier_factory(case.loaded, allowlist)

    def resolver(_reference: ArtifactRef, /) -> GeneratedArtifact:
        changed = type(allowlist.processor_provenance)(
            "99.0.0",
            allowlist.processor_provenance.class_fqname,
            allowlist.processor_provenance.config_sha256,
        )
        object.__setattr__(allowlist, "processor_provenance", changed)
        if failure:
            raise RuntimeError("private callback failure")
        return case.artifact

    verifier = factory.create(case.request, case.plan, resolver, case.style_resolver)
    try:
        with pytest.raises(
            InfrastructureError, match="^production verification contract violation$"
        ):
            verifier.verify((case.artifact.ref,), case.plan.applicable_rule_definitions)
    finally:
        case.close()


@pytest.mark.parametrize("failure", (False, True))
def test_provenance_drift_in_encoder_callback_has_contract_priority(
    tmp_path: Path, failure: bool
) -> None:
    case = _make_production_case(tmp_path, l3_status="DRAFT")
    pipeline = case.loaded.borrow_pipeline()
    original = pipeline.image_encoder.call_impl

    def encode(*args: object, **kwargs: object) -> object:
        provenance = case.loaded._processor_provenance
        object.__setattr__(provenance, "transformers_version", "99.0.0")
        if failure:
            raise RuntimeError("private callback failure")
        return original(*args, **kwargs)

    pipeline.image_encoder.call_impl = encode
    try:
        with pytest.raises(
            InfrastructureError, match="^production verification contract violation$"
        ):
            _result(case, "l2_style")
    finally:
        case.close()


def test_missing_style_return_rechecks_threshold_drift(tmp_path: Path) -> None:
    case = _make_production_case(tmp_path, l3_status="DRAFT")

    def resolver(_reference: object, /) -> None:
        _drift_l2_threshold(case)
        return None

    verifier, rules = _bind_with(case, case.artifact_resolver, resolver)
    try:
        with pytest.raises(
            InfrastructureError, match="^production verification contract violation$"
        ):
            verifier.verify((case.artifact.ref,), rules)
        assert case.evidence_calls == {}
    finally:
        case.close()
