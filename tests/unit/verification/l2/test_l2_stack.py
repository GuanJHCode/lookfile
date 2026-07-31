"""L2-001..004 shipped path tests."""

from __future__ import annotations

from pathlib import Path

from specstyle.domain.enums import RuleStatus
from specstyle.domain.identifiers import ArtifactId, Sha256
from specstyle.observability.hashing import hash_bytes
from specstyle.verification.l2.batch_consistency import evaluate_batch_consistency
from specstyle.verification.l2.encoder import (
    EncoderPin,
    FakeStyleEncoder,
    FeatureCache,
    encode_with_cache,
)
from specstyle.verification.l2.fidelity import evaluate_style_fidelity
from specstyle.verification.l2.metrics import cosine_similarity
from specstyle.verification.l2.threshold_profile import (
    MetricThreshold,
    ThresholdProfile,
    dump_threshold_profile,
    load_threshold_profile,
)


def _pin() -> EncoderPin:
    return EncoderPin("style-enc", "r1", "layer0", "prep1")


def _sha(ch: str = "a") -> Sha256:
    return Sha256((ch if ch in "0123456789abcdef" else "a") * 64)


def _profile(**kwargs) -> ThresholdProfile:
    base = dict(
        profile_id="p1",
        revision="r1",
        status="VALIDATED",
        style_pack_id="s1",
        domain_profile="product_instance",
        encoder_id="style-enc",
        encoder_revision="r1",
        preprocessing_version="prep1",
        calibration_dataset_sha256=_sha("c"),
        validation_dataset_sha256=_sha("d"),
        thresholds=(
            MetricThreshold("reference_style_fidelity_min", "gte", 0.0),
            MetricThreshold("batch_style_dispersion_max", "lte", 2.0),
        ),
    )
    base.update(kwargs)
    return ThresholdProfile(**base)


def test_encoder_cache_hit_and_miss_on_pin_change(tmp_path: Path) -> None:
    enc = FakeStyleEncoder(_pin())
    cache = FeatureCache(tmp_path)
    data = b"\x89PNG-fake"
    h = hash_bytes(data)
    f1 = encode_with_cache(enc, cache, data, h)
    f2 = encode_with_cache(enc, cache, data, h)
    assert f1.vector == f2.vector
    other = FakeStyleEncoder(EncoderPin("style-enc", "r2", "layer0", "prep1"))
    f3 = encode_with_cache(other, cache, data, h)
    assert f3.vector != f1.vector or f3.pin.revision == "r2"


def test_threshold_profile_roundtrip_and_required() -> None:
    profile = _profile()
    raw = dump_threshold_profile(profile)
    loaded = load_threshold_profile(raw)
    assert loaded.status == "VALIDATED"
    loaded.require_for_gate()
    draft = _profile(status="DRAFT")
    try:
        draft.require_for_gate()
        raise AssertionError("expected DomainError")
    except Exception as exc:
        assert "VALIDATED" in str(exc)


def test_fidelity_empty_refs_unverifiable() -> None:
    enc = FakeStyleEncoder(_pin())
    out = enc.encode(b"img", _sha("1"))
    result = evaluate_style_fidelity(ArtifactId("a1"), out, (), _profile())
    assert result.status is RuleStatus.UNVERIFIABLE


def test_fidelity_pass_with_self_reference() -> None:
    enc = FakeStyleEncoder(_pin())
    feat = enc.encode(b"same", _sha("2"))
    result = evaluate_style_fidelity(ArtifactId("a1"), feat, (feat,), _profile())
    assert result.status is RuleStatus.PASS
    assert result.score is not None
    assert cosine_similarity(feat, feat) > 0.99


def test_batch_consistency_outlier_affected_ids() -> None:
    enc = FakeStyleEncoder(_pin())
    a = enc.encode(b"aaa", _sha("1"))
    b = enc.encode(b"aaa", _sha("1"))  # same bytes → same vector
    # force far vector
    from specstyle.verification.l2.encoder import StyleFeature

    far = StyleFeature(tuple(1.0 for _ in a.vector), a.pin, _sha("9"))
    items = (
        (ArtifactId("i1"), a),
        (ArtifactId("i2"), b),
        (ArtifactId("i3"), far),
    )
    result = evaluate_batch_consistency(
        items,
        _profile(
            thresholds=(MetricThreshold("batch_style_dispersion_max", "lte", 0.01),)
        ),
    )
    assert result.status is RuleStatus.FAIL
    assert ArtifactId("i3") in result.affected_artifact_ids
