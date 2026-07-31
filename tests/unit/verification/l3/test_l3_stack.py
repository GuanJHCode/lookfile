"""L3-001..003 CPU path tests."""

from __future__ import annotations

from specstyle.domain.enums import RuleStatus
from specstyle.domain.identifiers import ArtifactId
from specstyle.verification.l3.base import DomainContext, resolve_applicability
from specstyle.verification.l3.mask_provider import DictMaskProvider, Mask
from specstyle.verification.l3.product_features import evaluate_product_features
from specstyle.verification.l3.product_geometry import ProductGeometryPlugin, mask_iou
from specstyle.verification.l3.product_instance import combine_geometry_and_features


def _mask(w: int, h: int, ones: set[tuple[int, int]]) -> Mask:
    data = []
    for y in range(h):
        for x in range(w):
            data.append(1 if (x, y) in ones else 0)
    return Mask(w, h, tuple(data))


def test_applicability_not_for_wrong_domain() -> None:
    plugin = ProductGeometryPlugin(DictMaskProvider({}), DictMaskProvider({}), (4, 4))
    ctx = DomainContext("face_identity", "product_geometry", "v1")
    assert resolve_applicability(plugin, ctx) == "NOT_APPLICABLE"


def test_missing_mask_unverifiable() -> None:
    aid = ArtifactId("a1")
    plugin = ProductGeometryPlugin(DictMaskProvider({}), DictMaskProvider({}), (4, 4))
    ctx = DomainContext("product_instance", "product_geometry", "v1")
    result = plugin.verify(aid, ctx)
    assert result.status is RuleStatus.UNVERIFIABLE


def test_geometry_iou_and_coverage() -> None:
    aid = ArtifactId("a1")
    m = _mask(4, 4, {(1, 1), (2, 1), (1, 2), (2, 2)})
    ref = _mask(4, 4, {(1, 1), (2, 1), (1, 2), (2, 2)})
    plugin = ProductGeometryPlugin(
        DictMaskProvider({aid: m}),
        DictMaskProvider({aid: ref}),
        (4, 4),
        min_iou=0.9,
        min_coverage=0.1,
        max_coverage=0.5,
    )
    ctx = DomainContext("product_instance", "product_geometry", "v1")
    assert plugin.verify(aid, ctx).status is RuleStatus.PASS
    assert mask_iou(m, ref) == 1.0


def test_features_and_composite() -> None:
    aid = ArtifactId("a1")
    m = _mask(2, 2, {(0, 0), (1, 0)})
    provider = DictMaskProvider({aid: m})
    img = b"product-bytes"
    feat = evaluate_product_features(
        aid, img, provider, img, provider, (2, 2), min_similarity=1.0
    )
    assert feat.status is RuleStatus.PASS
    from specstyle.verification.rule_models import RuleResult
    from specstyle.domain.identifiers import RuleId

    geo = RuleResult(RuleId("g"), RuleStatus.PASS, (aid,), 0.9)
    composite = combine_geometry_and_features(aid, geo, feat)
    assert composite.status is RuleStatus.PASS
    bad = evaluate_product_features(
        aid, img, provider, b"other", provider, (2, 2), min_similarity=1.0
    )
    assert bad.status is RuleStatus.FAIL
    assert combine_geometry_and_features(aid, geo, bad).status is RuleStatus.FAIL
