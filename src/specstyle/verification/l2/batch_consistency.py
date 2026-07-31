"""Batch L2 style consistency and outlier attribution."""

from __future__ import annotations

import math

from specstyle.domain.enums import RuleStatus
from specstyle.domain.identifiers import ArtifactId, RuleId
from specstyle.errors import DomainError
from specstyle.verification.l2.encoder import StyleFeature
from specstyle.verification.l2.threshold_profile import ThresholdProfile
from specstyle.verification.rule_models import RuleResult

RULE_BATCH_CONSISTENCY = RuleId("L2_BATCH_STYLE_CONSISTENCY")


def _centroid(features: tuple[StyleFeature, ...]) -> tuple[float, ...]:
    dim = len(features[0].vector)
    acc = [0.0] * dim
    for feat in features:
        if len(feat.vector) != dim:
            raise DomainError("feature dimension mismatch")
        for i, v in enumerate(feat.vector):
            acc[i] += v
    n = float(len(features))
    return tuple(x / n for x in acc)


def _distance(vec: tuple[float, ...], center: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, center, strict=True)))


def evaluate_batch_consistency(
    items: tuple[tuple[ArtifactId, StyleFeature], ...],
    profile: ThresholdProfile,
    /,
) -> RuleResult:
    if type(items) is not tuple or not items or type(profile) is not ThresholdProfile:
        raise DomainError("invalid batch consistency inputs")
    ids = tuple(i[0] for i in items)
    features = tuple(i[1] for i in items)
    if any(type(i) is not ArtifactId for i in ids):
        raise DomainError("invalid artifact ids")
    if any(type(f) is not StyleFeature for f in features):
        raise DomainError("invalid features")
    center = _centroid(features)
    distances = [_distance(f.vector, center) for f in features]
    if len(distances) == 1:
        dispersion = 0.0
    else:
        mean_d = sum(distances) / len(distances)
        dispersion = math.sqrt(
            sum((d - mean_d) ** 2 for d in distances) / len(distances)
        )
    metric = next(
        (t for t in profile.thresholds if t.metric_id == "batch_style_dispersion_max"),
        None,
    )
    if metric is None:
        return RuleResult(
            RULE_BATCH_CONSISTENCY, RuleStatus.UNVERIFIABLE, ids, dispersion
        )
    # Outliers: above median + max(threshold, 1e-6)
    ordered = sorted(distances)
    median = ordered[len(ordered) // 2]
    cutoff = max(metric.value, median + 1e-6)
    outliers = tuple(
        aid for aid, dist in zip(ids, distances, strict=True) if dist > cutoff
    )
    if dispersion > metric.value:
        affected = outliers if outliers else ids
        return RuleResult(RULE_BATCH_CONSISTENCY, RuleStatus.FAIL, affected, dispersion)
    return RuleResult(RULE_BATCH_CONSISTENCY, RuleStatus.PASS, ids, dispersion)
