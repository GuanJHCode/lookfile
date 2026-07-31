"""EVAL-003 statistics from arm results — full denominator, bootstrap CI."""

from __future__ import annotations

import random
from dataclasses import dataclass

from specstyle.errors import DomainError
from specstyle.evaluation.arms import ArmResult


@dataclass(frozen=True, slots=True)
class ArmStats:
    arm: str
    n_inputs: int
    usable_count: int
    human_usable_yield: float
    mean_generations: float
    yield_ci95: tuple[float, float]


def summarize_arm(
    result: ArmResult, *, bootstrap: int = 200, seed: int = 0
) -> ArmStats:
    if type(result) is not ArmResult:
        raise DomainError("invalid arm result")
    n = len(result.records)
    if n == 0:
        raise DomainError("empty arm result")
    usable = sum(1 for r in result.records if r.usable)
    hy = usable / float(n)
    gens = [r.generations_used for r in result.records]
    mean_g = sum(gens) / float(n)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(max(1, bootstrap)):
        picks = [result.records[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(1 for p in picks if p.usable) / float(n))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return ArmStats(result.arm, n, usable, hy, mean_g, (lo, hi))


def compare_arms(results: tuple[ArmResult, ...]) -> tuple[ArmStats, ...]:
    if type(results) is not tuple or not results:
        raise DomainError("empty results")
    return tuple(summarize_arm(r, seed=i) for i, r in enumerate(results))
