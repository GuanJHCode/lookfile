"""Fail-closed production verifier static binding tests."""

from __future__ import annotations

import importlib
from dataclasses import replace

import pytest

from specstyle.domain.enums import RuleScope
from specstyle.domain.identifiers import Identifier, RuleId, Sha256
from specstyle.errors import DomainError
from specstyle.generation.requests import GenerationRequest
from specstyle.spec.compiler import compile_style_spec
from tests.unit.verification._production_fixtures import (
    _ProductionCase,
    production_case as production_case,
)


def _request_for_context(
    case: _ProductionCase,
    context: object,
    *,
    environment_hash: Sha256 | None = None,
) -> tuple[GenerationRequest, object]:
    compiled = compile_style_spec(case.request.compiled_spec.source_spec, context)
    original = case.request
    request = GenerationRequest(
        original.job_id,
        original.attempt_id,
        original.parent_attempt_id,
        compiled,
        original.generation_profile,
        original.output_profile,
        original.source,
        original.style_references,
        original.prompt,
        original.control_input,
        original.variation_index,
        original.environment_hash if environment_hash is None else environment_hash,
        original.execution_parameters,
    )
    plans = tuple(
        plan
        for plan in compiled.verification_plans
        if plan.output_profile == request.output_profile
    )
    assert len(plans) == 1
    return request, plans[0]


def _create(
    case: _ProductionCase,
    production: object,
    *,
    context: object | None = None,
    mappings: tuple[tuple[RuleId, str], ...] | None = None,
    request: GenerationRequest | None = None,
    plan: object | None = None,
) -> object:
    selected_context = case.compiler_context if context is None else context
    allowlist = case.allowlist(
        production, compiler_context=selected_context, mappings=mappings
    )
    factory = production._create_production_verifier_factory(case.loaded, allowlist)
    return factory.create(
        case.request if request is None else request,
        case.plan if plan is None else plan,
        case.artifact_resolver,
        case.style_resolver,
    )


@pytest.mark.parametrize("mapping_change", ("missing", "extra"))
def test_create_rejects_nonexact_l1_mapping_coverage(
    production_case: _ProductionCase, mapping_change: str
) -> None:
    production = importlib.import_module("specstyle.verification.production")
    mappings = production_case.l1_mappings
    if mapping_change == "missing":
        mappings = mappings[:-1]
    else:
        mappings = (*mappings, (RuleId("extra"), mappings[0][1]))

    with pytest.raises(DomainError, match="^invalid production verifier dependency$"):
        _create(production_case, production, mappings=mappings)


@pytest.mark.parametrize("mutation", ("metric", "operator", "value"))
def test_create_rejects_nonproduction_l2_metric_contract(
    production_case: _ProductionCase, mutation: str
) -> None:
    production = importlib.import_module("specstyle.verification.production")
    context = production_case.compiler_context
    catalog = context.rule_catalogs[0]
    profile = context.threshold_profiles[0]
    rules = list(catalog.rules)
    l2_index = next(
        index for index, rule in enumerate(rules) if rule.rule_id == RuleId("l2_style")
    )
    metric = profile.metrics[0]
    if mutation == "metric":
        replacement_id = Identifier("other_similarity")
        rules[l2_index] = replace(rules[l2_index], metric_id=replacement_id)
        metric = replace(metric, metric_id=replacement_id)
    elif mutation == "operator":
        metric = replace(metric, operator="<=")
    else:
        metric = replace(metric, value=1.5)
    context = replace(
        context,
        rule_catalogs=(replace(catalog, rules=tuple(rules)),),
        threshold_profiles=(
            replace(profile, metrics=(metric,)),
            *context.threshold_profiles[1:],
        ),
    )
    request, plan = _request_for_context(production_case, context)

    with pytest.raises(DomainError, match="^invalid production verifier dependency$"):
        _create(
            production_case,
            production,
            context=context,
            request=request,
            plan=plan,
        )


@pytest.mark.parametrize("mutation", ("kind", "required"))
def test_create_rejects_coarse_or_required_l3_rule(
    production_case: _ProductionCase, mutation: str
) -> None:
    production = importlib.import_module("specstyle.verification.production")
    context = production_case.compiler_context
    plugin = context.l3_plugins[0]
    rule = plugin.rules[0]
    rule = replace(
        rule,
        kind="L3_DOMAIN_FIDELITY" if mutation == "kind" else rule.kind,
        requirement="always_required" if mutation == "required" else rule.requirement,
    )
    context = replace(context, l3_plugins=(replace(plugin, rules=(rule,)),))
    request, plan = _request_for_context(production_case, context)

    with pytest.raises(DomainError, match="^invalid production verifier dependency$"):
        _create(
            production_case,
            production,
            context=context,
            request=request,
            plan=plan,
        )


@pytest.mark.parametrize("mutation", ("coarse", "required"))
def test_create_rejects_nonadvisory_l3_semantics_when_rule_is_not_applicable(
    production_case: _ProductionCase, mutation: str
) -> None:
    production = importlib.import_module("specstyle.verification.production")
    context = production_case.compiler_context
    plugin = context.l3_plugins[0]
    rule = plugin.rules[0]
    rule = replace(
        rule,
        kind="L3_DOMAIN_FIDELITY" if mutation == "coarse" else rule.kind,
        requirement="always_required" if mutation == "required" else rule.requirement,
        supported_output_profiles=("background_sequence",),
    )
    context = replace(context, l3_plugins=(replace(plugin, rules=(rule,)),))
    request, plan = _request_for_context(production_case, context)
    l3_rule = next(item for item in plan.rules if item.definition.level.value == "L3")
    assert plan.l3_status == "NOT_APPLICABLE"
    assert plan.l3_reason == "NO_APPLICABLE_RULE"
    assert l3_rule.definition.applicability.value == "NOT_APPLICABLE"

    with pytest.raises(DomainError, match="^invalid production verifier dependency$"):
        _create(
            production_case,
            production,
            context=context,
            request=request,
            plan=plan,
        )
    assert production_case.artifact_resolver.calls == []
    assert production_case.style_resolver.calls == []
    assert production_case.evidence_calls == {}


def test_create_rejects_applicable_batch_rule(
    production_case: _ProductionCase,
) -> None:
    production = importlib.import_module("specstyle.verification.production")
    context = production_case.compiler_context
    catalog = context.rule_catalogs[0]
    rules = tuple(
        replace(rule, supported_output_profiles=("xhs_grid",))
        if rule.scope is RuleScope.BATCH
        else rule
        for rule in catalog.rules
    )
    l2_profile = context.threshold_profiles[0]
    batch_rule = next(rule for rule in rules if rule.scope is RuleScope.BATCH)
    batch_metric = replace(l2_profile.metrics[0], metric_id=batch_rule.metric_id)
    context = replace(
        context,
        rule_catalogs=(replace(catalog, rules=rules),),
        threshold_profiles=(
            replace(l2_profile, metrics=(*l2_profile.metrics, batch_metric)),
            *context.threshold_profiles[1:],
        ),
    )
    request, plan = _request_for_context(production_case, context)

    with pytest.raises(DomainError, match="^invalid production verifier dependency$"):
        _create(
            production_case,
            production,
            context=context,
            request=request,
            plan=plan,
        )


def test_create_rejects_request_environment_not_owned_by_loaded_pipeline(
    production_case: _ProductionCase,
) -> None:
    production = importlib.import_module("specstyle.verification.production")
    request, _ = _request_for_context(
        production_case,
        production_case.compiler_context,
        environment_hash=Sha256("f" * 64),
    )

    with pytest.raises(DomainError, match="^invalid production verifier dependency$"):
        _create(production_case, production, request=request)


def test_create_rejects_equal_but_nonmember_plan(
    production_case: _ProductionCase,
) -> None:
    production = importlib.import_module("specstyle.verification.production")

    with pytest.raises(DomainError, match="^invalid production verifier dependency$"):
        _create(production_case, production, plan=replace(production_case.plan))
