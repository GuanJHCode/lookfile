"""Pure static contracts for production verifier binding."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from specstyle.domain.enums import RuleLevel, RuleScope, StaticApplicability
from specstyle.domain.artifacts import AssetRef
from specstyle.domain.identifiers import AssetId, Identifier, RuleId, Sha256
from specstyle.errors import DomainError
from specstyle.generation.image_evidence import _ProcessorProvenance
from specstyle.generation.requests import GenerationRequest
from specstyle.spec.compiled_models import (
    CompiledExecutionGraph,
    CompiledRule,
    CompiledStyleSpec,
    CompiledVerificationPlan,
    CompilerContext,
    L3PluginCapability,
    ResourcePin,
    RuleCapability,
    ThresholdProfileCapability,
)
from specstyle.spec.compiler import compile_style_spec
from specstyle.verification.l1.production_bindings import (
    _validate_production_l1_rule_registry,
)

__all__ = ()

_L2_METRIC = Identifier("reference_style_statistics_similarity")
_L3_METRIC = Identifier("subject_semantic_similarity")


class _ProductionContractViolation(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _LoadedVerificationBinding:
    environment_hash: Sha256
    runtime: tuple[str, str, str, str]
    profile: str
    base_pin: ResourcePin
    ip_adapter_pin: ResourcePin
    controlnet_pin: ResourcePin
    evidence_pin: ResourcePin
    evidence_preprocessing_version: str
    evidence_layer: str

    def __post_init__(self) -> None:
        if (
            type(self.environment_hash) is not Sha256
            or type(self.runtime) is not tuple
            or len(self.runtime) != 4
            or any(type(value) is not str for value in self.runtime)
            or self.profile != "production"
            or any(
                type(pin) is not ResourcePin
                for pin in (
                    self.base_pin,
                    self.ip_adapter_pin,
                    self.controlnet_pin,
                    self.evidence_pin,
                )
            )
            or type(self.evidence_preprocessing_version) is not str
            or type(self.evidence_layer) is not str
        ):
            raise _ProductionContractViolation


@dataclass(frozen=True, slots=True)
class _FactoryIssuedState:
    loaded: object = field(repr=False, compare=False)
    allowlist: object = field(repr=False, compare=False)
    evidence: object = field(repr=False, compare=False)
    torch: object = field(repr=False, compare=False)
    context: CompilerContext = field(repr=False, compare=False)
    context_snapshot: CompilerContext = field(repr=False, compare=False)
    mapping_identities: tuple[object, ...] = field(repr=False, compare=False)
    mappings: tuple[tuple[RuleId, str], ...] = field(repr=False, compare=False)
    loaded_binding: _LoadedVerificationBinding = field(repr=False, compare=False)
    provenance_identity: object = field(repr=False, compare=False)
    provenance: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _CanonicalProductionState:
    factory: object = field(repr=False, compare=False)
    issued: _FactoryIssuedState = field(repr=False, compare=False)
    live_request: object = field(repr=False, compare=False)
    live_plan: CompiledVerificationPlan = field(repr=False, compare=False)
    artifact_resolver: object = field(repr=False, compare=False)
    style_resolver: object = field(repr=False, compare=False)
    live_rules: tuple[object, ...] = field(repr=False, compare=False)
    request: object = field(repr=False, compare=False)
    plan: CompiledVerificationPlan = field(repr=False, compare=False)
    mappings: tuple[tuple[RuleId, str], ...] = field(repr=False, compare=False)
    loaded_binding: _LoadedVerificationBinding = field(repr=False, compare=False)
    provenance: object = field(repr=False, compare=False)
    digest: Sha256 = field(repr=False, compare=False)


def _pin_parts(pin: ResourcePin) -> tuple[str, str, str]:
    return pin.id, pin.revision, pin.sha256.value


def _copy_pin(pin: object) -> ResourcePin:
    if type(pin) is not ResourcePin:
        raise _ProductionContractViolation
    return ResourcePin(pin.id, pin.revision, Sha256(pin.sha256.value))


def _descriptor_pin(descriptor: Any) -> ResourcePin:
    return ResourcePin(
        descriptor.model_id,
        descriptor.revision,
        Sha256(descriptor.expected_sha256.value),
    )


def _provenance_snapshot(value: object) -> _ProcessorProvenance:
    if type(value) is not _ProcessorProvenance:
        raise _ProductionContractViolation
    return _ProcessorProvenance(
        value.transformers_version,
        value.class_fqname,
        Sha256(value.config_sha256.value),
    )


def _canonical_l1_mappings(mappings: object) -> tuple[tuple[RuleId, str], ...]:
    if type(mappings) is not tuple:
        raise _ProductionContractViolation
    values = tuple(
        (RuleId(mapping.rule_id.value), mapping.implementation) for mapping in mappings
    )
    _validate_l1_registry(values)
    return values


def _rebuild_request(
    request: GenerationRequest, compiled: CompiledStyleSpec
) -> GenerationRequest:
    styles = tuple(
        AssetRef(AssetId(reference.asset_id.value), Sha256(reference.sha256.value))
        for reference in request.style_references
    )
    return replace(request, compiled_spec=compiled, style_references=styles)


def _binding_digest(
    request: Any,
    plan: CompiledVerificationPlan,
    mappings: tuple[tuple[RuleId, str], ...],
    loaded: _LoadedVerificationBinding,
    provenance: Any,
) -> Sha256:
    pins = (
        loaded.base_pin,
        loaded.ip_adapter_pin,
        loaded.controlnet_pin,
        loaded.evidence_pin,
    )
    material = {
        "schema": "specstyle.production_verifier.binding.v1",
        "compiled": request.compiled_spec.compiled_spec_hash.value,
        "plan": [plan.output_profile, *_pin_parts(plan.output_profile_pin)],
        "request": [request.request_hash.value, request.generation_fingerprint.value],
        "seed": [request.seed.algorithm, request.seed.seed],
        "mappings": [[rule.value, implementation] for rule, implementation in mappings],
        "loaded": [
            loaded.environment_hash.value,
            *loaded.runtime,
            loaded.profile,
            *[part for pin in pins for part in _pin_parts(pin)],
            loaded.evidence_preprocessing_version,
            loaded.evidence_layer,
        ],
        "provenance": [
            provenance.transformers_version,
            provenance.class_fqname,
            provenance.config_sha256.value,
        ],
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return Sha256(hashlib.sha256(encoded).hexdigest())


def _build_canonical_state(
    factory: object,
    issued: _FactoryIssuedState,
    request: GenerationRequest,
    plan: CompiledVerificationPlan,
    artifact_resolver: object,
    style_resolver: object,
    compiled: CompiledStyleSpec,
    canonical_plan: CompiledVerificationPlan,
) -> _CanonicalProductionState:
    canonical_request = _rebuild_request(request, compiled)
    if canonical_request != request:
        raise _ProductionContractViolation
    mappings = tuple(
        (RuleId(rule.value), implementation) for rule, implementation in issued.mappings
    )
    digest = _binding_digest(
        canonical_request,
        canonical_plan,
        mappings,
        issued.loaded_binding,
        issued.provenance,
    )
    return _CanonicalProductionState(
        factory,
        issued,
        request,
        plan,
        artifact_resolver,
        style_resolver,
        plan.applicable_rule_definitions,
        canonical_request,
        canonical_plan,
        mappings,
        issued.loaded_binding,
        issued.provenance,
        digest,
    )


def _one(values: tuple[Any, ...], predicate: Callable[[Any], bool]) -> Any:
    selected = tuple(value for value in values if predicate(value))
    if len(selected) != 1:
        raise _ProductionContractViolation
    return selected[0]


def _clone_compiler_context(context: object) -> CompilerContext:
    if type(context) is not CompilerContext:
        raise _ProductionContractViolation
    try:
        cloned = copy.deepcopy(context)
    except Exception:
        raise _ProductionContractViolation from None
    if type(cloned) is not CompilerContext or cloned != context:
        raise _ProductionContractViolation
    return cloned


def _validate_l1_registry(mappings: tuple[tuple[RuleId, str], ...]) -> None:
    try:
        _validate_production_l1_rule_registry(mappings)
    except DomainError:
        raise _ProductionContractViolation from None


def _profile_matches_binding(
    profile: ThresholdProfileCapability, rule: CompiledRule
) -> bool:
    binding = rule.threshold_binding
    if binding is None:
        return False
    metrics = tuple(
        metric for metric in profile.metrics if metric.metric_id == rule.metric_id
    )
    return (
        len(metrics) == 1
        and profile.pin == binding.profile_pin
        and profile.logical_name == binding.logical_name
        and profile.status == binding.status
        and metrics[0].metric_id == binding.metric_id
        and metrics[0].operator == binding.operator
        and metrics[0].value == binding.value
        and profile.calibration_dataset_sha256 == binding.calibration_dataset_sha256
        and profile.validation_dataset_sha256 == binding.validation_dataset_sha256
        and profile.annotation_protocol_sha256 == binding.annotation_protocol_sha256
    )


def _validate_loaded_binding(
    binding: _LoadedVerificationBinding,
    graph: CompiledExecutionGraph,
    environment_hash: Sha256,
) -> None:
    runtime = graph.runtime
    if (
        environment_hash != binding.environment_hash
        or (
            runtime.rocm_version,
            runtime.torch_version,
            runtime.diffusers_version,
            runtime.dtype,
        )
        != binding.runtime
        or graph.base_model.pin != binding.base_pin
        or graph.ip_adapter.pin != binding.ip_adapter_pin
        or graph.controlnet.pin != binding.controlnet_pin
        or graph.ip_adapter.pin != binding.evidence_pin
    ):
        raise _ProductionContractViolation


def _catalog_rules(
    context: CompilerContext, compiled: CompiledStyleSpec
) -> dict[RuleId, RuleCapability]:
    catalog = _one(
        context.rule_catalogs,
        lambda item: (
            item.pin == compiled.ruleset_pin
            and item.ruleset_version
            == compiled.source_spec.verification.ruleset_version
        ),
    )
    return {rule.rule_id: rule for rule in catalog.rules}


def _validate_l1_bindings(
    mappings: tuple[tuple[RuleId, str], ...],
    applicable: tuple[CompiledRule, ...],
    catalog: dict[RuleId, RuleCapability],
) -> None:
    _validate_l1_registry(mappings)
    l1_rules = tuple(
        rule for rule in applicable if rule.definition.level is RuleLevel.L1
    )
    if {rule_id for rule_id, _ in mappings} != {
        rule.definition.rule_id for rule in l1_rules
    }:
        raise _ProductionContractViolation
    for rule in l1_rules:
        capability = catalog.get(rule.definition.rule_id)
        if (
            capability is None
            or capability.kind != "L1_TECHNICAL"
            or capability.level is not RuleLevel.L1
            or capability.scope is not RuleScope.ITEM
            or capability.threshold_source != "none"
            or capability.metric_id is not None
            or capability.verifier_pin != rule.verifier_pin
            or rule.metric_id is not None
            or rule.threshold_binding is not None
        ):
            raise _ProductionContractViolation


def _validate_l2_binding(
    context: CompilerContext,
    binding: _LoadedVerificationBinding,
    compiled: CompiledStyleSpec,
    rule: CompiledRule,
    capability: RuleCapability,
) -> None:
    encoder = _one(
        context.encoder_capabilities, lambda item: item.pin == binding.evidence_pin
    )
    profiles = tuple(
        profile
        for profile in context.threshold_profiles
        if rule.threshold_binding is not None
        and profile.pin == rule.threshold_binding.profile_pin
    )
    if len(profiles) != 1:
        raise _ProductionContractViolation
    profile, metric = profiles[0], rule.threshold_binding
    valid = (
        capability.kind == "L2_STYLE_FIDELITY"
        and capability.level is RuleLevel.L2
        and capability.scope is RuleScope.ITEM
        and capability.threshold_source == "l2"
        and capability.metric_id == _L2_METRIC == rule.metric_id
        and capability.verifier_pin == rule.verifier_pin
        and profile.source == "l2"
        and profile.style_pack_id == Identifier(compiled.source_spec.style.preset_id)
        and profile.domain_profile == compiled.source_spec.domain.profile
        and profile.encoder_pin == binding.evidence_pin
        and profile.plugin_pin is None
        and encoder.preprocessing_version == binding.evidence_preprocessing_version
        and binding.evidence_layer
        == encoder.layer
        == compiled.l2_encoder.layer
        == "hidden_states[-2]"
        and encoder.distance_function
        == compiled.l2_encoder.distance_function
        == "median_cosine_patch_mean_std_v1"
        and compiled.l2_encoder.preprocessing_version == encoder.preprocessing_version
        and _profile_matches_binding(profile, rule)
    )
    if metric is None or metric.operator != ">=" or not -1.0 <= metric.value <= 1.0:
        valid = False
    if not valid:
        raise _ProductionContractViolation


def _validate_l3_semantics(
    context: CompilerContext,
    plan: CompiledVerificationPlan,
    rule: CompiledRule,
) -> L3PluginCapability:
    plugin = _one(context.l3_plugins, lambda item: item.pin == plan.l3_plugin_pin)
    capability = _one(
        plugin.rules, lambda item: item.rule_id == rule.definition.rule_id
    )
    if (
        capability.kind != "L3_DIAGNOSTIC"
        or capability.level is not RuleLevel.L3
        or capability.scope is not RuleScope.ITEM
        or capability.requirement != "always_advisory"
        or capability.threshold_source != "l3"
        or capability.metric_id != _L3_METRIC
        or rule.metric_id != _L3_METRIC
        or capability.verifier_pin != rule.verifier_pin
        or rule.definition.required
    ):
        raise _ProductionContractViolation
    return plugin


def _validate_l3_binding(
    context: CompilerContext,
    compiled: CompiledStyleSpec,
    plan: CompiledVerificationPlan,
    rule: CompiledRule,
    plugin: L3PluginCapability,
) -> None:
    source = compiled.source_spec
    profiles = tuple(
        profile
        for profile in context.threshold_profiles
        if rule.threshold_binding is not None
        and profile.pin == rule.threshold_binding.profile_pin
    )
    if len(profiles) != 1:
        raise _ProductionContractViolation
    profile, metric = profiles[0], rule.threshold_binding
    valid = (
        not source.domain.fidelity_required
        and plugin.domain_profile == source.domain.profile
        and plugin.domain_verifier_version == source.domain.verifier_version
        and profile.source == "l3"
        and profile.style_pack_id == Identifier(source.style.preset_id)
        and profile.domain_profile == source.domain.profile
        and profile.encoder_pin is None
        and profile.plugin_pin == plugin.pin
        and plan.l3_threshold_profile_pin == profile.pin
        and _profile_matches_binding(profile, rule)
    )
    if metric is None or metric.operator != ">=" or not -1.0 <= metric.value <= 1.0:
        valid = False
    if not valid:
        raise _ProductionContractViolation


def _validate_rule_bindings(
    context: CompilerContext,
    binding: _LoadedVerificationBinding,
    compiled: CompiledStyleSpec,
    plan: CompiledVerificationPlan,
    mappings: tuple[tuple[RuleId, str], ...],
) -> None:
    catalog = _catalog_rules(context, compiled)
    applicable = tuple(
        rule
        for rule in plan.rules
        if rule.definition.applicability is StaticApplicability.APPLICABLE
    )
    _validate_l1_bindings(mappings, applicable, catalog)
    for rule in plan.rules:
        definition = rule.definition
        capability = catalog.get(definition.rule_id)
        if (
            definition.applicability is StaticApplicability.APPLICABLE
            and definition.level is RuleLevel.L2
        ):
            if capability is None:
                raise _ProductionContractViolation
            _validate_l2_binding(context, binding, compiled, rule, capability)
        elif definition.level is RuleLevel.L3:
            plugin = _validate_l3_semantics(context, plan, rule)
            if definition.applicability is StaticApplicability.APPLICABLE:
                _validate_l3_binding(context, compiled, plan, rule, plugin)


def _validate_production_binding(
    binding: _LoadedVerificationBinding,
    compiled: CompiledStyleSpec,
    generation_profile: str,
    output_profile: str,
    environment_hash: Sha256,
    graph: CompiledExecutionGraph,
    plan: CompiledVerificationPlan,
    context: CompilerContext,
    mappings: tuple[tuple[RuleId, str], ...],
) -> tuple[CompiledStyleSpec, CompiledVerificationPlan]:
    try:
        context_snapshot = _clone_compiler_context(context)
        source_type = type(compiled.source_spec)
        source_snapshot = source_type.model_validate(
            compiled.source_spec.model_dump(mode="python", round_trip=True)
        )
        independent = compile_style_spec(source_snapshot, context_snapshot)
    except Exception:
        raise _ProductionContractViolation from None
    matching = tuple(
        candidate
        for candidate in compiled.verification_plans
        if candidate.output_profile == output_profile
    )
    canonical = tuple(
        candidate
        for candidate in independent.verification_plans
        if candidate.output_profile == output_profile
    )
    if (
        independent != compiled
        or generation_profile != "production"
        or len(matching) != 1
        or len(canonical) != 1
        or plan is not matching[0]
        or any(
            rule.definition.scope is not RuleScope.ITEM
            for rule in plan.rules
            if rule.definition.applicability is StaticApplicability.APPLICABLE
        )
    ):
        raise _ProductionContractViolation
    _validate_loaded_binding(binding, graph, environment_hash)
    _validate_rule_bindings(
        context_snapshot, binding, independent, canonical[0], mappings
    )
    return independent, canonical[0]
