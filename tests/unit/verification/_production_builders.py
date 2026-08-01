"""Static contract builders for production verifier tests."""

from __future__ import annotations

from typing import Any

from specstyle.domain.enums import RuleLevel, RuleScope
from specstyle.domain.identifiers import Identifier, RuleId
from specstyle.observability.hashing import hash_bytes
from specstyle.spec.compiled_models import (
    CompilerContext,
    EncoderCapability,
    L3PluginCapability,
    ModelCapability,
    OutputProfileCapability,
    ResourcePin,
    RuleCapability,
    RuleCatalogCapability,
    RuntimeCapability,
    StrengthMappingCapability,
    StrengthMappingEntry,
    ThresholdMetricCapability,
    ThresholdProfileCapability,
)
from specstyle.spec.models import StyleSpecV1

_L2_METRIC = Identifier("reference_style_statistics_similarity")
_L3_METRIC = Identifier("subject_semantic_similarity")
_L1_MAPPINGS = (
    (RuleId("l1_bundle"), "technical_rgb_png_bundle_v1"),
    (RuleId("l1_decode"), "decode_png_rgb_no_metadata_v1"),
    (RuleId("l1_dimensions"), "dimensions_exact_v1"),
    (RuleId("l1_pixels"), "pixels_nonblank_v1"),
)


def _pin(identifier: str, material: str) -> ResourcePin:
    return ResourcePin(identifier, "r1", hash_bytes(material.encode()))


def _l1_rules() -> tuple[RuleCapability, ...]:
    return tuple(
        RuleCapability(
            rule_id,
            "L1_TECHNICAL",
            RuleLevel.L1,
            RuleScope.ITEM,
            "always_required",
            ("product_instance",),
            ("xhs_grid",),
            _pin(f"{rule_id.value}-verifier", rule_id.value),
            "none",
            None,
            index,
            (),
        )
        for index, (rule_id, _implementation) in enumerate(_L1_MAPPINGS)
    )


def _l2_rules() -> tuple[RuleCapability, RuleCapability]:
    style = RuleCapability(
        RuleId("l2_style"),
        "L2_STYLE_FIDELITY",
        RuleLevel.L2,
        RuleScope.ITEM,
        "always_advisory",
        ("product_instance",),
        ("xhs_grid",),
        _pin("l2-verifier", "l2-verifier"),
        "l2",
        _L2_METRIC,
        10,
        (),
    )
    batch = RuleCapability(
        RuleId("l2_batch"),
        "L2_BATCH_CONSISTENCY",
        RuleLevel.L2,
        RuleScope.BATCH,
        "always_advisory",
        ("product_instance",),
        ("background_sequence",),
        _pin("batch-verifier", "batch-verifier"),
        "l2",
        Identifier("batch_metric"),
        11,
        (),
    )
    return style, batch


def _l3_plugin(
    plugin_pin: ResourcePin, kind: str, requirement: str
) -> L3PluginCapability:
    rule = RuleCapability(
        RuleId("l3_diagnostic"),
        kind,
        RuleLevel.L3,
        RuleScope.ITEM,
        requirement,
        ("product_instance",),
        ("xhs_grid",),
        _pin("l3-verifier", "l3-verifier"),
        "l3",
        _L3_METRIC,
        20,
        (),
    )
    return L3PluginCapability(
        plugin_pin,
        "product_instance",
        "v1",
        ("xhs_grid",),
        (rule,),
    )


def _model_capabilities(
    pipeline_graph: Any, runtime_pin: ResourcePin
) -> tuple[ModelCapability, ...]:
    return tuple(
        ModelCapability(
            descriptor.role,
            ResourcePin(
                descriptor.model_id, descriptor.revision, descriptor.expected_sha256
            ),
            "canny" if descriptor.role == "controlnet" else None,
            ("sdxl_turbo", "sdxl_base"),
            ("float16",),
            (runtime_pin.sha256,),
        )
        for descriptor in (
            pipeline_graph.base,
            pipeline_graph.ip_adapter,
            pipeline_graph.controlnet,
        )
    )


def _threshold_profile(
    source: str,
    status: str,
    metric_id: Identifier,
    *,
    encoder_pin: ResourcePin | None,
    plugin_pin: ResourcePin | None,
) -> ThresholdProfileCapability:
    return ThresholdProfileCapability(
        _pin(f"{source}-profile", f"{source}-profile"),
        f"{source}-profile",
        source,
        status,
        Identifier("preset"),
        "product_instance",
        encoder_pin,
        plugin_pin,
        (ThresholdMetricCapability(metric_id, ">=", 0.5),),
        hash_bytes(f"{source}-calibration".encode()),
        hash_bytes(f"{source}-validation".encode()),
        hash_bytes(f"{source}-protocol".encode()),
    )


def _threshold_profiles(
    ip_pin: ResourcePin,
    plugin_pin: ResourcePin,
    l2_status: str,
    l3_status: str,
) -> tuple[ThresholdProfileCapability, ThresholdProfileCapability]:
    return (
        _threshold_profile(
            "l2",
            l2_status,
            _L2_METRIC,
            encoder_pin=ip_pin,
            plugin_pin=None,
        ),
        _threshold_profile(
            "l3",
            l3_status,
            _L3_METRIC,
            encoder_pin=None,
            plugin_pin=plugin_pin,
        ),
    )


def _ip_adapter_pin(pipeline_graph: Any) -> ResourcePin:
    descriptor = pipeline_graph.ip_adapter
    return ResourcePin(
        descriptor.model_id,
        descriptor.revision,
        descriptor.expected_sha256,
    )


def _strength_mapping() -> StrengthMappingCapability:
    return StrengthMappingCapability(
        _pin("mapping", "mapping"),
        Identifier("preset"),
        (
            StrengthMappingEntry(0.0, 0.0, 0.0),
            StrengthMappingEntry(0.7, 0.55, 0.72),
            StrengthMappingEntry(1.0, 1.0, 1.0),
        ),
    )


def _compiler_context(
    pipeline_graph: Any,
    preprocessing_version: str,
    *,
    l2_status: str,
    l3_status: str,
    l3_kind: str,
    l3_requirement: str,
) -> CompilerContext:
    runtime_pin = _pin("runtime", "runtime")
    ip_pin = _ip_adapter_pin(pipeline_graph)
    plugin_pin = _pin("diagnostic-plugin", "diagnostic-plugin")
    runtime = RuntimeCapability(
        runtime_pin, "rocm", "7.2.1", "2.8.0", "0.39.0", "float16"
    )
    encoder = EncoderCapability(
        ip_pin,
        preprocessing_version,
        "hidden_states[-2]",
        "median_cosine_patch_mean_std_v1",
        (runtime_pin.sha256,),
    )
    output = OutputProfileCapability(
        _pin("output", "output"),
        "xhs_grid",
        ("product_instance",),
        ("preview", "production"),
    )
    return CompilerContext(
        _pin("compiler", "compiler"),
        (runtime,),
        _model_capabilities(pipeline_graph, runtime_pin),
        (encoder,),
        (_strength_mapping(),),
        (output,),
        (
            RuleCatalogCapability(
                "1", _pin("rules", "rules"), _l1_rules() + _l2_rules()
            ),
        ),
        _threshold_profiles(ip_pin, plugin_pin, l2_status, l3_status),
        (_l3_plugin(plugin_pin, l3_kind, l3_requirement),),
    )


def _metadata() -> dict[str, object]:
    return {
        "spec_id": "spec",
        "name": "Spec",
        "author": "author",
        "created_at": "2026-07-30T00:00:00Z",
        "parent_spec": None,
    }


def _runtime_spec() -> dict[str, object]:
    return {
        "backend": "rocm",
        "rocm_version": "7.2.1",
        "torch_version": "2.8.0",
        "diffusers_version": "0.39.0",
        "dtype": "float16",
    }


def _model_specs(pipeline_graph: Any) -> dict[str, object]:
    models: dict[str, object] = {}
    for name, descriptor in (
        ("base", pipeline_graph.base),
        ("ip_adapter", pipeline_graph.ip_adapter),
        ("controlnet", pipeline_graph.controlnet),
    ):
        value = {
            "id": descriptor.model_id,
            "revision": descriptor.revision,
            "sha256": descriptor.expected_sha256.value,
        }
        if name == "controlnet":
            value["type"] = "canny"
        models[name] = value
    return models


def _style_assets(style_contents: tuple[bytes, ...]) -> dict[str, object]:
    return {
        "style_references": tuple(
            {
                "asset_sha256": hash_bytes(content).value,
                "source_url": f"https://example.com/style-{index}",
                "license": "CC0",
                "attribution": "author",
                "consent": "not_applicable",
            }
            for index, content in enumerate(style_contents)
        )
    }


def _profile_specs() -> dict[str, object]:
    return {
        "preview": {
            "pipeline": "sdxl_turbo",
            "resolution": (64, 64),
            "steps": 4,
            "guidance_scale": 0.0,
        },
        "production": {
            "pipeline": "sdxl_base",
            "resolution": (64, 64),
            "steps": 30,
            "guidance_scale": 5.0,
            "scheduler": "euler",
        },
    }


def _style_contract_sections(fidelity_required: bool) -> dict[str, object]:
    return {
        "style": {
            "preset_id": "preset",
            "user_strength": 0.7,
            "preview_ip_adapter_scale": 0.55,
            "production_ip_adapter_scale": 0.72,
        },
        "generation": {
            "img2img_strength": 0.45,
            "controlnet_scale": 0.7,
            "seed_policy": "per_asset_deterministic",
            "batch_execution": "sequential",
        },
        "domain": {
            "profile": "product_instance",
            "verifier_version": "v1",
            "fidelity_required": fidelity_required,
        },
        "outputs": {"profiles": ("xhs_grid",)},
    }


def _verification_spec(
    pipeline_graph: Any, context: CompilerContext
) -> dict[str, object]:
    l2_profile = context.threshold_profiles[0]
    plugin = context.l3_plugins[0]
    return {
        "ruleset_version": "1",
        "gate_defaults": {
            "on_unverifiable": "reject",
            "on_warning": "manual_review",
        },
        "l2": {
            "encoder_id": pipeline_graph.ip_adapter.model_id,
            "encoder_revision": pipeline_graph.ip_adapter.revision,
            "preprocessing_version": context.encoder_capabilities[
                0
            ].preprocessing_version,
            "threshold_profile": {
                "id": l2_profile.pin.id,
                "revision": l2_profile.pin.revision,
                "sha256": l2_profile.pin.sha256.value,
            },
        },
        "l3": {
            "plugin_id": plugin.pin.id,
            "plugin_revision": plugin.pin.revision,
            "threshold_profile": context.threshold_profiles[1].logical_name,
        },
    }


def _replay_contract() -> dict[str, object]:
    return {
        "mode": "semantic",
        "tolerated_metric_delta": {
            "l2_style_fidelity": 0.0,
            "l3_fidelity": 0.0,
        },
        "new_batch": {
            "contract": "same_compiled_graph_and_gate_definitions",
            "per_item_metric_equality_required": False,
        },
    }


def _raw_spec(
    pipeline_graph: Any,
    context: CompilerContext,
    style_contents: tuple[bytes, ...],
    *,
    fidelity_required: bool,
) -> StyleSpecV1:
    data = {
        "schema_version": "1.0",
        "schema_uri": "schemas/style-spec-1.0.schema.json",
        "metadata": _metadata(),
        "runtime": _runtime_spec(),
        "models": _model_specs(pipeline_graph),
        "assets": _style_assets(style_contents),
        "profiles": _profile_specs(),
        **_style_contract_sections(fidelity_required),
        "verification": _verification_spec(pipeline_graph, context),
        "repair": {
            "policy_version": "1",
            "max_rounds": 1,
            "stop_after_no_improvement": 1,
        },
        "replay_contract": _replay_contract(),
    }
    return StyleSpecV1.model_validate(data)
