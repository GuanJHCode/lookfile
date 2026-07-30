import hashlib
import json
from dataclasses import fields, replace
from io import BytesIO

import pytest
from PIL import Image

from specstyle.domain.artifacts import AssetRef
from specstyle.domain.identifiers import AssetId, AttemptId, Identifier, JobId, Sha256
from specstyle.errors import DomainError
from specstyle.generation.preprocess import (
    PreparedImage,
    PreprocessPlan,
    preprocess_image,
)
from specstyle.generation.requests import (
    GenerationRequest,
    PreparedControlInput,
    RenderedPrompt,
)
from specstyle.observability.hashing import hash_bytes
from specstyle.spec.compiler import compile_style_spec
from specstyle.spec.compiled_models import ResourcePin
from tests.unit.spec.test_compiler import context, raw_spec


def _request(**changes: object) -> GenerationRequest:
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "red").save(buffer, "PNG")
    encoded = buffer.getvalue()
    source_ref = AssetRef(AssetId("source"), hash_bytes(encoded))
    pin = ResourcePin("processor", "r1", Sha256("f" * 64))
    prepared = preprocess_image(
        encoded, source_ref, PreprocessPlan((512, 512), "contain_pad", (0, 0, 0), pin)
    )
    compiled = changes.get("compiled_spec", compile_style_spec(raw_spec(), context()))
    assert type(compiled).__name__ == "CompiledStyleSpec"
    graph = compiled.preview_graphs[0]
    prompt = RenderedPrompt(
        ResourcePin("template", "r1", Sha256("e" * 64)), graph.preset_id, "a prompt", ""
    )
    control = PreparedControlInput("canny", prepared)
    values = dict(
        job_id=JobId("job"),
        attempt_id=AttemptId("attempt"),
        parent_attempt_id=None,
        compiled_spec=compiled,
        generation_profile="preview",
        output_profile="xhs_grid",
        source=prepared,
        style_references=(AssetRef(AssetId("style"), graph.style_reference_hashes[0]),),
        prompt=prompt,
        control_input=control,
        variation_index=0,
        environment_hash=Sha256("d" * 64),
    )
    values.update(changes)
    return GenerationRequest(**values)  # type: ignore[arg-type]


def test_request_resolves_graph_and_excludes_attempt_and_environment_from_fingerprint() -> (
    None
):
    base = _request()
    changed = _request(
        attempt_id=AttemptId("attempt2"), environment_hash=Sha256("c" * 64)
    )
    assert base.graph == base.compiled_spec.preview_graphs[0]
    assert base.seed == changed.seed
    assert base.generation_fingerprint == changed.generation_fingerprint
    assert base.request_hash != changed.request_hash


def test_request_rejects_style_order_and_cross_field_mismatches() -> None:
    request = _request()
    with pytest.raises(DomainError):
        replace(request, style_references=())
    with pytest.raises(DomainError):
        replace(request, prompt=replace(request.prompt, preset_id=Identifier("other")))
    with pytest.raises(DomainError):
        replace(request, control_input=replace(request.control_input, kind="depth"))


def test_request_rejects_forged_compiled_hash() -> None:
    request = _request()
    object.__setattr__(request.compiled_spec, "compiled_spec_hash", Sha256("0" * 64))
    with pytest.raises(DomainError):
        replace(request)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("template_pin", "forged-pin"),
        ("preset_id", "forged-preset"),
        ("positive", " forged positive"),
        ("negative", "forged\nnegative"),
    ),
)
def test_request_rejects_pre_forged_prompt_material_before_hashing(
    field: str, value: object
) -> None:
    request = _request()
    object.__setattr__(request.prompt, field, value)

    with pytest.raises(DomainError):
        _request(prompt=request.prompt)


def test_rendered_prompt_has_only_contract_fields() -> None:
    assert [field.name for field in fields(RenderedPrompt)] == [
        "template_pin",
        "preset_id",
        "positive",
        "negative",
    ]


def test_request_graph_is_read_only_property_and_rejects_forged_prepared_hash() -> None:
    request = _request()
    assert "graph" not in {item.name for item in fields(type(request))}
    object.__setattr__(request.source, "sha256", "forged")
    rebuilt = replace(request)
    assert rebuilt.source.sha256 != "forged"
    assert rebuilt.generation_fingerprint == request.generation_fingerprint


def test_request_rejects_string_subclass_selector() -> None:
    class Selector(str):
        pass

    with pytest.raises(DomainError):
        _request(generation_profile=Selector("preview"))


def _prepared(
    color: str = "red", *, source_id: str = "source", plan: PreprocessPlan | None = None
) -> PreparedImage:
    output = BytesIO()
    Image.new("RGB", (64, 64), color).save(output, "PNG")
    content = output.getvalue()
    effective_plan = plan or PreprocessPlan(
        (512, 512),
        "contain_pad",
        (0, 0, 0),
        ResourcePin("processor", "r1", Sha256("f" * 64)),
    )
    return preprocess_image(
        content, AssetRef(AssetId(source_id), hash_bytes(content)), effective_plan
    )


def test_request_rejects_zero_and_multiple_graph_selector_matches() -> None:
    request = _request()
    graph = request.graph
    none = replace(
        request.compiled_spec,
        preview_graphs=(replace(graph, output_profile="talking_head_cover"),),
    )
    many = replace(request.compiled_spec, preview_graphs=(graph, graph))

    with pytest.raises(DomainError, match="exactly one"):
        _request(compiled_spec=none)
    with pytest.raises(DomainError, match="exactly one"):
        _request(compiled_spec=many)


def test_generation_hashes_match_fixed_independent_golden_values() -> None:
    request = _request()

    assert request.seed.seed == 6964608022339675490
    assert request.generation_fingerprint.value == (
        "adc89df1cc70bc5ec08ac934c9bb7550d9104234981efd73f3aa71413b10100d"
    )
    assert request.request_hash.value == (
        "b54ba7403046f7129e851db637d00e0c3eed6c9ccc7c73541c8e3b37824534e3"
    )
    materials_payload = {
        "domain": "specstyle.generation.materials.v1",
        "compiled_spec_hash": "c21f061f246a258eb1ffc9043343f4bcc886ea775fdc2302c7d2cfe03538141a",
        "generation_profile": "preview",
        "output_profile": "xhs_grid",
        "source": {
            "source": {
                "asset_id": "source",
                "sha256": "6e35e4c2d632339a28e9d33792e5135892dd7292f450041988f7b30c5aced00f",
            },
            "sha256": "25ffbe7019e77de42d4a2eae889b6758a01d2a93a0a45add244752f3e1fb0953",
            "snapshot": {
                "plan": {
                    "target_size": [512, 512],
                    "resize_mode": "contain_pad",
                    "background": [0, 0, 0],
                    "processor_pin": {
                        "id": "processor",
                        "revision": "r1",
                        "sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                    },
                },
                "input_format": "PNG",
                "input_mode": "RGB",
                "input_size": [64, 64],
                "exif_orientation": 1,
                "pillow_version": "12.1.1",
            },
        },
        "style_references": [
            {
                "asset_id": "style",
                "sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            },
        ],
        "prompt": {
            "template_pin": {
                "id": "template",
                "revision": "r1",
                "sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            },
            "preset_id": "preset",
            "positive": "a prompt",
            "negative": "",
        },
        "control_kind": "canny",
        "control_image": {
            "source": {
                "asset_id": "source",
                "sha256": "6e35e4c2d632339a28e9d33792e5135892dd7292f450041988f7b30c5aced00f",
            },
            "sha256": "25ffbe7019e77de42d4a2eae889b6758a01d2a93a0a45add244752f3e1fb0953",
            "snapshot": {
                "plan": {
                    "target_size": [512, 512],
                    "resize_mode": "contain_pad",
                    "background": [0, 0, 0],
                    "processor_pin": {
                        "id": "processor",
                        "revision": "r1",
                        "sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                    },
                },
                "input_format": "PNG",
                "input_mode": "RGB",
                "input_size": [64, 64],
                "exif_orientation": 1,
                "pillow_version": "12.1.1",
            },
        },
        "variation_index": 0,
        "seed": 6964608022339675490,
    }
    assert materials_payload["source"]["snapshot"]["plan"]["target_size"] == [
        512,
        512,
    ]
    assert materials_payload["control_image"]["snapshot"]["plan"]["target_size"] == [
        512,
        512,
    ]
    materials_encoded = json.dumps(
        materials_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert (
        hashlib.sha256(materials_encoded).hexdigest()
        == request.generation_fingerprint.value
    )
    request_payload = {
        "domain": "specstyle.generation.request.v1",
        "generation_fingerprint": "adc89df1cc70bc5ec08ac934c9bb7550d9104234981efd73f3aa71413b10100d",
        "job_id": "job",
        "attempt_id": "attempt",
        "parent_attempt_id": None,
        "environment_hash": "d" * 64,
    }
    encoded = json.dumps(
        request_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == request.request_hash.value


def test_fingerprint_excludes_job_attempt_parent_and_environment() -> None:
    base = _request()
    changed = _request(
        job_id=JobId("job2"),
        attempt_id=AttemptId("attempt2"),
        parent_attempt_id=AttemptId("parent"),
        environment_hash=Sha256("c" * 64),
    )

    assert changed.generation_fingerprint == base.generation_fingerprint
    assert changed.request_hash != base.request_hash


def test_fingerprint_changes_for_compiled_hash_and_selector_materials() -> None:
    base = _request()
    changed_spec = raw_spec().model_copy(
        update={"metadata": raw_spec().metadata.model_copy(update={"name": "Other"})}
    )
    compiled = compile_style_spec(changed_spec, context())
    different_compiled = _request(compiled_spec=compiled)
    production_plan = PreprocessPlan(
        (1024, 1024),
        "contain_pad",
        (0, 0, 0),
        ResourcePin("processor", "r1", Sha256("f" * 64)),
    )
    production_source = _prepared(plan=production_plan)

    assert different_compiled.generation_fingerprint != base.generation_fingerprint
    assert (
        _request(
            generation_profile="production",
            output_profile="xhs_grid",
            source=production_source,
            control_input=PreparedControlInput("canny", production_source),
        ).generation_fingerprint
        != base.generation_fingerprint
    )

    selected_graph = replace(base.graph, output_profile="talking_head_cover")
    selected_compiled = replace(base.compiled_spec, preview_graphs=(selected_graph,))
    assert (
        _request(
            compiled_spec=selected_compiled, output_profile="talking_head_cover"
        ).generation_fingerprint
        != base.generation_fingerprint
    )


def test_fingerprint_changes_for_source_reference_and_prepared_hash() -> None:
    base = _request()
    same_content_different_ref = _prepared(source_id="source2")
    alternate_content = _prepared("blue")
    same_source_and_snapshot = PreparedImage(
        base.source.source, alternate_content.content, base.source.snapshot
    )

    assert (
        _request(
            source=same_content_different_ref,
            control_input=PreparedControlInput("canny", same_content_different_ref),
        ).generation_fingerprint
        != base.generation_fingerprint
    )
    assert (
        _request(
            source=same_source_and_snapshot,
            control_input=PreparedControlInput("canny", base.source),
        ).generation_fingerprint
        != base.generation_fingerprint
    )


@pytest.mark.parametrize(
    "snapshot",
    (
        lambda value: replace(value, input_format="JPEG"),
        lambda value: replace(value, input_mode="L"),
        lambda value: replace(value, input_size=(63, 64)),
        lambda value: replace(value, exif_orientation=2),
        lambda value: replace(value, pillow_version="fixture-version"),
    ),
)
@pytest.mark.parametrize("target", ("source", "control"))
def test_fingerprint_changes_source_and_control_snapshots_independently(
    snapshot: object, target: str
) -> None:
    base = _request()
    altered = PreparedImage(
        base.source.source,
        base.source.content,
        snapshot(base.source.snapshot),
    )
    changed = (
        _request(
            source=altered, control_input=PreparedControlInput("canny", base.source)
        )
        if target == "source"
        else _request(control_input=PreparedControlInput("canny", altered))
    )

    assert changed.generation_fingerprint != base.generation_fingerprint


@pytest.mark.parametrize(
    "pin",
    (
        ResourcePin("processor2", "r1", Sha256("f" * 64)),
        ResourcePin("processor", "r2", Sha256("f" * 64)),
        ResourcePin("processor", "r1", Sha256("9" * 64)),
    ),
)
@pytest.mark.parametrize("target", ("source", "control"))
def test_fingerprint_changes_processor_pin_fields_independently(
    pin: ResourcePin, target: str
) -> None:
    base = _request()
    snapshot = replace(
        base.source.snapshot,
        plan=replace(base.source.snapshot.plan, processor_pin=pin),
    )
    altered = PreparedImage(base.source.source, base.source.content, snapshot)
    changed = (
        _request(
            source=altered, control_input=PreparedControlInput("canny", base.source)
        )
        if target == "source"
        else _request(control_input=PreparedControlInput("canny", altered))
    )

    assert changed.generation_fingerprint != base.generation_fingerprint


@pytest.mark.parametrize(
    "snapshot",
    (
        lambda value: replace(value, input_format="JPEG"),
        lambda value: replace(value, input_mode="L"),
        lambda value: replace(value, input_size=(63, 64)),
        lambda value: replace(value, exif_orientation=2),
        lambda value: replace(value, pillow_version="fixture-version"),
        lambda value: replace(
            value,
            plan=replace(
                value.plan,
                processor_pin=ResourcePin("processor2", "r1", Sha256("9" * 64)),
            ),
        ),
        lambda value: replace(
            value, plan=replace(value.plan, resize_mode="cover_center")
        ),
        lambda value: replace(value, plan=replace(value.plan, background=(1, 2, 3))),
    ),
)
def test_fingerprint_changes_for_each_prepared_snapshot_material(
    snapshot: object,
) -> None:
    base = _request()
    changed_snapshot = snapshot(base.source.snapshot)
    changed = PreparedImage(base.source.source, base.source.content, changed_snapshot)
    request = _request(
        source=changed,
        control_input=PreparedControlInput("canny", changed),
    )

    assert request.generation_fingerprint != base.generation_fingerprint


def test_fingerprint_changes_for_style_order_and_template_pin_fields() -> None:
    base = _request()
    graph = base.graph
    second_hash = Sha256("9" * 64)
    changed_graph = replace(
        graph, style_reference_hashes=(second_hash, graph.style_reference_hashes[0])
    )
    compiled = replace(base.compiled_spec, preview_graphs=(changed_graph,))
    ordered = _request(
        compiled_spec=compiled,
        style_references=(
            AssetRef(AssetId("style2"), second_hash),
            AssetRef(AssetId("style"), graph.style_reference_hashes[0]),
        ),
    )

    assert ordered.generation_fingerprint != base.generation_fingerprint
    changed_ref = _request(
        style_references=(AssetRef(AssetId("style2"), graph.style_reference_hashes[0]),)
    )
    assert changed_ref.generation_fingerprint != base.generation_fingerprint
    for pin in (
        ResourcePin("template2", "r1", Sha256("e" * 64)),
        ResourcePin("template", "r2", Sha256("e" * 64)),
        ResourcePin("template", "r1", Sha256("9" * 64)),
    ):
        changed = _request(prompt=replace(base.prompt, template_pin=pin))
        assert changed.generation_fingerprint != base.generation_fingerprint


def test_fingerprint_changes_when_same_hash_style_references_are_reordered() -> None:
    base = _request()
    hash_value = base.graph.style_reference_hashes[0]
    graph = replace(base.graph, style_reference_hashes=(hash_value, hash_value))
    compiled = replace(base.compiled_spec, preview_graphs=(graph,))
    forward = _request(
        compiled_spec=compiled,
        style_references=(
            AssetRef(AssetId("style-a"), hash_value),
            AssetRef(AssetId("style-b"), hash_value),
        ),
    )
    reverse = _request(
        compiled_spec=compiled,
        style_references=(
            AssetRef(AssetId("style-b"), hash_value),
            AssetRef(AssetId("style-a"), hash_value),
        ),
    )

    assert (
        forward.compiled_spec.compiled_spec_hash
        == reverse.compiled_spec.compiled_spec_hash
    )
    assert forward.generation_fingerprint != reverse.generation_fingerprint


def test_request_rejects_target_size_that_does_not_match_resolved_graph() -> None:
    wrong_size = _prepared(
        plan=PreprocessPlan(
            (64, 64),
            "contain_pad",
            (0, 0, 0),
            ResourcePin("processor", "r1", Sha256("f" * 64)),
        )
    )

    with pytest.raises(DomainError, match="dimensions"):
        _request(
            source=wrong_size,
            control_input=PreparedControlInput("canny", wrong_size),
        )


def test_fingerprint_changes_for_prompt_control_variation_and_seed() -> None:
    base = _request()
    alternative = _prepared("blue")
    control_only = PreparedImage(
        base.source.source, alternative.content, base.source.snapshot
    )
    depth_graph = replace(
        base.graph,
        controlnet=replace(base.graph.controlnet, controlnet_type="depth"),
    )
    depth_compiled = replace(base.compiled_spec, preview_graphs=(depth_graph,))
    preset_graph = replace(base.graph, preset_id=Identifier("preset2"))
    preset_compiled = replace(base.compiled_spec, preview_graphs=(preset_graph,))
    changed = (
        _request(prompt=replace(base.prompt, positive="different prompt")),
        _request(prompt=replace(base.prompt, negative="negative")),
        _request(control_input=PreparedControlInput("canny", control_only)),
        _request(
            compiled_spec=depth_compiled,
            control_input=PreparedControlInput("depth", base.source),
        ),
        _request(
            compiled_spec=preset_compiled,
            prompt=replace(base.prompt, preset_id=Identifier("preset2")),
        ),
        _request(variation_index=1),
    )

    assert all(
        item.generation_fingerprint != base.generation_fingerprint for item in changed
    )
    assert changed[-1].seed != base.seed


@pytest.mark.parametrize(
    "changes",
    (
        {"job_id": JobId("job2")},
        {"attempt_id": AttemptId("attempt2")},
        {"parent_attempt_id": AttemptId("parent")},
        {"environment_hash": Sha256("c" * 64)},
    ),
)
def test_request_hash_changes_for_each_request_only_material(
    changes: dict[str, object],
) -> None:
    base = _request()
    changed = _request(**changes)

    assert changed.generation_fingerprint == base.generation_fingerprint
    assert changed.request_hash != base.request_hash


@pytest.mark.parametrize(
    "value",
    (" positive", "positive ", "", "bad\ntext", "x" * 2049),
)
def test_rendered_prompt_rejects_non_safe_positive_text(value: str) -> None:
    with pytest.raises(DomainError):
        RenderedPrompt(
            ResourcePin("template", "r1", Sha256("e" * 64)),
            Identifier("preset"),
            value,
            "",
        )


def test_generation_value_objects_are_frozen_slotted_and_graph_is_not_a_field() -> None:
    request = _request()
    for value in (request.prompt, request.control_input, request):
        assert not hasattr(value, "__dict__")
        with pytest.raises((AttributeError, TypeError)):
            setattr(value, "unexpected", "value")
    assert "graph" not in {item.name for item in fields(GenerationRequest)}
