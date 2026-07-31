"""specstyle.domain.artifacts 单测：AssetRef、ArtifactRef。

覆盖 Module 1（Domain Foundation）冻结合同：
- immutable AssetRef(asset_id, sha256) 与 ArtifactRef(artifact_id, sha256)；
- 严格类型（sibling ID 错配、非 Sha256 抛 DomainError）；
- 精确 mapping round-trip（to_primitive/from_primitive，额外/缺失 key 报错）；
- frozen+slots、hashable、无 __dict__、不含 path、无 IO/解码/hash 计算。

重点测试：Sibling ID 错配、额外/缺失 mapping key。
"""

import dataclasses
from collections import UserDict
from types import MappingProxyType

import pytest

from specstyle.errors import DomainError
from specstyle.domain.identifiers import AssetId, ArtifactId, Sha256
from specstyle.domain.artifacts import AssetRef, ArtifactRef

HEX = "a" * 64
HEX_UPPER = "A" * 64


def _asset_ref(value="a1"):
    return AssetRef(AssetId(value), Sha256(HEX))


def _artifact_ref(value="r1"):
    return ArtifactRef(ArtifactId(value), Sha256(HEX))


# --- 构造与字段 ---


def test_asset_ref_construction():
    ref = _asset_ref("a1")
    assert ref.asset_id == AssetId("a1")
    assert ref.sha256 == Sha256(HEX)


def test_artifact_ref_construction():
    ref = _artifact_ref("r1")
    assert ref.artifact_id == ArtifactId("r1")
    assert ref.sha256 == Sha256(HEX)


def test_refs_have_no_path_field():
    assert not hasattr(_asset_ref(), "path")
    assert not hasattr(_artifact_ref(), "path")


# --- 严格类型：sibling ID 错配与非 Sha256 ---


def test_asset_ref_rejects_plain_string_asset_id():
    with pytest.raises(DomainError):
        AssetRef("a1", Sha256(HEX))  # type: ignore[arg-type]


def test_asset_ref_rejects_sibling_artifact_id():
    with pytest.raises(DomainError):
        AssetRef(ArtifactId("a1"), Sha256(HEX))


def test_asset_ref_rejects_plain_string_sha256():
    with pytest.raises(DomainError):
        AssetRef(AssetId("a1"), HEX)  # type: ignore[arg-type]


def test_asset_ref_rejects_invalid_sha256_value():
    with pytest.raises(DomainError):
        AssetRef(AssetId("a1"), Sha256("g" * 64))


def test_artifact_ref_rejects_sibling_asset_id():
    with pytest.raises(DomainError):
        ArtifactRef(AssetId("r1"), Sha256(HEX))


def test_artifact_ref_rejects_plain_strings():
    with pytest.raises(DomainError):
        ArtifactRef("r1", Sha256(HEX))  # type: ignore[arg-type]
    with pytest.raises(DomainError):
        ArtifactRef(ArtifactId("r1"), HEX)  # type: ignore[arg-type]


# --- to_primitive / from_primitive 精确 round-trip ---


def test_asset_ref_to_primitive_shape():
    assert _asset_ref("a1").to_primitive() == {"asset_id": "a1", "sha256": HEX}


def test_artifact_ref_to_primitive_shape():
    assert _artifact_ref("r1").to_primitive() == {"artifact_id": "r1", "sha256": HEX}


def test_asset_ref_round_trip():
    ref = _asset_ref("a1")
    restored = AssetRef.from_primitive(ref.to_primitive())
    assert restored == ref
    assert type(restored) is AssetRef


def test_artifact_ref_round_trip():
    ref = _artifact_ref("r1")
    restored = ArtifactRef.from_primitive(ref.to_primitive())
    assert restored == ref
    assert type(restored) is ArtifactRef


def test_asset_ref_from_primitive_rejects_extra_key():
    with pytest.raises(DomainError):
        AssetRef.from_primitive({"asset_id": "a1", "sha256": HEX, "extra": 1})


def test_asset_ref_from_primitive_rejects_missing_key():
    with pytest.raises(DomainError):
        AssetRef.from_primitive({"asset_id": "a1"})
    with pytest.raises(DomainError):
        AssetRef.from_primitive({"sha256": HEX})


def test_asset_ref_from_primitive_rejects_non_mapping():
    for bad in (None, "not-a-mapping", 123, []):
        with pytest.raises(DomainError):
            AssetRef.from_primitive(bad)


def test_asset_ref_from_primitive_rejects_bad_inner_values():
    with pytest.raises(DomainError):
        AssetRef.from_primitive({"asset_id": "-bad", "sha256": HEX})  # 非法 ID
    with pytest.raises(DomainError):
        AssetRef.from_primitive({"asset_id": "a1", "sha256": "g" * 64})  # 非法 sha


def test_asset_ref_from_primitive_rejects_wrong_key_name():
    # key 名不匹配（artifact_id 而非 asset_id）
    with pytest.raises(DomainError):
        AssetRef.from_primitive({"artifact_id": "a1", "sha256": HEX})


# --- 非 dict Mapping 正例（MappingProxyType / UserDict）---


def test_asset_ref_accepts_mapping_proxy_type():
    data = MappingProxyType({"asset_id": "a1", "sha256": HEX_UPPER})
    ref = AssetRef.from_primitive(data)
    assert ref.asset_id == AssetId("a1")
    assert ref.sha256.value == HEX  # 大写 SHA 规范为小写
    assert ref == AssetRef(AssetId("a1"), Sha256(HEX_UPPER))
    restored = AssetRef.from_primitive(ref.to_primitive())
    assert restored == ref
    assert type(restored) is AssetRef


def test_asset_ref_accepts_user_dict():
    data = UserDict({"asset_id": "a1", "sha256": HEX_UPPER})
    ref = AssetRef.from_primitive(data)
    assert ref.asset_id == AssetId("a1")
    assert ref.sha256.value == HEX
    assert AssetRef.from_primitive(ref.to_primitive()) == ref


def test_artifact_ref_accepts_mapping_proxy_type():
    data = MappingProxyType({"artifact_id": "r1", "sha256": HEX_UPPER})
    ref = ArtifactRef.from_primitive(data)
    assert ref.artifact_id == ArtifactId("r1")
    assert ref.sha256.value == HEX
    assert ref == ArtifactRef(ArtifactId("r1"), Sha256(HEX_UPPER))
    assert ArtifactRef.from_primitive(ref.to_primitive()) == ref


def test_artifact_ref_accepts_user_dict():
    data = UserDict({"artifact_id": "r1", "sha256": HEX_UPPER})
    ref = ArtifactRef.from_primitive(data)
    assert ref.artifact_id == ArtifactId("r1")
    assert ref.sha256.value == HEX
    assert ArtifactRef.from_primitive(ref.to_primitive()) == ref


# --- ArtifactRef from_primitive 负例（与 AssetRef 对称）---


@pytest.mark.parametrize(
    "bad", [None, "not-a-mapping", 123, []], ids=["none", "str", "int", "list"]
)
def test_artifact_ref_from_primitive_rejects_non_mapping(bad):
    with pytest.raises(DomainError):
        ArtifactRef.from_primitive(bad)


def test_artifact_ref_from_primitive_rejects_missing_key():
    with pytest.raises(DomainError):
        ArtifactRef.from_primitive({"artifact_id": "r1"})
    with pytest.raises(DomainError):
        ArtifactRef.from_primitive({"sha256": HEX})


def test_artifact_ref_from_primitive_rejects_extra_key():
    with pytest.raises(DomainError):
        ArtifactRef.from_primitive({"artifact_id": "r1", "sha256": HEX, "extra": 1})


def test_artifact_ref_from_primitive_rejects_wrong_key_name():
    # key 名不匹配（asset_id 而非 artifact_id）
    with pytest.raises(DomainError):
        ArtifactRef.from_primitive({"asset_id": "r1", "sha256": HEX})


# --- frozen / slots / hashable / 类型隔离 ---


def test_refs_frozen():
    ref = _asset_ref()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.asset_id = AssetId("z9")  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.sha256 = Sha256("b" * 64)


def test_refs_no_dict():
    assert not hasattr(_asset_ref(), "__dict__")
    assert not hasattr(_artifact_ref(), "__dict__")


def test_refs_hashable():
    a = _asset_ref("a1")
    b = _asset_ref("a1")
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_asset_ref_not_equal_artifact_ref():
    a = AssetRef(AssetId("x"), Sha256(HEX))
    b = ArtifactRef(ArtifactId("x"), Sha256(HEX))
    assert a != b
