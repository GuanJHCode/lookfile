"""specstyle.domain.enums 领域枚举单测。

覆盖 BOOT-002A 冻结合同：
- 七个枚举均继承 enum.StrEnum，member.name == member.value；
- 无 alias（__members__ 与迭代等长）；
- 无自定义 _missing_、无别名、无宽松转换；
- 精确成员顺序与精确值（经 __members__ 读取，出现 alias 时必须失败）；
- member 为 str 实例，str(member) == member.value；
- JSON 字符串 round-trip（json.dumps(member) == json.dumps(value)；
  json.loads 后 EnumClass(value) 还原原 member）；
- 未知值 / 错误大小写 / 空字符串均抛 ValueError；
- RuleStatus 不含 NOT_APPLICABLE，NOT_APPLICABLE 只属于 StaticApplicability；
- specstyle.domain 不 re-export 任一枚举、不定义 __all__。
"""
import enum
import json

import pytest

from specstyle.domain.enums import (
    ArtifactStatus,
    DecisionReason,
    RepairStopReason,
    RuleLevel,
    RuleScope,
    RuleStatus,
    StaticApplicability,
)

ALL_ENUMS = [
    RuleStatus,
    StaticApplicability,
    ArtifactStatus,
    DecisionReason,
    RepairStopReason,
    RuleLevel,
    RuleScope,
]


def _members(cls):
    """[(name, value), ...] 按 __members__ 顺序，含 alias（出现 alias 时精确成员测试必须失败）。"""
    return [(name, member.value) for name, member in cls.__members__.items()]


# --- 精确成员顺序与精确值 ---

def test_rulestatus_members():
    assert _members(RuleStatus) == [
        ("PASS", "PASS"),
        ("FAIL", "FAIL"),
        ("WARNING", "WARNING"),
        ("UNVERIFIABLE", "UNVERIFIABLE"),
    ]


def test_staticapplicability_members():
    assert _members(StaticApplicability) == [
        ("APPLICABLE", "APPLICABLE"),
        ("NOT_APPLICABLE", "NOT_APPLICABLE"),
    ]


def test_artifactstatus_members():
    assert _members(ArtifactStatus) == [
        ("APPROVED", "APPROVED"),
        ("REJECTED", "REJECTED"),
        ("MANUAL_REVIEW", "MANUAL_REVIEW"),
    ]


def test_decisionreason_members():
    assert _members(DecisionReason) == [
        ("ALL_REQUIRED_PASS", "ALL_REQUIRED_PASS"),
        ("REQUIRED_GATE_FAILED", "REQUIRED_GATE_FAILED"),
        ("REQUIRED_GATE_UNVERIFIABLE", "REQUIRED_GATE_UNVERIFIABLE"),
        ("REPAIR_EXHAUSTED", "REPAIR_EXHAUSTED"),
        ("MANUAL_POLICY", "MANUAL_POLICY"),
    ]


def test_repairstopreason_members():
    assert _members(RepairStopReason) == [
        ("PASS_ALL_REQUIRED", "PASS_ALL_REQUIRED"),
        ("NO_ACTION", "NO_ACTION"),
        ("NO_IMPROVEMENT", "NO_IMPROVEMENT"),
        ("MAX_ROUNDS", "MAX_ROUNDS"),
        ("UNVERIFIABLE", "UNVERIFIABLE"),
        ("MANUAL_REQUEST", "MANUAL_REQUEST"),
    ]


def test_rulelevel_members():
    assert _members(RuleLevel) == [
        ("L1", "L1"),
        ("L2", "L2"),
        ("L3", "L3"),
    ]


def test_rulescope_members():
    assert _members(RuleScope) == [
        ("ITEM", "ITEM"),
        ("BATCH", "BATCH"),
    ]


# --- 不变量 ---

@pytest.mark.parametrize("cls", ALL_ENUMS, ids=lambda c: c.__name__)
def test_name_equals_value(cls):
    for m in cls:
        assert m.name == m.value


@pytest.mark.parametrize("cls", ALL_ENUMS, ids=lambda c: c.__name__)
def test_members_are_str(cls):
    for m in cls:
        assert isinstance(m, str)


@pytest.mark.parametrize("cls", ALL_ENUMS, ids=lambda c: c.__name__)
def test_json_roundtrip(cls):
    for m in cls:
        dumped = json.dumps(m)
        assert dumped == json.dumps(m.value)  # member 序列化与纯值字符串一致
        loaded = json.loads(dumped)
        assert loaded == m.value
        assert cls(loaded) is m  # EnumClass(value) 还原原 member


@pytest.mark.parametrize("cls", ALL_ENUMS, ids=lambda c: c.__name__)
def test_invalid_values_raise_valueerror(cls):
    first = next(iter(cls))
    bad_values = (
        "DEFINITELY_NOT_A_MEMBER_XYZ",  # 未知值
        first.value.lower(),  # 错误大小写
        "",  # 空字符串
    )
    for bad in bad_values:
        with pytest.raises(ValueError):
            cls(bad)


@pytest.mark.parametrize("cls", ALL_ENUMS, ids=lambda c: c.__name__)
def test_no_custom_missing(cls):
    # 仅验证未自定义 _missing_；alias 由 test_enum_has_no_aliases 单独把关
    assert "_missing_" not in cls.__dict__


@pytest.mark.parametrize("cls", ALL_ENUMS, ids=lambda c: c.__name__)
def test_enum_has_no_aliases(cls):
    # __members__ 含 alias；与迭代（仅 canonical）等长则无 alias
    assert len(cls.__members__) == len(cls)


@pytest.mark.parametrize("cls", ALL_ENUMS, ids=lambda c: c.__name__)
def test_enum_uses_strenum(cls):
    assert issubclass(cls, enum.StrEnum)
    for member in cls:
        assert str(member) == member.value


# --- RuleStatus / StaticApplicability 分离 ---

def test_rulestatus_excludes_not_applicable():
    names = {m.name for m in RuleStatus}
    values = {m.value for m in RuleStatus}
    assert "NOT_APPLICABLE" not in names
    assert "NOT_APPLICABLE" not in values
    assert not hasattr(RuleStatus, "NOT_APPLICABLE")


def test_not_applicable_only_in_staticapplicability():
    assert StaticApplicability.NOT_APPLICABLE.value == "NOT_APPLICABLE"
    for cls in ALL_ENUMS:
        if cls is StaticApplicability:
            assert hasattr(cls, "NOT_APPLICABLE")
        else:
            assert not hasattr(cls, "NOT_APPLICABLE"), (
                f"{cls.__name__} must not define NOT_APPLICABLE"
            )


# --- specstyle.domain 不 re-export ---

def test_domain_package_does_not_reexport_enums():
    import specstyle.domain as dom

    for name in (
        "RuleStatus",
        "StaticApplicability",
        "ArtifactStatus",
        "DecisionReason",
        "RepairStopReason",
        "RuleLevel",
        "RuleScope",
    ):
        assert not hasattr(dom, name), f"specstyle.domain must not re-export {name}"


def test_domain_package_has_no_all():
    import specstyle.domain as dom

    assert not hasattr(dom, "__all__")
