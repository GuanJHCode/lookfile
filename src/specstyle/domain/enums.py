"""specstyle 领域枚举。

使用 enum.StrEnum：member.name == member.value，值为 str，
JSON round-trip 可靠，未知值抛 ValueError（无 _missing_、无别名、无宽松转换）。
所有 value 显式书写，不使用 auto()。
"""

import enum


class RuleStatus(enum.StrEnum):
    """规则执行结果状态。绝对不含 NOT_APPLICABLE。"""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    UNVERIFIABLE = "UNVERIFIABLE"


class StaticApplicability(enum.StrEnum):
    """规则适用性的静态判定（仅 Compiler 产生）。NOT_APPLICABLE 只属于此处。"""

    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ArtifactStatus(enum.StrEnum):
    """产出终态。APPROVED 是白名单终态。"""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class DecisionReason(enum.StrEnum):
    """终态决策原因。"""

    ALL_REQUIRED_PASS = "ALL_REQUIRED_PASS"
    REQUIRED_GATE_FAILED = "REQUIRED_GATE_FAILED"
    REQUIRED_GATE_UNVERIFIABLE = "REQUIRED_GATE_UNVERIFIABLE"
    REPAIR_EXHAUSTED = "REPAIR_EXHAUSTED"
    MANUAL_POLICY = "MANUAL_POLICY"


class RepairStopReason(enum.StrEnum):
    """Repair 终止原因。"""

    PASS_ALL_REQUIRED = "PASS_ALL_REQUIRED"
    NO_ACTION = "NO_ACTION"
    NO_IMPROVEMENT = "NO_IMPROVEMENT"
    MAX_ROUNDS = "MAX_ROUNDS"
    UNVERIFIABLE = "UNVERIFIABLE"
    MANUAL_REQUEST = "MANUAL_REQUEST"


class RuleLevel(enum.StrEnum):
    """验证分层。L1 技术 / L2 风格 / L3 分域。"""

    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class RuleScope(enum.StrEnum):
    """规则作用域。ITEM 单张 / BATCH 整批。"""

    ITEM = "ITEM"
    BATCH = "BATCH"
