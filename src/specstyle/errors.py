"""specstyle 领域异常基础层级。

仅定义继承关系（SpecStyleError ← Exception；DomainError/InfrastructureError ← SpecStyleError）。
错误码、错误上下文与序列化在后续 Task 实现，当前不引入。
"""


class SpecStyleError(Exception):
    """所有 specstyle 异常的根。"""


class DomainError(SpecStyleError):
    """领域语义错误：业务规则违反、Spec 校验失败等。"""


class InfrastructureError(SpecStyleError):
    """基础设施错误：GPU/ROCm、文件 IO、依赖缺失等运行环境问题。"""
