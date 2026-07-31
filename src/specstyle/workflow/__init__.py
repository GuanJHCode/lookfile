"""SpecStyle workflow Job 状态机与本地 Store（WF-001）。

按 architecture/contracts.md §5 状态转换、§2 值对象、§4 不变量冻结（derived）。
本模块只依赖 :mod:`specstyle.domain` 与 :mod:`specstyle.errors`，不导入
generation/verification/repair/exporting/spec 业务类型，维持三平面解耦。
"""
