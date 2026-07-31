"""SpecStyle 安全导出平面。

冻结于架构合同 ``architecture/contracts.md`` §13。

EXP-001A 负责输入模型、跨对象不变量、纯路径规划与 canonical
manifest/QA/credits/style_spec/payload/digest 的纯内存生成；不执行文件系统、
环境捕获、网络或 native syscall。EXP-001B 负责 trusted root fd、
stage/write/readback/fsync、native no-replace rename、安全清理与 ``ExportBundle``。

本模块仅文档字符串：不 re-export、不 alias、不 lazy-load 任何类型。
公共类型分别由 :mod:`specstyle.exporting.manifest` 与
:mod:`specstyle.exporting.bundle` 直接定义。
"""
