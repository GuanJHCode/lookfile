"""specstyle package 基础层级单测：import、版本、异常继承与抛出/捕获。

覆盖 BOOT-001 冻结合同：
- specstyle 可作为顶层 package 导入；
- specstyle.__version__ 严格等于 "0.1.0"；
- SpecStyleError 继承 Exception；
- DomainError / InfrastructureError 继承 SpecStyleError；
- 三者均可正常 raise 并被捕获。
"""

import pytest


def test_package_importable():
    """specstyle 必须可作为顶层 package 导入。"""
    import specstyle  # noqa: F401


def test_version_is_0_1_0():
    """__version__ 必须严格等于 0.1.0。"""
    import specstyle

    assert specstyle.__version__ == "0.1.0"


def test_specstyle_error_inherits_exception():
    """SpecStyleError 必须继承 Exception。"""
    from specstyle.errors import SpecStyleError

    assert issubclass(SpecStyleError, Exception)


def test_domain_error_inherits_specstyle_error():
    """DomainError 必须继承 SpecStyleError。"""
    from specstyle.errors import DomainError, SpecStyleError

    assert issubclass(DomainError, SpecStyleError)


def test_infrastructure_error_inherits_specstyle_error():
    """InfrastructureError 必须继承 SpecStyleError。"""
    from specstyle.errors import InfrastructureError, SpecStyleError

    assert issubclass(InfrastructureError, SpecStyleError)


def test_specstyle_error_raise_and_catch():
    """SpecStyleError 可正常抛出并被自身捕获。"""
    from specstyle.errors import SpecStyleError

    with pytest.raises(SpecStyleError):
        raise SpecStyleError("boom")


def test_domain_error_raise_and_catch_as_base():
    """DomainError 可抛出，且作为 SpecStyleError 也可被捕获。"""
    from specstyle.errors import DomainError, SpecStyleError

    with pytest.raises(DomainError):
        raise DomainError("domain")
    with pytest.raises(SpecStyleError):
        raise DomainError("as base")


def test_infrastructure_error_raise_and_catch_as_base():
    """InfrastructureError 可抛出，且作为 SpecStyleError 也可被捕获。"""
    from specstyle.errors import InfrastructureError, SpecStyleError

    with pytest.raises(InfrastructureError):
        raise InfrastructureError("infra")
    with pytest.raises(SpecStyleError):
        raise InfrastructureError("as base")
