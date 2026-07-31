"""WF-003 pipeline cancel token."""

from __future__ import annotations

from specstyle.workflow.real_pipeline import CancelToken


def test_cancel_token() -> None:
    token = CancelToken()
    assert token.cancelled is False
    token.cancel()
    assert token.cancelled is True
