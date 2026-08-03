from __future__ import annotations

import sys

import pytest

from specstyle.ui.app import UiServices, create_app
from specstyle.ui.view_models import (
    PreviewReadinessUiView,
    PreviewRunUiView,
    SpecEditorView,
)
from tests.unit.ui.test_app_production_replay import _fake_gradio


def test_preview_controls_use_independent_view_and_gpu_run_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buttons = []
    monkeypatch.setitem(sys.modules, "gradio", _fake_gradio(buttons))
    calls: list[tuple[object, ...]] = []

    def run(*args: object) -> PreviewRunUiView:
        calls.append(args)
        return PreviewRunUiView(
            "preview-ui",
            "COMPLETED",
            "OK",
            "preview",
            ("/display/output.png",),
            "a" * 64,
            "NOT_RUN",
            "NOT_RUN",
            "NOT_RUN",
        )

    services = UiServices(
        lambda _text: SpecEditorView("1.0", "spec", True, (), "b" * 64),
        get_preview_readiness=lambda: PreviewReadinessUiView("CONFIGURED", "READY"),
        run_preview_job=run,
    )
    create_app(services)
    readiness = next(
        item for item in buttons if item.label == "Refresh preview readiness"
    )
    preview = next(item for item in buttons if item.label == "Run preview")
    assert preview.event["concurrency_id"] == "production-run"
    assert preview.event["concurrency_limit"] == 2
    assert len(preview.event["inputs"]) == 5
    assert len(preview.event["outputs"]) == 3
    assert readiness.event["concurrency_id"] == "production-control"

    status, images, evidence = preview.event["fn"](
        "source", "style", "spec", "positive", "negative"
    )
    assert status == "preview-ui\tCOMPLETED\tprofile=preview\tOK"
    assert images == ["/display/output.png"]
    assert "verification=NOT_RUN" in evidence
    assert calls == [("source", "style", "spec", "positive", "negative")]
    assert readiness.event["fn"]() == "CONFIGURED\tREADY"
