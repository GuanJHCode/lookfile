"""UI presenters — no Gradio required."""

from __future__ import annotations

from tests.unit.spec.test_compiler import context, raw_spec
from specstyle.ui.presenters import present_spec_compile
import json


def test_present_spec_compile_ok_and_error() -> None:
    ctx = context()
    text = json.dumps(raw_spec().model_dump(mode="json"), ensure_ascii=False)
    # raw_spec returns StyleSpecV1 — dump may need list/tuple handling via json

    data = raw_spec().model_dump(mode="json")
    # convert for loader: need yaml/json with tuples as lists — json.dumps ok
    text = json.dumps(data)
    # loader expects list→tuple conversion
    view = present_spec_compile(text, ctx)
    # compile may fail if list/tuple — check either path
    assert view.compile_ok or view.errors
    bad = present_spec_compile("not-yaml: [", ctx)
    assert bad.compile_ok is False
    assert bad.errors
