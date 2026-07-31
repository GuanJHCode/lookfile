"""Gradio application shell — lazy import; services injected."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from specstyle.errors import DomainError
from specstyle.ui.presenters import present_spec_compile
from specstyle.ui.view_models import SpecEditorView


@dataclass(frozen=True, slots=True)
class UiServices:
    """Injected application services — UI never constructs domain backends."""

    compile_spec: Callable[[str], SpecEditorView]
    # Optional job runners omitted for shell-only launch.


def build_default_services(context: object) -> UiServices:
    from specstyle.spec.compiled_models import CompilerContext

    if type(context) is not CompilerContext:
        raise DomainError("invalid compiler context")

    def _compile(text: str) -> SpecEditorView:
        return present_spec_compile(text, context)

    return UiServices(_compile)


def create_app(services: UiServices) -> Any:
    """Create Gradio Blocks app. Requires `gradio` installed."""
    if type(services) is not UiServices:
        raise DomainError("invalid ui services")
    try:
        import gradio as gr
    except ImportError as exc:
        raise DomainError("gradio not installed") from exc

    def on_compile(spec_text: str) -> str:
        view = services.compile_spec(spec_text or "")
        if view.compile_ok:
            return f"OK {view.spec_id} hash={view.compiled_hash}"
        return "ERR " + "; ".join(view.errors)

    with gr.Blocks(title="SpecStyle") as demo:
        gr.Markdown("# SpecStyle StyleOps")
        spec_in = gr.Textbox(label="Style Spec YAML/JSON", lines=12)
        out = gr.Textbox(label="Compile result")
        btn = gr.Button("Compile")
        btn.click(on_compile, inputs=[spec_in], outputs=[out])
    return demo


def launch_app(services: UiServices, **kwargs: object) -> Any:
    app = create_app(services)
    return app.launch(**kwargs)
