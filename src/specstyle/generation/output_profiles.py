"""Three output profiles: shared post-process after main generation."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

from specstyle.errors import DomainError
from specstyle.observability.hashing import hash_bytes

OutputProfileName = str  # xhs_grid | talking_head_cover | background_sequence


@dataclass(frozen=True, slots=True)
class ProfileLayout:
    name: str
    size: tuple[int, int]
    safe_zone: tuple[float, float, float, float]  # x0,y0,x1,y1 normalized


PROFILES: dict[str, ProfileLayout] = {
    "xhs_grid": ProfileLayout("xhs_grid", (1080, 1080), (0.08, 0.08, 0.92, 0.92)),
    "talking_head_cover": ProfileLayout(
        "talking_head_cover", (1080, 1440), (0.1, 0.12, 0.9, 0.88)
    ),
    "background_sequence": ProfileLayout(
        "background_sequence", (1920, 1080), (0.05, 0.08, 0.95, 0.92)
    ),
}


def render_output_profile(
    source_png: bytes,
    profile: str,
    *,
    text: str | None = None,
    sequence_index: int | None = None,
) -> bytes:
    if type(source_png) is not bytes or not source_png:
        raise DomainError("invalid source image")
    if profile not in PROFILES:
        raise DomainError("unknown output profile")
    layout = PROFILES[profile]
    if profile == "background_sequence":
        if (
            type(sequence_index) is not int
            or isinstance(sequence_index, bool)
            or sequence_index < 0
        ):
            raise DomainError("sequence index required")
    image = Image.open(BytesIO(source_png)).convert("RGB")
    out = image.resize(layout.size)
    if text:
        draw = ImageDraw.Draw(out)
        # Deterministic default font
        font = ImageFont.load_default()
        # Anchor text in safe zone top-left
        x = int(layout.safe_zone[0] * layout.size[0])
        y = int(layout.safe_zone[1] * layout.size[1])
        draw.text((x, y), text[:64], fill=(255, 255, 255), font=font)
    if sequence_index is not None and profile == "background_sequence":
        draw = ImageDraw.Draw(out)
        draw.text((10, 10), f"seq={sequence_index}", fill=(200, 200, 200))
    buf = BytesIO()
    out.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def profile_content_hash(png: bytes) -> str:
    return hash_bytes(png).value
