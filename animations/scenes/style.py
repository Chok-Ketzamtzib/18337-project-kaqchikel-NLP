"""Palette, fonts, data locations, and layout helpers shared by every scene."""

import json
from pathlib import Path

from manim import DOWN, ManimColor, Text, VGroup, config

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "animations" / "data"
RESULTS_DIR = REPO_ROOT / "src" / "results"

FONT = "Segoe UI"
# Cascadia Mono draws ’ (U+2019) and ' (U+0027) visibly differently.
MONO = "Cascadia Mono"

CLASSICAL = ManimColor("#F2B84B")  # Kaqchikel Chronicle (Kiwujil)
MODERN = ManimColor("#4FB8E6")  # Tang/Bennett corpus
SHARED = ManimColor("#EDEDED")

CURLY, STRAIGHT = "\u2019", "'"
# Text(..., **APOSTROPHE_STYLE) colors each glottal-stop spelling by corpus.
APOSTROPHE_STYLE = {"t2c": {STRAIGHT: MODERN, CURLY: CLASSICAL}, "t2f": {STRAIGHT: MONO, CURLY: MONO}}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def is_portrait() -> bool:
    return config.frame_height > config.frame_width


class Portrait:
    """Mixin that renders a scene at 3:4: 1080x1440 at -qh, 480x640 at -ql.

    List it before the scene class, e.g. `class FooVertical(Portrait, Foo)`.
    Manim restores the config after every scene, so landscape scenes rendered
    in the same run are unaffected.
    """

    def __init__(self, *args, **kwargs):
        width = min(config.pixel_width, config.pixel_height)
        config.pixel_width = width
        config.pixel_height = round(width * 4 / 3)
        # 8 units across keeps text the same size in pixels as the 16:9 version.
        config.frame_width = 8.0
        config.frame_height = 8.0 * 4 / 3
        super().__init__(*args, **kwargs)


def text_block(text, font_size, max_width, font=FONT, **kwargs):
    """Centered lines of Text, wrapped at word breaks to fit `max_width`."""
    words = text.split(" ")
    lines, line = [], words[0]
    for word in words[1:]:
        if Text(f"{line} {word}", font=font, font_size=font_size).width <= max_width:
            line = f"{line} {word}"
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return VGroup(*[Text(l, font=font, font_size=font_size, **kwargs) for l in lines]).arrange(DOWN, buff=0.14)
