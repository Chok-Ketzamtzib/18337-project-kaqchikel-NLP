"""Palette, fonts, data locations, and layout helpers shared by every scene."""

import json
from pathlib import Path

from manim import (
    DOWN, GREY_B, UP, UL, WHITE, FadeIn, FadeOut, LaggedStart, ManimColor, Text, ValueTracker, VGroup, Write, config,
)

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

# The Tang/Bennett corpus authors permit posting derived figures only if the
# source is clearly indicated, so every scene ends on this card.
CREDITS = (
    ("Data: Kaqchikel written corpus by Kevin Tang and Ryan Bennett, used with permission", 26, WHITE),
    ("Tang & Bennett (2018), J. Acoust. Soc. Am. 144(2); "
     "Bennett, Tang & Ajsivinac Sian (2018), Laboratory Phonology 9(1)", 20, GREY_B),
    (f"and the Kaqchikel Chronicle (Kiwujil Xajila{CURLY})", 26, WHITE),
    ("Analysis and animation: William J. Wakefield", 20, GREY_B),
)


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


def text_block(text, font_size, max_width, font=FONT, buff=0.14, **kwargs):
    """Centered lines of Text, wrapped at word breaks to fit `max_width`.

    The returned group's `lines` holds the string of each line.
    """
    words = text.split(" ")
    lines, line = [], words[0]
    for word in words[1:]:
        if Text(f"{line} {word}", font=font, font_size=font_size).width <= max_width:
            line = f"{line} {word}"
        else:
            lines.append(line)
            line = word
    lines.append(line)
    group = VGroup(*[Text(l, font=font, font_size=font_size, **kwargs) for l in lines]).arrange(DOWN, buff=buff)
    group.lines = lines
    return group


def text_width() -> float:
    """Width for titles and captions: the frame less its side margins."""
    return config.frame_width - (0.8 if is_portrait() else 1.4)


def end_credits(scene, pin=None):
    """Clear the frame and hold a card crediting the data sources and the author.

    3D scenes pass their `fix_in_frame` as `pin` so the card stays flat on screen.
    """
    card = VGroup(*[text_block(text, size, text_width(), color=color) for text, size, color in CREDITS])
    card.arrange(DOWN, buff=0.4)
    if pin is not None:
        pin(card)
    visible = [m for m in scene.mobjects if not isinstance(m, ValueTracker)]
    if visible:
        scene.play(*[FadeOut(m) for m in visible], run_time=1)
    scene.play(FadeIn(card, shift=0.2 * UP), run_time=1)
    scene.wait(4)


class Captions:
    """Title and bottom captions for 2D scenes, laid out for either orientation."""

    caption_mob = None

    def title(self, text):
        title = text_block(text, 34, text_width(), **APOSTROPHE_STYLE)
        if is_portrait():
            title.to_edge(UP, buff=0.5)
        else:
            title.to_corner(UL, buff=0.4)
        self.play(Write(title), run_time=1.5)
        return title

    def caption(self, text):
        anims = []
        if self.caption_mob is not None:
            anims.append(FadeOut(self.caption_mob, shift=0.2 * UP))
        self.caption_mob = text_block(text, 28, text_width(), **APOSTROPHE_STYLE).to_edge(DOWN, buff=0.4)
        anims.append(FadeIn(self.caption_mob, shift=0.2 * UP))
        self.play(LaggedStart(*anims, lag_ratio=0.5), run_time=0.8)

    def roll(self, old, new):
        """Replace one overlay with another, rolling upward."""
        return FadeOut(old, shift=0.3 * UP), FadeIn(new, shift=0.3 * UP)
