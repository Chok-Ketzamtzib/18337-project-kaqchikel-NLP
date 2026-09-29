"""Palette, fonts, and data locations shared by every scene."""

import json
from pathlib import Path

from manim import ManimColor

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "animations" / "data"
RESULTS_DIR = REPO_ROOT / "src" / "results"

FONT = "Segoe UI"
# Cascadia Mono draws ’ (U+2019) and ' (U+0027) visibly differently.
MONO = "Cascadia Mono"

CLASSICAL = ManimColor("#F2B84B")  # Kaqchikel Chronicle (Kiwujil)
MODERN = ManimColor("#4FB8E6")  # Tang/Bennett corpus
SHARED = ManimColor("#EDEDED")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
