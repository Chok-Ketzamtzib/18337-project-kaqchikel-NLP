"""Naive Bayes draws one flat cut.

Multinomial Naive Bayes scores a sentence by adding one log-odds weight per
trigram: score = offset + x·w, with w the difference of the two classes'
log-likelihoods. export_data.jl splits that sum by trigram type (apostrophe
trigrams, other symbols, letters only), so every test sentence is a point in
3D and the decision boundary, score = 0, is the plane where the three parts
add up to -offset. Turned edge-on, the plane becomes a cut on a number line:
the scene stacks every test sentence by score and replays regimes A, B, and D
of confound_check.jl on those histograms.

Reads animations/data/nb_scores.json (written by export_data.jl).

Render from animations/:
    uv run manim -pqh scenes/decision_boundary.py DecisionBoundary          # 16:9, 1920x1080
    uv run manim -pqh scenes/decision_boundary.py DecisionBoundaryVertical  # 3:4, 1080x1440
"""

import math
from dataclasses import dataclass

import numpy as np
from manim import *

from style import APOSTROPHE_STYLE, CLASSICAL, CURLY, DATA_DIR, FONT, MODERN, SHARED, STRAIGHT, load_json

ERROR = ManimColor("#FF5A5F")
PER_CLASS = 200
CUBE_LIMIT = 550.0  # score units from the origin to the end of each 3D axis
HALF = 2.4  # scene units for CUBE_LIMIT
SAMPLE_LIMIT = 800.0  # sampled sentences have every part within this, so none fly off screen
SCORE_LIMIT = 1600.0  # histogram range, in score units, either side of the cut
BINS = 64
# (key in nb_scores.json, axis label); x, y, z in that order.
AXES = (("apostrophe", "apostrophes"), ("letters", "letters only"), ("other_symbols", "punctuation, digits"))
REGIMES = {
    "A_raw": "raw text",
    "B_apostrophe_fold": f"{CURLY} folded into {STRAIGHT}",
    "D_shared_charset": "shared characters only",
}
# Looking along (1, -1, 0), which lies in the plane, with the normal (1, 1, 1)
# pointing right: the plane shows as a vertical line and x on screen is the score.
EDGE_ON = {"phi": 90 * DEGREES, "theta": -45 * DEGREES, "gamma": -math.atan(1 / math.sqrt(2))}


@dataclass
class Layout:
    portrait: bool
    zoom: float
    text_width: float
    hist_width: float
    baseline: float
    bar_height: float  # tallest bar in any regime


def current_layout():
    if config.frame_height > config.frame_width:
        return Layout(True, zoom=0.9, text_width=config.frame_width - 0.8,
                      hist_width=7.2, baseline=-0.8, bar_height=3.0)
    return Layout(False, zoom=0.95, text_width=config.frame_width - 1.4,
                  hist_width=12.4, baseline=-0.5, bar_height=2.45)


def text_block(text, font_size, max_width, **kwargs):
    """Centered lines of Text, wrapped at word breaks to fit `max_width`."""
    words = text.split(" ")
    lines, line = [], words[0]
    for word in words[1:]:
        if Text(f"{line} {word}", font=FONT, font_size=font_size).width <= max_width:
            line = f"{line} {word}"
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return VGroup(*[Text(l, font=FONT, font_size=font_size, **kwargs) for l in lines]).arrange(DOWN, buff=0.14)


def percent(x):
    return f"{100 * x:.1f}%"


def unit(axis):
    v = np.zeros(3)
    v[axis] = 1.0
    return v


def legend_row(color, text):
    return VGroup(
        Dot(radius=0.08, color=color),
        Text(text, font=FONT, font_size=20, color=GREY_A),
    ).arrange(RIGHT, buff=0.15)


class DecisionBoundary(ThreeDScene):
    def construct(self):
        data = load_json(DATA_DIR / "nb_scores.json")
        self.lay = lay = current_layout()
        classical = np.array(data["labels"]) == "classical"
        n_test = len(classical)
        regimes = data["regimes"]
        scores = {name: np.array(regimes[name]["score"]) for name in (*REGIMES, "word_unigram")}
        wrong = {name: int(np.sum((s >= 0) != classical)) for name, s in scores.items()}

        raw = regimes["A_raw"]
        parts = np.column_stack([raw["parts"][key] for key, _ in AXES])
        inside = np.all(np.abs(parts) <= SAMPLE_LIMIT, axis=1)
        rng = np.random.default_rng(18337)
        sample = np.concatenate([
            rng.choice(np.flatnonzero(inside & mask), PER_CLASS, replace=False) for mask in (classical, ~classical)
        ])
        rng.shuffle(sample)
        scale = HALF / CUBE_LIMIT

        self.caption_mob = None
        self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES, zoom=lay.zoom)

        title = text_block("Naive Bayes draws one flat cut", 34, lay.text_width)
        if lay.portrait:
            title.to_edge(UP, buff=0.5)
        else:
            title.to_corner(UL, buff=0.4)
        self.fix_in_frame(title)
        self.caption("Naive Bayes scores a sentence by adding one weight per trigram")
        self.play(Write(title), run_time=1.5)
        self.wait(1.5)

        # Three partial sums, one axis each.
        axes = VGroup(*[
            Line(-HALF * unit(k), HALF * unit(k), stroke_color=GREY_C, stroke_width=2) for k in range(3)
        ])
        names = [
            Text(name, font=FONT, font_size=20, color=GREY_B).move_to((HALF + 0.45) * unit(k))
            for k, (_, name) in enumerate(AXES)
        ]
        self.face_camera(*names)
        self.caption("Split that sum three ways, by what each trigram contains")
        self.play(Create(axes), *[FadeIn(n) for n in names], run_time=2)
        self.begin_ambient_camera_rotation(rate=0.05)
        self.wait(1.5)

        dots = [
            Dot(parts[i] * scale, radius=0.05, color=CLASSICAL if classical[i] else MODERN, fill_opacity=0.9)
            for i in sample
        ]
        self.face_camera(*dots)
        legend = VGroup(legend_row(CLASSICAL, "Chronicle sentence"), legend_row(MODERN, "Tang/Bennett sentence"))
        if lay.portrait:
            legend.arrange(RIGHT, buff=0.5).to_edge(DOWN, buff=1.75)
        else:
            legend.arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(title, DOWN, aligned_edge=LEFT, buff=0.35)
        self.fix_in_frame(legend)
        self.caption(f"{2 * PER_CLASS} test sentences, each placed by its three sums")
        self.play(LaggedStart(*[FadeIn(d, scale=0.3) for d in dots], lag_ratio=0.01), FadeIn(legend), run_time=3.5)
        self.wait(1.5)
        self.caption("Tang/Bennett has no punctuation or digits, so its sentences lie flat")
        self.wait(3)

        # score = 0 is a plane through the cube.
        shift = -raw["offset"] * scale / math.sqrt(3) * np.ones(3) / math.sqrt(3)
        hexagon = [HALF * np.array(p, dtype=float) + shift for p in
                   ((1, -1, 0), (1, 0, -1), (0, 1, -1), (-1, 1, 0), (-1, 0, 1), (0, -1, 1))]
        plane = Polygon(*hexagon, stroke_color=SHARED, stroke_width=2, fill_color=SHARED, fill_opacity=0.14)
        self.caption("It says Chronicle when the three sums add up to more than 0")
        self.play(FadeIn(plane), run_time=1.5)
        self.wait(1.5)
        panel = self.panel("A_raw", raw["accuracy"], wrong["A_raw"], n_test)
        self.caption("That boundary is a flat plane: Naive Bayes is linear")
        self.play(FadeIn(panel, shift=0.2 * UP), run_time=1)
        self.wait(2.5)

        # Edge-on, the plane is a line and left to right is the score.
        self.stop_ambient_camera_rotation()
        self.caption("Turn the plane edge-on, and left to right is the total score")
        self.move_camera(**EDGE_ON, added_anims=[FadeOut(n) for n in names], run_time=3.5)
        self.wait(1)

        flat = [
            Dot(self.camera.project_point(d.get_center()) * [1, 1, 0], radius=d.radius,
                color=d.get_color(), fill_opacity=0.9)
            for d in dots
        ]
        self.fix_in_frame(*flat)
        self.remove(*dots)
        self.add(*flat)

        half_w = lay.hist_width / 2
        top, bottom = lay.baseline + lay.bar_height, lay.baseline - lay.bar_height
        cut = Line([0, bottom - 0.15, 0], [0, top + 0.15, 0], stroke_color=SHARED, stroke_width=3)
        cut_label = Text("score = 0", font=FONT, font_size=20, color=GREY_B).next_to(cut, UP, buff=0.1)
        number_line = Line([-half_w, lay.baseline, 0], [half_w, lay.baseline, 0], stroke_color=GREY_C, stroke_width=2)
        self.fix_in_frame(cut, cut_label, number_line)

        def x_of(score):
            return float(np.clip(score / SCORE_LIMIT, -1, 1)) * half_w

        self.caption("So each sentence collapses to one number")
        self.play(
            *[f.animate.move_to([x_of(scores["A_raw"][i]), lay.baseline, 0]) for f, i in zip(flat, sample)],
            FadeOut(axes), FadeOut(plane), FadeOut(legend),
            Create(number_line), Create(cut), FadeIn(cut_label),
            run_time=2.5,
        )

        # Every test sentence, stacked by score.
        edges = np.linspace(-SCORE_LIMIT, SCORE_LIMIT, BINS + 1)
        counts = {
            name: (np.histogram(scores[name][classical], edges)[0], np.histogram(scores[name][~classical], edges)[0])
            for name in REGIMES
        }
        count_height = lay.bar_height / max(max(c.max(), m.max()) for c, m in counts.values())
        bar_width = lay.hist_width / BINS

        def histogram(name, show_errors=True):
            gold, blue = counts[name]
            bars = VGroup()
            for k in range(BINS):
                x = -half_w + (k + 0.5) * bar_width
                for n, direction, color, wrong_side in (
                    (gold[k], UP, CLASSICAL, edges[k + 1] <= 0),
                    (blue[k], DOWN, MODERN, edges[k] >= 0),
                ):
                    h = n * count_height
                    fill = ERROR if show_errors and wrong_side else color
                    bar = Rectangle(width=0.86 * bar_width, height=h, stroke_width=0, fill_color=fill, fill_opacity=0.9)
                    bars.add(bar.move_to([x, lay.baseline, 0] + direction * h / 2))
            self.fix_in_frame(bars)
            return bars

        rows = VGroup(
            Text("Chronicle sentences", font=FONT, font_size=20, color=CLASSICAL)
            .move_to([-half_w, top, 0], aligned_edge=UL),
            Text("Tang/Bennett sentences", font=FONT, font_size=20, color=MODERN)
            .move_to([half_w, bottom, 0], aligned_edge=DR),
        )
        self.fix_in_frame(rows)
        hist = histogram("A_raw", show_errors=False)
        self.caption(f"Stack all {n_test:,} test sentences by score")
        self.play(
            *[GrowFromEdge(bar, DOWN if i % 2 == 0 else UP) for i, bar in enumerate(hist)],
            *[FadeOut(f) for f in flat], FadeIn(rows),
            run_time=2,
        )
        self.wait(1.5)
        highlighted = histogram("A_raw")
        self.caption("Red: sentences that land on the wrong side of the cut")
        self.play(ReplacementTransform(hist, highlighted), Indicate(panel[-1], color=ERROR), run_time=1.5)
        hist = highlighted
        self.wait(2)

        # Replay the confound: take the spelling cues away.
        for name, lines in (
            ("B_apostrophe_fold", (f"Fold {CURLY} into {STRAIGHT}: the same sound, now the same character",
                                   "The humps slide toward the cut, and more sentences cross it")),
            ("D_shared_charset", ("Keep only characters both corpora use",)),
        ):
            new_hist = histogram(name)
            new_panel = self.panel(name, regimes[name]["accuracy"], wrong[name], n_test)
            self.caption(lines[0])
            self.play(ReplacementTransform(hist, new_hist), *self.roll(panel, new_panel), run_time=2.5)
            hist, panel = new_hist, new_panel
            self.wait(1.5)
            for line in lines[1:]:
                self.caption(line)
                self.wait(2.5)

        word_line = Text(
            f"word model: {percent(regimes['word_unigram']['accuracy'])}, {wrong['word_unigram']} wrong",
            font=FONT, font_size=22, color=GREY_B,
        )
        if lay.portrait:
            word_line.next_to(panel[2], DOWN, aligned_edge=LEFT, buff=0.12)
        else:
            word_line.next_to(panel, DOWN, aligned_edge=RIGHT, buff=0.2)
        self.fix_in_frame(word_line)
        self.play(FadeIn(word_line, shift=0.2 * UP), run_time=0.8)
        self.caption("Without spelling cues, trigrams make more mistakes than whole words")
        self.wait(5)

    def panel(self, name, accuracy, wrong, n_test):
        heading = Text(f"char trigrams, {REGIMES[name]}", font=FONT, font_size=22, color=GREY_B, **APOSTROPHE_STYLE)
        big = Text(percent(accuracy), font=FONT, font_size=54, weight=BOLD)
        errors = Text(f"{wrong} of {n_test:,} on the wrong side", font=FONT, font_size=20, color=ERROR)
        if self.lay.portrait:
            side = VGroup(heading, errors).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            VGroup(big, side).arrange(RIGHT, buff=0.35).to_edge(UP, buff=1.35)
        else:
            VGroup(heading, big, errors).arrange(DOWN, aligned_edge=RIGHT, buff=0.12).to_corner(UR, buff=0.45)
        group = VGroup(heading, big, errors)
        self.fix_in_frame(group)
        return group

    def fix_in_frame(self, *mobjects):
        """Pin overlays to the screen without adding them yet, so they can fade in."""
        self.camera.add_fixed_in_frame_mobjects(*mobjects)

    def face_camera(self, *mobjects):
        """Keep 3D labels and dots facing the camera, each about its own center."""
        self.camera.add_fixed_orientation_mobjects(*mobjects)

    def roll(self, old, new):
        """Replace one pinned overlay with another, rolling upward."""
        return FadeOut(old, shift=0.3 * UP), FadeIn(new, shift=0.3 * UP)

    def caption(self, text):
        anims = []
        if self.caption_mob is not None:
            anims.append(FadeOut(self.caption_mob, shift=0.2 * UP))
        self.caption_mob = text_block(text, 28, self.lay.text_width, **APOSTROPHE_STYLE).to_edge(DOWN, buff=0.4)
        self.fix_in_frame(self.caption_mob)
        anims.append(FadeIn(self.caption_mob, shift=0.2 * UP))
        self.play(LaggedStart(*anims, lag_ratio=0.5), run_time=0.8)


class DecisionBoundaryVertical(DecisionBoundary):
    """The same scene at 3:4 for Instagram: 1080x1440 at -qh, 480x640 at -ql."""

    def __init__(self, **kwargs):
        width = min(config.pixel_width, config.pixel_height)
        config.pixel_width = width
        config.pixel_height = round(width * 4 / 3)
        # 8 units across keeps text the same size in pixels as the 16:9 version.
        config.frame_width = 8.0
        config.frame_height = 8.0 * 4 / 3
        super().__init__(**kwargs)
