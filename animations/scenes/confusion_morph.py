"""Too good to be true? The confusion matrix, one dot per test sentence.

Each of the 1,606 test sentences is a dot in its confusion-matrix cell: the row
is the corpus it came from, the column the model's answer. The dots move
between cells as the scene steps from the word model to raw character
trigrams (regime A) and then removes the spelling cues (regimes B and D of
confound_check.jl).

Reads src/results/train_metrics.json and src/results/confound_check.json.

Render from animations/:
    uv run manim -pqh scenes/confusion_morph.py ConfusionMorph          # 16:9, 1920x1080
    uv run manim -pqh scenes/confusion_morph.py ConfusionMorphVertical  # 3:4, 1080x1440
"""

import numpy as np
from manim import *

from style import (
    APOSTROPHE_STYLE, CLASSICAL, CURLY, FONT, MODERN, RESULTS_DIR, STRAIGHT, Captions, Portrait, is_portrait,
    load_json,
)

ERROR = ManimColor("#FF5A5F")
NAMES = ("Chronicle", "Tang/Bennett")
STEPS = (
    ("word_unigram", "word model"),
    ("A_raw", "character trigrams"),
    ("B_apostrophe_fold", f"trigrams, {CURLY} folded into {STRAIGHT}"),
    ("D_shared_charset", "trigrams, shared characters only"),
)


def cells_of(flat):
    """[c→c, m→c, c→m, m→m] (Julia's column-major confusion matrix) as rows [true][predicted]."""
    return np.array(flat).reshape(2, 2, order="F")


def percent(x):
    return f"{100 * x:.1f}%"


class ConfusionMorph(Captions, Scene):
    def construct(self):
        confound = load_json(RESULTS_DIR / "confound_check.json")
        train = load_json(RESULTS_DIR / "train_metrics.json")
        regimes = {r["regime"]: r for r in confound["regimes"]}
        matrices = {"word_unigram": cells_of(train["word_unigram"]["confusion_matrix"])}
        accuracy = {"word_unigram": train["word_unigram"]["test_accuracy_serial"]}
        for name in ("A_raw", "B_apostrophe_fold", "D_shared_charset"):
            matrices[name] = cells_of(regimes[name]["confusion_matrix"])
            accuracy[name] = regimes[name]["test_accuracy"]
        per_row = int(matrices["A_raw"][0].sum())
        n_test = confound["test_count"]
        self.portrait = portrait = is_portrait()

        self.title("Too good to be true?")

        # The empty matrix.
        self.cell = cell = 3.0 if portrait else 2.6
        self.origin = np.array([0.35, -0.75, 0]) if portrait else np.array([-2.2, -0.25, 0])
        squares = VGroup()
        for r in range(2):
            for c in range(2):
                sq = Square(side_length=cell, stroke_color=GREY_C, stroke_width=2)
                squares.add(sq.move_to(self.cell_center(r, c)))
        col_heads = VGroup(*[
            Text(n, font=FONT, font_size=22, color=col).next_to(self.cell_center(0, c) + cell / 2 * UP, UP, buff=0.12)
            for c, (n, col) in enumerate(zip(NAMES, (CLASSICAL, MODERN)))
        ])
        row_heads = VGroup(*[
            Text(n, font=FONT, font_size=22, color=col).rotate(PI / 2)
            .next_to(self.cell_center(r, 0) + cell / 2 * LEFT, LEFT, buff=0.12)
            for r, (n, col) in enumerate(zip(NAMES, (CLASSICAL, MODERN)))
        ])
        predicted = Text("the model says", font=FONT, font_size=20, color=GREY_B).next_to(col_heads, UP, buff=0.12)
        truth = Text("written in", font=FONT, font_size=20, color=GREY_B).rotate(PI / 2).next_to(row_heads, LEFT, buff=0.12)
        frame = VGroup(squares, col_heads, row_heads, predicted, truth)
        self.caption(f"{n_test:,} test sentences, {per_row} from each corpus")
        self.play(Create(squares), FadeIn(col_heads), FadeIn(row_heads), FadeIn(predicted), FadeIn(truth), run_time=1.5)

        # One dot per test sentence; dot k of a row sits in the correct cell
        # while k < the number that row got right.
        self.dot_step = (cell - 0.25) / 29
        dots = [[Dot(radius=0.36 * self.dot_step, color=(CLASSICAL, MODERN)[r], fill_opacity=0.95)
                 for _ in range(per_row)] for r in range(2)]
        m = matrices["word_unigram"]
        for r in range(2):
            for k, d in enumerate(dots[r]):
                d.move_to(self.dot_target(r, k, m))
        all_dots = [d for row in dots for d in row]
        self.play(LaggedStart(*[FadeIn(d, scale=0.3) for d in all_dots], lag_ratio=0.0008), run_time=2.5)

        red = VGroup(*[
            squares[1].copy().set_stroke(ERROR, width=4), squares[2].copy().set_stroke(ERROR, width=4),
        ])
        counts = self.counts(m)
        panel = self.panel(STEPS[0][1], accuracy["word_unigram"], m)
        self.caption("Rows: where a sentence comes from. Columns: what the model says")
        self.play(FadeIn(counts), FadeIn(panel, shift=0.2 * UP), run_time=1)
        self.wait(2)
        self.caption(f"Red: mistakes. The word model gets {self.wrong(m)} wrong")
        self.play(Create(red), run_time=1)
        self.wait(2.5)

        for key, label in STEPS[1:]:
            new = matrices[key]
            moves = [d.animate.move_to(self.dot_target(r, k, new)) for r in range(2) for k, d in enumerate(dots[r])
                     if not np.allclose(d.get_center(), self.dot_target(r, k, new))]
            new_counts = self.counts(new)
            new_panel = self.panel(label, accuracy[key], new)
            if key == "A_raw":
                self.caption(f"Character trigrams: only {self.wrong(new)} wrong")
            elif key == "B_apostrophe_fold":
                self.caption(f"Now write the glottal stop the same way in both: {CURLY} becomes {STRAIGHT}")
            else:
                self.caption("Then drop every character only one corpus uses")
            self.play(*moves, ReplacementTransform(counts, new_counts), *self.roll(panel, new_panel), run_time=2.5)
            counts, panel, m = new_counts, new_panel, new
            self.wait(1.5)
            if key == "A_raw":
                self.caption(f"{percent(accuracy[key])}. What is it looking at?")
                self.wait(3)
            elif key == "B_apostrophe_fold":
                self.caption(f"{self.wrong(new)} wrong: most of the advantage came from one character")
                self.wait(3)

        self.caption(f"Without spelling cues, trigrams ({percent(accuracy['D_shared_charset'])}) "
                     f"fall below words ({percent(accuracy['word_unigram'])})")
        self.wait(5)

    def cell_center(self, r, c):
        return self.origin + self.cell * np.array([c - 0.5, 0.5 - r, 0])

    def dot_target(self, r, k, m):
        n_right = m[r, r]
        c, slot = (r, k) if k < n_right else (1 - r, k - n_right)
        corner = self.cell_center(r, c) + (self.cell / 2 - 0.125) * np.array([-1, 1, 0])
        return corner + self.dot_step * np.array([slot % 29 + 0.5, -(slot // 29 + 0.5), 0])

    def wrong(self, m):
        return int(m[0, 1] + m[1, 0])

    def counts(self, m):
        labels = VGroup()
        for r in range(2):
            for c in range(2):
                color = WHITE if r == c else ERROR
                t = Text(f"{m[r, c]}", font=FONT, font_size=40 if r == c else 34, weight=BOLD, color=color)
                t.add_background_rectangle(color=BLACK, opacity=0.75, buff=0.08)
                labels.add(t.move_to(self.cell_center(r, c) + self.cell * 0.3 * DOWN))
        return labels

    def panel(self, label, accuracy, m):
        heading = Text(label, font=FONT, font_size=24, color=GREY_B, **APOSTROPHE_STYLE)
        big = Text(percent(accuracy), font=FONT, font_size=60, weight=BOLD)
        errors = Text(f"{self.wrong(m)} wrong", font=FONT, font_size=24, color=ERROR)
        if self.portrait:
            side = VGroup(heading, errors).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            group = VGroup(big, side).arrange(RIGHT, buff=0.35).move_to([0, 3.55, 0])
        else:
            group = VGroup(heading, big, errors).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            group.move_to([2.4, 0.3, 0], aligned_edge=LEFT)
        return group


class ConfusionMorphVertical(Portrait, ConfusionMorph):
    """The same scene at 3:4 for Instagram."""
