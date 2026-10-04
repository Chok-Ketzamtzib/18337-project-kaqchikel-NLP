"""Reading Kaqchikel three characters at a time.

The classifier's features are character trigrams: every run of three
characters in a lowercased line padded with ^ and $ (_line_char_trigrams in
kaq_pipeline.jl). The scene slides that window over one Chronicle phrase,
gives each trigram its regime-A Naive Bayes weight, and then shows which
trigrams weigh the most for each corpus.

Reads animations/data/trigram_window.json (written by export_data.jl).

Render from animations/:
    uv run manim -pqh scenes/trigram_window.py TrigramWindow          # 16:9, 1920x1080
    uv run manim -pqh scenes/trigram_window.py TrigramWindowVertical  # 3:4, 1080x1440
"""

from manim import *

from style import (
    APOSTROPHE_STYLE, CLASSICAL, CURLY, DATA_DIR, FONT, MODERN, MONO, STRAIGHT, Captions, Portrait, end_credits,
    is_portrait, load_json,
)

SPACE = "\u00b7"  # a visible stand-in for the space character
CHAR_STYLE = {"t2c": {**APOSTROPHE_STYLE["t2c"], SPACE: GREY_D}, "t2f": APOSTROPHE_STYLE["t2f"]}
TOP_N = 50


def visible(s):
    return s.replace(" ", SPACE)


def signed(x):
    return f"{x:+.1f}".replace("-", "\u2212")


def chip(trigram, font_size=26, stroke=GREY_C):
    label = Text(visible(trigram), font=MONO, font_size=font_size, **CHAR_STYLE)
    box = RoundedRectangle(corner_radius=0.08, width=label.width + 0.3, height=0.5 * font_size / 26,
                           stroke_color=stroke, stroke_width=2, fill_opacity=0)
    box.stretch_to_fit_width(max(box.width, 1.1 * font_size / 26))
    return VGroup(box, label.move_to(box))


class TrigramWindow(Captions, Scene):
    def construct(self):
        data = load_json(DATA_DIR / "trigram_window.json")
        portrait = is_portrait()
        trigrams = data["trigrams"]
        n_curly = sum(CURLY in t["trigram"] for t in trigrams)

        title = self.title("Reading Kaqchikel three characters at a time")
        phrase = Text(data["phrase"], font=MONO, font_size=64, **CHAR_STYLE)
        phrase.move_to(UP * (2.4 if portrait else 1.2))
        self.caption("A line from the Kaqchikel Chronicle")
        self.play(FadeIn(phrase, shift=0.2 * UP), run_time=1.2)
        self.wait(2)

        # Lowercase, pad with ^ and $, and box every character.
        padded = visible(data["padded"])
        chars = Text(padded, font=MONO, font_size=48, **CHAR_STYLE)
        chars.set_width(config.frame_width - (0.9 if portrait else 2.6))
        chars.move_to(UP * (3.2 if portrait else 1.9))
        step = (chars[-1].get_center()[0] - chars[0].get_center()[0]) / (len(padded) - 1)
        cells = VGroup(*[
            Square(side_length=step, stroke_color=GREY_D, stroke_width=1.5).move_to([c.get_center()[0], chars.get_y(), 0])
            for c in chars
        ])
        self.caption(f"Lowercase it, and mark where the line starts (^) and ends ($). {SPACE} is a space")
        self.play(ReplacementTransform(phrase, chars), Create(cells), run_time=1.8)
        self.wait(2)

        # Slide the window; each stop drops a trigram chip into a grid.
        cols = 4 if portrait else 8
        chips = VGroup(*[chip(t["trigram"]) for t in trigrams])
        chips.arrange_in_grid(cols=cols, buff=(0.3, 0.35))
        chips.next_to(cells, DOWN, buff=1.0 if portrait else 0.9)
        window = SurroundingRectangle(cells[0:3], buff=0.04, stroke_color=WHITE, stroke_width=4, corner_radius=0.06)
        self.caption("Slide a three-character window along it")
        self.play(Create(window), run_time=0.6)
        for i, c in enumerate(chips):
            slow = i < 3
            if i > 0:
                self.play(window.animate.move_to(cells[i:i + 3]), run_time=0.45 if slow else 0.14)
            self.play(FadeIn(c[0]), TransformFromCopy(chars[i:i + 3], c[1]), run_time=0.7 if slow else 0.2)
            if i == 2:
                self.caption(f"{len(trigrams)} windows, {len(trigrams)} trigrams")
        self.play(FadeOut(window), run_time=0.4)
        self.wait(1)

        self.caption(f"{n_curly} of the {len(trigrams)} contain {CURLY}")
        self.play(*[
            c[0].animate.set_stroke(CLASSICAL, width=3) if CURLY in t["trigram"] else c.animate.set_opacity(0.35)
            for c, t in zip(chips, trigrams)
        ], run_time=1)
        self.wait(2.5)

        # Weights: the chips become the labels of a bar chart.
        row_h = 0.42 if portrait else 0.335
        top_y = 3.6 if portrait else 2.75
        label_x = -3.1 if portrait else -3.7
        zero_x = -2.05 if portrait else -2.3
        unit = (4.3 if portrait else 5.6) / max(t["weight"] for t in trigrams)
        targets, bars, values = VGroup(), VGroup(), VGroup()
        for i, t in enumerate(trigrams):
            y = top_y - i * row_h
            target = chip(t["trigram"], font_size=22, stroke=CLASSICAL if CURLY in t["trigram"] else GREY_C)
            targets.add(target.move_to([label_x, y, 0]))
            w = t["weight"]
            color = CLASSICAL if w > 0 else MODERN
            bar = Rectangle(width=max(abs(w) * unit, 0.02), height=0.6 * row_h, stroke_width=0,
                            fill_color=color, fill_opacity=0.9)
            bars.add(bar.move_to([zero_x, y, 0], aligned_edge=LEFT if w > 0 else RIGHT))
            value = Text(signed(w), font=FONT, font_size=20, color=GREY_A)
            values.add(value.next_to(bar, RIGHT if w > 0 else LEFT, buff=0.1))
        # Negative bars run left, under the chips; put their values after the zero line instead.
        for v, b, t in zip(values, bars, trigrams):
            if t["weight"] < 0:
                v.next_to([zero_x, b.get_y(), 0], RIGHT, buff=0.1)
        axis = Line([zero_x, top_y + 0.35, 0], [zero_x, top_y - len(trigrams) * row_h + 0.1, 0],
                    stroke_color=GREY_C, stroke_width=2)
        ends = VGroup(
            Text(f"{STRAIGHT} Tang/Bennett", font=FONT, font_size=18, color=MODERN, **APOSTROPHE_STYLE)
            .next_to(axis.get_top(), LEFT, buff=0.15),
            Text(f"Chronicle {CURLY}", font=FONT, font_size=18, color=CLASSICAL, **APOSTROPHE_STYLE)
            .next_to(axis.get_top(), RIGHT, buff=0.15),
        )
        self.caption("Naive Bayes gives every trigram a weight: how strongly it points to one corpus")
        self.play(
            FadeOut(title), FadeOut(chars), FadeOut(cells),
            *[Transform(c, t) for c, t in zip(chips, targets)],
            run_time=1.6,
        )
        self.play(Create(axis), FadeIn(ends), run_time=0.6)
        self.play(LaggedStart(*[GrowFromEdge(b, LEFT if t["weight"] > 0 else RIGHT)
                                for b, t in zip(bars, trigrams)], lag_ratio=0.08),
                  LaggedStart(*[FadeIn(v) for v in values], lag_ratio=0.08), run_time=2.5)
        self.wait(1.5)
        curly_w = [t["weight"] for t in trigrams if CURLY in t["trigram"]]
        other_w = [t["weight"] for t in trigrams if CURLY not in t["trigram"]]
        self.caption(f"Every {CURLY} trigram weighs {min(curly_w):.0f} to {max(curly_w):.0f}. "
                     f"The other {len(other_w)} stay under {max(other_w):.0f}")
        self.wait(3.5)

        # The strongest trigrams overall, per corpus.
        chart = VGroup(chips, bars, values, axis, ends)
        grid_cols = 5 if portrait else 10
        grid_y = 0.4 if portrait else -0.05
        for key, mark, color, name in (("top_classical", CURLY, CLASSICAL, "the Chronicle"),
                                       ("top_modern", STRAIGHT, MODERN, "Tang/Bennett")):
            top = data[key][:TOP_N]
            n_mark = sum(mark in t["trigram"] for t in top)
            grid = VGroup(*[
                chip(t["trigram"], font_size=22, stroke=color if mark in t["trigram"] else GREY_D) for t in top
            ]).arrange_in_grid(cols=grid_cols, buff=(0.22, 0.2)).move_to(UP * grid_y)
            for g, t in zip(grid, top):
                if mark in t["trigram"]:
                    g[0].set_fill(color, opacity=0.18)
            heading = Text(f"The {TOP_N} strongest clues for {name}", font=FONT, font_size=30, color=color)
            heading.next_to(grid, UP, buff=0.45)
            self.play(FadeOut(chart), FadeIn(heading, shift=0.2 * DOWN),
                      LaggedStart(*[FadeIn(g, scale=0.6) for g in grid], lag_ratio=0.03), run_time=2.2)
            self.caption(f"{n_mark} of the {TOP_N} contain {mark}")
            self.wait(3.5)
            chart = VGroup(grid, heading)

        self.caption("The strongest clues are spelling conventions, not the language")
        self.wait(4)
        end_credits(self)


class TrigramWindowVertical(Portrait, TrigramWindow):
    """The same scene at 3:4 for Instagram."""
