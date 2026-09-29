"""Every character trigram is a point in a cube.

The axes are a trigram's first, second, and third characters, sorted by
Unicode codepoint, the order a computer sorts them in. The Chronicle writes
the glottal stop as ’ (U+2019) and Tang/Bennett as ' (U+0027), so the two
spellings sit at opposite ends of every axis. Folding ’ into ' (regime B in
confound_check.jl) moves the Chronicle's apostrophe trigrams onto
Tang/Bennett's; dropping characters only the Chronicle uses (regime D)
removes the rest of the spelling signal.

Reads animations/data/trigram_cube.json (written by export_data.jl) and the
accuracies in src/results/confound_check.json.

Render from animations/:
    uv run manim -pqh scenes/trigram_cube.py TrigramCube
"""

import math
from itertools import combinations, product

import numpy as np
from manim import *

from style import (
    APOSTROPHE_STYLE, CLASSICAL, CURLY, DATA_DIR, FONT, MODERN, MONO, RESULTS_DIR, SHARED, STRAIGHT,
    load_json,
)

SIDE = 5.0
HALF = SIDE / 2
AXIS_NAMES = ("1st character", "2nd character", "3rd character")
DEMO = "k" + CURLY + "a"
CHAR_NAMES = {
    ",": "commas", ".": "periods", ":": "colons", ";": "semicolons", "-": "hyphens",
    "–": "dashes", "(": "parentheses", ")": "parentheses", "“": "quotation marks",
    "”": "quotation marks", "_": "underscores",
}


def chronicle_share(classical, modern, totals):
    """Share of a trigram's per-token rate that comes from the Chronicle."""
    c = classical / totals["classical"]
    m = modern / totals["modern"]
    return c / (c + m)


def share_color(share):
    if share < 0.5:
        return interpolate_color(MODERN, SHARED, share / 0.5)
    return interpolate_color(SHARED, CLASSICAL, (share - 0.5) / 0.5)


def percent(x):
    return f"{100 * x:.1f}%"


def score_text(x):
    return Text(percent(x), font=FONT, font_size=54, weight=BOLD)


def describe_chars(chars):
    names = []
    for ch in sorted(chars):
        name = "digits" if ch.isdigit() else CHAR_NAMES.get(ch, ch)
        if name not in names:
            names.append(name)
    return " and ".join([", ".join(names[:-1]), names[-1]] if len(names) > 1 else names)


def unit(axis):
    v = np.zeros(3)
    v[axis] = 1.0
    return v


def edge_point(axis, value, offset):
    """Point beside the cube edge that carries `axis`, pushed `offset` outward."""
    p = np.full(3, -HALF - offset)
    p[axis] = value
    return p


def wall(axis, value, color):
    """Square across the cube where coordinate `axis` equals `value`."""
    a, b = [k for k in range(3) if k != axis]
    corners = []
    for u, v in ((-HALF, -HALF), (HALF, -HALF), (HALF, HALF), (-HALF, HALF)):
        p = np.zeros(3)
        p[axis], p[a], p[b] = value, u, v
        corners.append(p)
    return Polygon(
        *corners, stroke_color=color, stroke_width=2, stroke_opacity=0,
        fill_color=color, fill_opacity=0,
    )


def legend_row(color, text):
    return VGroup(
        Dot(radius=0.08, color=color),
        Text(text, font=FONT, font_size=20, color=GREY_A),
    ).arrange(RIGHT, buff=0.15)


class TrigramCube(ThreeDScene):
    def construct(self):
        cube = load_json(DATA_DIR / "trigram_cube.json")
        confound = load_json(RESULTS_DIR / "confound_check.json")
        accuracy = {r["regime"]: r["test_accuracy"] for r in confound["regimes"]}
        word_accuracy = confound["word_unigram_baseline"]["test_accuracy"]
        totals = cube["trigram_totals"]
        rows = cube["trigrams"]
        # ^ and $ are the sentence-boundary padding added by the trigram extractor.
        kept_chars = set(cube["shared_chars_after_fold"]) | {"^", "$"}

        alphabet = sorted({ch for row in rows for ch in row["trigram"]})
        index = {ch: i for i, ch in enumerate(alphabet)}

        def coord(ch):
            return (index[ch] / (len(alphabet) - 1) - 0.5) * SIDE

        def position(trigram):
            return np.array([coord(ch) for ch in trigram])

        self.caption_mob = None
        self.set_camera_orientation(phi=68 * DEGREES, theta=-52 * DEGREES, zoom=0.74, frame_center=[0, 0, -0.35])

        title = Text("Every trigram is a point in a cube", font=FONT, font_size=34).to_corner(UL, buff=0.4)
        self.fix_in_frame(title)
        self.play(Write(title), run_time=1.5)

        # The empty cube, its axes, and a few tick labels.
        corners = [np.array(c) for c in product((-HALF, HALF), repeat=3)]
        edges = VGroup(*[
            Line(a, b, stroke_color=GREY_D, stroke_width=1.5)
            for a, b in combinations(corners, 2) if np.count_nonzero(a != b) == 1
        ])
        ticks = {}
        for axis in range(3):
            ticks[axis, STRAIGHT] = Text(STRAIGHT, font=MONO, font_size=40, weight=BOLD, color=MODERN)
            ticks[axis, CURLY] = Text(CURLY, font=MONO, font_size=40, weight=BOLD, color=CLASSICAL)
            if DEMO[axis] not in (STRAIGHT, CURLY):
                ticks[axis, DEMO[axis]] = Text(DEMO[axis], font=MONO, font_size=24, color=GREY_B)
        for (axis, ch), label in ticks.items():
            label.move_to(edge_point(axis, coord(ch), 0.3))
        names = [
            Text(name, font=FONT, font_size=20, color=GREY_B).move_to(edge_point(axis, 0, 0.65))
            for axis, name in enumerate(AXIS_NAMES)
        ]
        strip = Text(
            "".join("␣" if ch == " " else ch for ch in alphabet), font=MONO, font_size=30,
            t2c=APOSTROPHE_STYLE["t2c"],
        ).next_to(title, DOWN, aligned_edge=LEFT, buff=0.35)
        curly_walls = VGroup(*[wall(a, coord(CURLY), CLASSICAL) for a in range(3)])
        straight_walls = VGroup(*[wall(a, coord(STRAIGHT), MODERN) for a in range(3)])
        self.add(curly_walls, straight_walls)
        self.face_camera(*ticks.values(), *names)
        self.fix_in_frame(strip)
        self.caption(f"Each axis runs through the same {len(alphabet)} characters, in Unicode order")
        self.play(Create(edges), *[FadeIn(m) for m in (*ticks.values(), *names)], FadeIn(strip), run_time=2)
        self.begin_ambient_camera_rotation(rate=0.04)
        self.wait(2)

        # One trigram's characters are its coordinates.
        self.play(FadeOut(strip), run_time=0.5)
        self.caption("The three characters of a trigram are its coordinates")
        word = Text(DEMO, font=MONO, font_size=64).to_corner(UR, buff=0.6)
        self.fix_in_frame(word)
        self.play(FadeIn(word, shift=0.2 * DOWN), run_time=0.8)
        stops = [np.full(3, -HALF)]
        for axis, ch in enumerate(DEMO):
            stop = stops[-1].copy()
            stop[axis] = coord(ch)
            stops.append(stop)
        tracer = Dot(stops[0], radius=0.08, color=CLASSICAL)
        self.face_camera(tracer)
        self.play(FadeIn(tracer, scale=0.3), run_time=0.5)
        trail = VGroup()
        for axis, ch in enumerate(DEMO):
            leg = DashedLine(stops[axis], stops[axis + 1], color=CLASSICAL, stroke_width=2)
            trail.add(leg)
            self.play(
                word[axis].animate.set_color(CLASSICAL),
                Indicate(ticks[axis, ch], color=CLASSICAL, scale_factor=1.6),
                tracer.animate.move_to(stops[axis + 1]),
                Create(leg),
                run_time=1.2,
            )
        self.wait(0.5)

        # The most common trigrams of both corpora.
        rates = [max(r["classical"] / totals["classical"], r["modern"] / totals["modern"]) for r in rows]
        lo, hi = math.log(min(rates)), math.log(max(rates))
        dots = {}
        for row, rate in zip(rows, rates):
            color = share_color(chronicle_share(row["classical"], row["modern"], totals))
            radius = 0.04 + 0.05 * (math.log(rate) - lo) / (hi - lo)
            dots[row["trigram"]] = Dot(position(row["trigram"]), radius=radius, color=color, fill_opacity=0.85)
        demo_dot = dots.pop(DEMO)
        others = [dots[r["trigram"]] for _, r in sorted(zip(rates, rows), key=lambda p: -p[0]) if r["trigram"] in dots]
        dots[DEMO] = tracer
        self.face_camera(*others)

        legend = VGroup(
            legend_row(CLASSICAL, "only in the Chronicle"),
            legend_row(SHARED, "in both corpora"),
            legend_row(MODERN, "only in Tang/Bennett"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(title, DOWN, aligned_edge=LEFT, buff=0.35)
        size_note = Text("bigger = more frequent", font=FONT, font_size=18, color=GREY_B)
        size_note.next_to(legend, DOWN, aligned_edge=LEFT, buff=0.2)
        self.fix_in_frame(legend, size_note)
        self.caption(f"The {cube['top_per_class']} most common trigrams from each corpus")
        self.play(
            FadeOut(word), FadeOut(trail),
            tracer.animate.set(width=demo_dot.width).set_color(demo_dot.get_color()),
            LaggedStart(*[FadeIn(d, scale=0.3) for d in others], lag_ratio=0.01),
            FadeIn(legend), FadeIn(size_note),
            run_time=4,
        )
        self.wait(2)

        # Two spellings of the glottal stop, on opposite walls.
        curly_dots = [dots[r["trigram"]] for r in rows if CURLY in r["trigram"]]
        straight_dots = [dots[r["trigram"]] for r in rows if STRAIGHT in r["trigram"]]
        self.caption(f"The Chronicle writes the glottal stop as {CURLY}  (U+2019)")
        self.play(
            curly_walls.animate.set_fill(opacity=0.1).set_stroke(opacity=0.7),
            *[d.animate.scale(1.6) for d in curly_dots],
            run_time=1.5,
        )
        self.play(*[d.animate.scale(1 / 1.6) for d in curly_dots], run_time=0.8)
        self.caption(f"Tang/Bennett writes it as {STRAIGHT}  (U+0027), and never uses {CURLY}")
        self.play(
            straight_walls.animate.set_fill(opacity=0.1).set_stroke(opacity=0.7),
            *[d.animate.scale(1.6) for d in straight_dots],
            run_time=1.5,
        )
        self.play(*[d.animate.scale(1 / 1.6) for d in straight_dots], run_time=0.8)

        panel_title = Text("char-trigram accuracy", font=FONT, font_size=22, color=GREY_B)
        panel_title.to_corner(UR, buff=0.5)
        score = score_text(accuracy["A_raw"]).next_to(panel_title, DOWN, aligned_edge=RIGHT, buff=0.15)
        self.fix_in_frame(panel_title, score)
        self.caption("So the spelling alone gives away which corpus a sentence came from")
        self.play(FadeIn(panel_title), FadeIn(score, shift=0.2 * UP), run_time=1)
        self.wait(2.5)

        # Fold ’ into ': the Chronicle's apostrophe trigrams land on Tang/Bennett's.
        groups = {}
        for row in rows:
            groups.setdefault(row["folded"], []).append(row)
        keepers = {key: dots.get(key, dots[members[0]["trigram"]]) for key, members in groups.items()}
        moves = []
        for row in rows:
            dot, key = dots[row["trigram"]], row["folded"]
            color = share_color(chronicle_share(row["folded_classical"], row["folded_modern"], totals))
            if row["trigram"] != key:
                moves.append(dot.animate.move_to(position(key)).set_color(color).set(width=keepers[key].width))
            elif STRAIGHT in key:
                moves.append(dot.animate.set_color(color))
        shift = coord(STRAIGHT) - coord(CURLY)
        self.caption(f"Fold {CURLY} into {STRAIGHT}: the same sound, now the same character")
        self.play(*moves, *[w.animate.shift(shift * unit(a)) for a, w in enumerate(curly_walls)], run_time=3.5)
        self.remove(*[d for d in dots.values() if d not in keepers.values()])
        folded_score = score_text(accuracy["B_apostrophe_fold"]).move_to(score, aligned_edge=RIGHT)
        self.play(
            *[w.animate.set_color(SHARED).set_fill(opacity=0.08) for w in (*curly_walls, *straight_walls)],
            *self.roll(score, folded_score),
            run_time=1.2,
        )
        score = folded_score
        self.caption("The apostrophe trigrams now appear in both corpora")
        self.wait(2)
        self.play(FadeOut(curly_walls), FadeOut(straight_walls), run_time=1)

        # Whatever only the Chronicle still uses goes next.
        leftovers = [d for key, d in keepers.items() if not set(key) <= kept_chars]
        extra = {ch for key in keepers for ch in key} - kept_chars
        self.caption(f"Still Chronicle-only: {len(leftovers)} trigrams with {describe_chars(extra)}")
        self.play(*[d.animate.scale(2).set_color(CLASSICAL) for d in leftovers], run_time=1.2)
        self.wait(1.5)
        shared_score = score_text(accuracy["D_shared_charset"]).move_to(score, aligned_edge=RIGHT)
        word_line = Text(f"word model: {percent(word_accuracy)}", font=FONT, font_size=22, color=GREY_B)
        word_line.next_to(shared_score, DOWN, aligned_edge=RIGHT, buff=0.2)
        self.fix_in_frame(word_line)
        self.caption("Keep only characters both corpora use")
        self.play(*[FadeOut(d, scale=0.2) for d in leftovers], *self.roll(score, shared_score), run_time=1.5)
        self.play(FadeIn(word_line, shift=0.2 * UP), run_time=0.8)
        self.caption("Without spelling cues, trigrams trail the plain word model")
        self.wait(5)

    def fix_in_frame(self, *mobjects):
        """Pin overlays to the screen without adding them yet, so they can fade in."""
        self.camera.add_fixed_in_frame_mobjects(*mobjects)

    def face_camera(self, *mobjects):
        """Keep 3D labels and dots facing the camera, each about its own center."""
        self.camera.add_fixed_orientation_mobjects(*mobjects)

    def roll(self, old, new):
        """Replace one pinned overlay with another, rolling upward."""
        self.fix_in_frame(new)
        return FadeOut(old, shift=0.3 * UP), FadeIn(new, shift=0.3 * UP)

    def caption(self, text):
        anims = []
        if self.caption_mob is not None:
            anims.append(FadeOut(self.caption_mob, shift=0.2 * UP))
        self.caption_mob = Text(text, font=FONT, font_size=28, **APOSTROPHE_STYLE).to_edge(DOWN, buff=0.4)
        self.fix_in_frame(self.caption_mob)
        anims.append(FadeIn(self.caption_mob, shift=0.2 * UP))
        self.play(LaggedStart(*anims, lag_ratio=0.5), run_time=0.8)
