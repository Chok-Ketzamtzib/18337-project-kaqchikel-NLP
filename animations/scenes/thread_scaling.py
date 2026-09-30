"""Why 8 threads are not 8 times faster.

Each stage is drawn as a timeline with one lane per thread. Only one phase of
each stage runs on every thread: counting trigrams in
char_trigram_tfidf_matrix, and summing features per class in
train_multinomial_nb. Everything else runs on thread 1 while the other lanes
sit idle, so adding threads shrinks one block and leaves the rest (Amdahl's
law). In training, the one-thread part is mostly sparse(transpose(X)).

Reads animations/data/thread_phases.json (written by export_phases.jl) for
the timelines, and the measured speedups in src/results/benchmark_results.json
(written by src/benchmark.jl) for the closing chart. export_phases.jl collects
garbage before every run and benchmark.jl does not, so the timelines are
faster than the benchmark's medians; the speedups agree to within noise.

Render from animations/:
    uv run manim -pqh scenes/thread_scaling.py ThreadScaling          # 16:9
    uv run manim -pqh scenes/thread_scaling.py ThreadScalingVertical  # 3:4
"""

from dataclasses import dataclass

import numpy as np
from manim import *

from style import DATA_DIR, FONT, MONO, RESULTS_DIR, Portrait, is_portrait, load_json, text_block

SPLITS = ManimColor("#7BD389")
MERGE = ManimColor("#F79D65")
BUILD = ManimColor("#B388FF")
TRANSPOSE = ManimColor("#FF6B6B")
OTHER = GREY_B
PHASE_COLORS = {
    "count": SPLITS, "merge": MERGE, "build": BUILD,
    "setup": OTHER, "transpose": TRANSPOSE, "accumulate": SPLITS, "finish": OTHER,
}
LEGENDS = {
    "feature_extraction": (
        ("count trigrams (splits)", SPLITS), ("merge thread results", MERGE), ("build sparse matrix", BUILD),
    ),
    "naive_bayes_train": (
        ("sparse(transpose(X))", TRANSPOSE), ("sum per class (splits)", SPLITS),
        ("merge thread results", MERGE), ("other steps", OTHER),
    ),
}
WORKERS = (1, 2, 4, 8)


@dataclass
class Layout:
    portrait: bool
    text_width: float
    x0: float  # left end of the time axis
    width: float  # length of the time axis
    axis_y: float
    lane_h: float
    lane_gap: float
    chart_origin: np.ndarray
    chart_size: float


def current_layout():
    if is_portrait():
        return Layout(True, text_width=config.frame_width - 0.8, x0=-3.2, width=6.7, axis_y=2.05,
                      lane_h=0.52, lane_gap=0.14, chart_origin=np.array([-2.7, -3.0, 0]), chart_size=5.0)
    return Layout(False, text_width=config.frame_width - 1.4, x0=-5.9, width=11.8, axis_y=1.8,
                  lane_h=0.4, lane_gap=0.12, chart_origin=np.array([-2.6, -2.1, 0]), chart_size=4.2)


def ms(seconds):
    v = 1e3 * seconds
    return f"{v:.0f} ms" if v >= 10 else f"{v:.1f} ms"


def times(x):
    return f"{x:.2f}×"


class ThreadScaling(Scene):
    def construct(self):
        phases = load_json(DATA_DIR / "thread_phases.json")
        bench = load_json(RESULTS_DIR / "benchmark_results.json")
        self.lay = lay = current_layout()
        self.caption_mob = None
        features, training = phases["feature_extraction"], phases["naive_bayes_train"]

        title = text_block("Why 8 threads are not 8× faster", 34, lay.text_width)
        if lay.portrait:
            title.to_edge(UP, buff=0.5)
        else:
            title.to_corner(UL, buff=0.4)
        self.caption(f"Building trigram features for {phases['rows']:,} sentences, on one thread")
        self.play(Write(title), run_time=1.5)

        # One thread: count, then build.
        self.stage = features
        axis = self.time_axis(0.05)
        legend = self.legend("feature_extraction")
        timeline = self.timeline(1)
        panel = self.panel(1)
        tracks, splits, serial, numbers, end = timeline
        self.play(FadeIn(axis), FadeIn(legend), FadeIn(tracks), FadeIn(numbers), run_time=1)
        self.play(GrowFromEdge(splits[0], LEFT), run_time=1.5)
        self.play(*[GrowFromEdge(b, LEFT) for b in serial], run_time=1.2)
        self.play(Create(end), FadeIn(panel, shift=0.2 * UP), run_time=0.8)
        one = features["workers"]["1"]
        self.caption(f"Counting takes {ms(one['count'])}; building the matrix takes {ms(one['build'])}")
        self.wait(2)

        # More threads: the counting block splits, the rest stays on thread 1.
        for w, text in (
            (2, "Split the sentences across 2 threads: each counts half"),
            (4, "4 threads: thread 1 still merges and builds alone"),
            (8, "8 threads: counting shrinks less each time, merging grows"),
        ):
            self.caption(text)
            new_timeline, new_panel = self.timeline(w), self.panel(w)
            self.play(ReplacementTransform(timeline, new_timeline), *self.roll(panel, new_panel), run_time=1.8)
            timeline, panel = new_timeline, new_panel
            self.wait(1.8)
        self.caption("Grey is idle: 7 threads wait while thread 1 builds the matrix")
        self.play(Indicate(timeline[0][1:], color=WHITE, scale_factor=1.0), run_time=1.5)
        self.wait(1)

        serial_one = sum(one[p] for p in features["phases"] if p != features["parallel_phase"])
        total_one = sum(one[p] for p in features["phases"])
        best = total_one / serial_one
        build_block = timeline[2][-1]
        floor_label = Text("never splits", font=FONT, font_size=19, color=BLACK).move_to(build_block)
        self.caption("The one-thread steps set a floor that no thread count can lower")
        self.play(Indicate(build_block, color=BUILD, scale_factor=1.08), FadeIn(floor_label), run_time=1.2)
        self.wait(1.5)
        self.caption(f"Even unlimited threads could make this at most {best:.1f}× faster")
        self.wait(2.5)

        # Training: almost nothing to split.
        self.caption("Naive Bayes training has even less that splits")
        self.play(*[FadeOut(m) for m in (timeline, panel, axis, legend, floor_label)], run_time=0.8)
        self.stage = training
        axis = self.time_axis(0.001)
        legend = self.legend("naive_bayes_train")
        timeline, panel = self.timeline(1), self.panel(1)
        tracks, splits, serial, numbers, end = timeline
        blocks = sorted([*splits, *serial], key=lambda b: b.get_left()[0])
        self.play(FadeIn(axis), FadeIn(legend), FadeIn(tracks), FadeIn(numbers), run_time=1)
        self.play(Succession(*[GrowFromEdge(b, LEFT, run_time=0.5) for b in blocks]), run_time=2)
        self.play(Create(end), FadeIn(panel, shift=0.2 * UP), run_time=0.8)
        t1 = training["workers"]["1"]
        share = t1["transpose"] / sum(t1[p] for p in training["phases"])
        self.caption(f"{100 * share:.0f}% of it is sparse(transpose(X)), on one thread")
        self.play(Indicate(serial[1], color=TRANSPOSE, scale_factor=1.05), run_time=1.2)
        self.wait(1.5)
        self.caption(f"8 threads split the {ms(t1['accumulate'])} that can split")
        new_timeline, new_panel = self.timeline(8), self.panel(8)
        self.play(ReplacementTransform(timeline, new_timeline), *self.roll(panel, new_panel), run_time=1.8)
        timeline, panel = new_timeline, new_panel
        self.wait(2.5)

        # Measured speedups against the ideal line.
        self.play(*[FadeOut(m) for m in (timeline, panel, axis, legend)], run_time=0.8)
        self.caption("Speedup over one thread, as measured by benchmark.jl")
        frame, point = self.chart_frame()
        ideal = DashedLine(point(1, 1), point(8, 8), color=GREY_B, stroke_width=3)
        ideal_label = Text("ideal", font=FONT, font_size=20, color=GREY_B).next_to(point(8, 8), RIGHT, buff=0.15)
        self.play(FadeIn(frame), run_time=0.8)
        self.play(Create(ideal), FadeIn(ideal_label), run_time=1)

        s = 1 - one[features["parallel_phase"]] / total_one
        bound = DashedVMobject(
            VMobject().set_points_smoothly([point(w, 1 / (s + (1 - s) / w)) for w in np.linspace(1, 8, 40)]),
            num_dashes=30, dashed_ratio=0.5,
        ).set_stroke(SPLITS, width=3, opacity=0.9)
        curves = {}
        for key, color in (("feature_speedup_vs_serial", SPLITS), ("naive_bayes_speedup_vs_serial", TRANSPOSE)):
            values = [bench[key][str(w)] for w in WORKERS]
            pts = [point(w, v) for w, v in zip(WORKERS, values)]
            line = VMobject().set_points_as_corners(pts).set_stroke(color, width=4)
            dots = VGroup(*[Dot(p, radius=0.07, color=color) for p in pts])
            label = Text(times(values[-1]), font=FONT, font_size=22, color=color).next_to(pts[-1], RIGHT, buff=0.15)
            curves[key] = VGroup(line, dots, label)
        bound_label = Text("best possible", font=FONT, font_size=18, color=SPLITS)
        bound_label.set_opacity(0.8).next_to(point(8, 1 / (s + (1 - s) / 8)), UR, buff=0.08)

        self.caption(f"Features: {times(bench['feature_speedup_vs_serial']['8'])} on 8 threads")
        self.play(Create(curves["feature_speedup_vs_serial"]), run_time=1.5)
        self.play(Create(bound), FadeIn(bound_label), run_time=1.2)
        self.wait(1.5)
        self.caption(f"Training: {times(bench['naive_bayes_speedup_vs_serial']['8'])}, a little slower than one thread")
        self.play(Create(curves["naive_bayes_speedup_vs_serial"]), run_time=1.5)
        self.wait(1.5)
        self.caption("More threads only speed up the part that splits")
        self.wait(4)

    def lane_y(self, i):
        lay = self.lay
        return lay.axis_y - 0.3 - lay.lane_h / 2 - i * (lay.lane_h + lay.lane_gap)

    def block(self, x, lane, width, color, opacity=0.95):
        rect = Rectangle(width=width, height=self.lay.lane_h, stroke_width=0, fill_color=color, fill_opacity=opacity)
        return rect.move_to([x + width / 2, self.lane_y(lane), 0])

    def timeline(self, workers):
        """VGroup(tracks, split blocks, one-thread blocks, lane numbers, end marker)."""
        lay, stage = self.lay, self.stage
        t = stage["workers"][str(workers)]
        total = sum(t[p] for p in stage["phases"]) * self.scale
        tracks = VGroup(*[self.block(lay.x0, i, total, GREY_D, opacity=0.5) for i in range(workers)])
        splits, serial = VGroup(), VGroup()
        x = lay.x0
        for p in stage["phases"]:
            width = t[p] * self.scale
            if p == stage["parallel_phase"]:
                splits.add(*[self.block(x, i, width, PHASE_COLORS[p]) for i in range(workers)])
            else:
                serial.add(self.block(x, 0, width, PHASE_COLORS[p]))
            x += width
        numbers = VGroup(*[
            Text(str(i + 1), font=FONT, font_size=18, color=GREY_B).move_to([lay.x0 - 0.3, self.lane_y(i), 0])
            for i in range(workers)
        ])
        top = self.lane_y(0) + lay.lane_h / 2 + 0.1
        bottom = self.lane_y(workers - 1) - lay.lane_h / 2 - 0.1
        end = DashedLine([x, top, 0], [x, bottom, 0], color=WHITE, stroke_width=2)
        return VGroup(tracks, splits, serial, numbers, end)

    def time_axis(self, step):
        """Axis for the current stage, long enough for its slowest run; sets self.scale."""
        lay, stage = self.lay, self.stage
        slowest = max(sum(t[p] for p in stage["phases"]) for t in stage["workers"].values())
        limit = step * np.ceil(slowest / step)
        self.scale = lay.width / limit
        line = Line([lay.x0, lay.axis_y, 0], [lay.x0 + lay.width, lay.axis_y, 0], stroke_color=GREY_C, stroke_width=2)
        ticks = VGroup()
        for k in range(int(round(limit / step)) + 1):
            x = lay.x0 + k * step * self.scale
            ticks.add(Line([x, lay.axis_y - 0.07, 0], [x, lay.axis_y + 0.07, 0], stroke_color=GREY_C, stroke_width=2))
            ticks.add(Text(ms(k * step) if k else "0", font=FONT, font_size=18, color=GREY_B)
                      .move_to([x, lay.axis_y + 0.28, 0]))
        header = Text("threads", font=FONT, font_size=18, color=GREY_B)
        header.move_to([lay.x0 - 0.3, lay.axis_y - 0.15, 0])
        return VGroup(line, ticks, header)

    def legend(self, stage_name):
        rows = [
            VGroup(Square(0.22, stroke_width=0, fill_color=color, fill_opacity=0.95),
                   Text(text, font=MONO if "(X)" in text else FONT, font_size=19, color=GREY_A))
            .arrange(RIGHT, buff=0.15)
            for text, color in LEGENDS[stage_name]
        ]
        group = VGroup(*rows).arrange_in_grid(cols=2, buff=(0.5, 0.14), col_alignments="ll")
        group.move_to([0, self.lay.axis_y + 0.95, 0])
        if not self.lay.portrait:
            group.align_to([self.lay.x0, 0, 0], LEFT)
        return group

    def panel(self, workers):
        stage = self.stage
        t = stage["workers"][str(workers)]
        label = Text(f"{workers} thread{'s' if workers > 1 else ''}", font=FONT, font_size=22, color=GREY_B)
        total = Text(ms(sum(t[p] for p in stage["phases"])), font=FONT, font_size=46, weight=BOLD)
        if self.lay.portrait:
            return VGroup(label, total).arrange(RIGHT, buff=0.3).to_edge(UP, buff=1.25)
        return VGroup(label, total).arrange(DOWN, aligned_edge=RIGHT, buff=0.1).to_corner(UR, buff=0.45)

    def chart_frame(self):
        """Axes for speedup against thread count, and a data-to-scene mapping."""
        origin, size = self.lay.chart_origin, self.lay.chart_size

        def point(workers, speedup):
            return origin + np.array([(workers - 1) / 7 * size, speedup / 8 * size, 0])

        axes = VGroup(
            Line(point(1, 0), point(8, 0), stroke_color=GREY_C, stroke_width=2),
            Line(point(1, 0), point(1, 8), stroke_color=GREY_C, stroke_width=2),
        )
        labels = VGroup()
        for w in WORKERS:
            labels.add(Text(str(w), font=FONT, font_size=18, color=GREY_B).next_to(point(w, 0), DOWN, buff=0.12))
            labels.add(Text(f"{w}×", font=FONT, font_size=18, color=GREY_B).next_to(point(1, w), LEFT, buff=0.12))
        labels.add(Text("threads", font=FONT, font_size=20, color=GREY_B).next_to(point(4.5, 0), DOWN, buff=0.45))
        labels.add(Text("speedup", font=FONT, font_size=20, color=GREY_B)
                   .rotate(PI / 2).next_to(point(1, 4), LEFT, buff=0.6))
        return VGroup(axes, labels), point

    def roll(self, old, new):
        return FadeOut(old, shift=0.3 * UP), FadeIn(new, shift=0.3 * UP)

    def caption(self, text):
        anims = []
        if self.caption_mob is not None:
            anims.append(FadeOut(self.caption_mob, shift=0.2 * UP))
        self.caption_mob = text_block(text, 28, self.lay.text_width).to_edge(DOWN, buff=0.4)
        anims.append(FadeIn(self.caption_mob, shift=0.2 * UP))
        self.play(LaggedStart(*anims, lag_ratio=0.5), run_time=0.8)


class ThreadScalingVertical(Portrait, ThreadScaling):
    """The same scene at 3:4 for Instagram."""
