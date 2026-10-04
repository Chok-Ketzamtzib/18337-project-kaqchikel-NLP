"""The whole training matrix, one point of light per nonzero.

Starts from a toy sentence-by-trigram grid and pulls back to all of X_train:
6,420 sentences by 7,784 trigrams. Rows are grouped by corpus and columns by
which apostrophe the trigram contains, so the spelling split shows as two dark
corners: Tang/Bennett rows never touch the ’ columns, and Chronicle rows almost
never touch the ' columns.

Reads animations/data/starfield.pgm and starfield.json (written by
export_data.jl). Each pixel counts the nonzeros in a 4x4 block of cells, too
coarse to recover any sentence.

Render from animations/:
    uv run manim -pqh scenes/starfield_zoom.py StarfieldZoom          # 16:9, 1920x1080
    uv run manim -pqh scenes/starfield_zoom.py StarfieldZoomVertical  # 3:4, 1080x1440
"""

import math

import numpy as np
from manim import *
from PIL import Image

from style import (
    APOSTROPHE_STYLE, CLASSICAL, CURLY, DATA_DIR, FONT, MODERN, STRAIGHT, Captions, Portrait, end_credits,
    is_portrait, load_json,
)

START_ZOOM = 40.0
BAND_ZOOM = 4.0
# Below this zoom the view samples a 2x2 max-pooled copy, so no point is lost when shrunk.
POOL_BELOW = 2.2


def colorize(counts, split_row, max_count):
    """Gold rows above `split_row`, blue below; brighter where a block holds more nonzeros."""
    rows = np.arange(counts.shape[0])[:, None, None]
    base = np.where(rows < split_row, np.array(CLASSICAL.to_rgb()), np.array(MODERN.to_rgb())) * 255
    level = np.where(counts > 0, 0.4 + 0.6 * np.sqrt(counts / max_count), 0)[..., None]
    rgb = (base * level).clip(0, 255).astype(np.uint8)
    alpha = np.full(counts.shape + (1,), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=2)


def redrawn(build):
    """A Group whose content is rebuilt every frame. Unlike always_redraw, this
    works for images, whose size changes with each crop."""
    holder = Group(build())

    def refresh(group):
        group.remove(*group.submobjects)
        group.add(build())

    return holder.add_updater(refresh)


def max_pool(counts, k):
    h, w = counts.shape[0] // k, counts.shape[1] // k
    return counts[:h * k, :w * k].reshape(h, k, w, k).max(axis=(1, 3))


class StarfieldZoom(Captions, Scene):
    def construct(self):
        meta = load_json(DATA_DIR / "starfield.json")
        counts = np.asarray(Image.open(DATA_DIR / "starfield.pgm"), dtype=float)
        portrait = is_portrait()
        self.split = split = meta["classical_rows"] / meta["block"]
        self.full = colorize(counts, split, counts.max())
        self.pooled = colorize(max_pool(counts, 2), split / 2, counts.max())
        self.rows, self.cols = counts.shape
        n_rows, n_cols, nnz = meta["rows"], meta["cols"], meta["nnz"]
        bands, by = meta["band_cols"], meta["nnz_by_class_band"]
        self.curly_col = bands["curly"] / meta["block"]
        self.straight_col = (bands["curly"] + bands["shared"]) / meta["block"]

        # The rectangle every view of the matrix is drawn into.
        width = config.frame_width - 0.6 if portrait else 5.2 * self.cols / self.rows
        height = width * self.rows / self.cols
        center = UP * 0.35 if portrait else RIGHT * 0.2 + DOWN * 0.15
        self.box = Rectangle(width=width, height=height).move_to(center)
        left, right = self.box.get_left()[0], self.box.get_right()[0]
        top, bottom = self.box.get_top()[1], self.box.get_bottom()[1]

        self.zoom = ValueTracker(START_ZOOM)
        self.cx = ValueTracker(300.0)
        self.cy = ValueTracker(400.0)
        self.show_dividers = ValueTracker(0.0)

        self.title(f"{nnz:,} points of light")

        # A toy grid first: a dot wherever a sentence contains a trigram.
        rng = np.random.default_rng(18337)
        cell = 0.42
        toy = VGroup()
        for r in range(6):
            for c in range(10):
                p = np.array([(c - 4.5) * cell, (2.5 - r) * cell, 0])
                toy.add(Square(side_length=cell, stroke_color=GREY_D, stroke_width=1).move_to(p))
                if rng.random() < 0.18:
                    toy.add(Dot(p, radius=0.07, color=CLASSICAL if r < 3 else MODERN))
        toy.move_to(center)
        self.caption("A row per sentence, a column per trigram, and a dot wherever the sentence has that trigram")
        self.play(FadeIn(toy, lag_ratio=0.02), run_time=2)
        self.wait(2.5)

        # Pull back from deep inside the matrix to all of it.
        view = redrawn(self.view)
        dividers = redrawn(self.dividers)
        border = self.box.copy().set_stroke(GREY_D, width=1.5)
        self.add(view, dividers)
        self.bring_to_front(toy)
        mid_x, mid_y = self.cols / 2, self.rows / 2
        self.caption(f"Now all of it: {n_rows:,} training sentences by {n_cols:,} trigrams")
        self.play(
            UpdateFromAlphaFunc(self.zoom, lambda m, a: m.set_value(START_ZOOM ** (1 - a))),
            self.cx.animate.set_value(mid_x), self.cy.animate.set_value(mid_y),
            toy.animate.scale(1 / START_ZOOM).set_opacity(0),
            FadeIn(border),
            run_time=4.5, rate_func=smooth,
        )
        self.remove(toy)
        density = nnz / (n_rows * n_cols)
        self.caption(f"{100 * (1 - density):.1f}% of the cells are empty. Each point sums a 4 × 4 block of cells")
        self.wait(3)

        # Label the row and column groups.
        split_y = top - split / self.rows * height
        row_labels = VGroup(
            Text(f"{meta['classical_rows']:,} Chronicle", font=FONT, font_size=22, color=CLASSICAL),
            Text(f"{n_rows - meta['classical_rows']:,} Tang/Bennett", font=FONT, font_size=22, color=MODERN),
        )
        col_labels = VGroup(
            Text(f"{bands['curly']} with {CURLY}", font=FONT, font_size=20, color=CLASSICAL, **APOSTROPHE_STYLE),
            Text(f"{bands['shared']:,} trigrams with neither", font=FONT, font_size=20, color=GREY_B),
            Text(f"{bands['straight']} with {STRAIGHT}", font=FONT, font_size=20, color=MODERN, **APOSTROPHE_STYLE),
        )
        edge_side, edge = (DOWN, bottom) if portrait else (UP, top)
        col_labels[0].next_to([left, edge, 0], edge_side, buff=0.12, aligned_edge=LEFT)
        col_labels[1].move_to([(left + right) / 2, col_labels[0].get_y(), 0])
        col_labels[2].next_to([right, edge, 0], edge_side, buff=0.12, aligned_edge=RIGHT)
        if portrait:
            row_labels[0].next_to([left, top, 0], UP, buff=0.12, aligned_edge=LEFT)
            row_labels[1].next_to(col_labels, DOWN, buff=0.14).align_to([left, 0, 0], LEFT)
        else:
            for lab, y in zip(row_labels, ((top + split_y) / 2, (split_y + bottom) / 2)):
                lab.rotate(PI / 2).next_to([left, y, 0], LEFT, buff=0.2)
        labels = VGroup(row_labels, col_labels)
        self.caption("Rows grouped by corpus, columns by which apostrophe the trigram has")
        self.play(self.show_dividers.animate.set_value(1), FadeIn(labels), run_time=2)
        self.wait(3)

        # Zoom into each apostrophe band where it crosses the corpus split.
        for band, mark in (("curly", CURLY), ("straight", STRAIGHT)):
            band_col = self.curly_col if band == "curly" else self.straight_col
            self.caption(f"Zoom in where the {mark} columns meet the corpus split")
            self.play(
                FadeOut(labels),
                self.zoom.animate.set_value(BAND_ZOOM),
                self.cx.animate.set_value(band_col), self.cy.animate.set_value(split),
                run_time=3,
            )
            notes = VGroup(*[
                VGroup(
                    Text(f"{name} rows", font=FONT, font_size=24, color=color),
                    Text(f"{by[key][band]:,} points", font=FONT, font_size=34 if portrait else 30, weight=BOLD,
                         color=color),
                ).arrange(RIGHT if portrait else DOWN, aligned_edge=DOWN if portrait else LEFT, buff=0.2)
                for name, key, color in (("Chronicle", "classical", CLASSICAL), ("Tang/Bennett", "modern", MODERN))
            ])
            if portrait:
                notes[0].next_to(self.box, UP, buff=0.15)
                notes[1].next_to(self.box, DOWN, buff=0.15)
            else:
                room = config.frame_width / 2 - right - 0.5
                for note, y in zip(notes, ((top + split_y) / 2, (split_y + bottom) / 2)):
                    if note.width > room:
                        note.scale_to_fit_width(room)
                    note.next_to([right, y, 0], RIGHT, buff=0.3)
            self.play(FadeIn(notes[0], shift=0.2 * DOWN), run_time=0.8)
            self.play(FadeIn(notes[1], shift=0.2 * UP), run_time=0.8)
            if band == "curly":
                self.caption(f"Tang/Bennett never types {CURLY}, so its rows stay dark here")
            else:
                self.caption(f"And the Chronicle almost never types {STRAIGHT}")
            self.wait(3.5)
            self.play(
                FadeOut(notes),
                self.zoom.animate.set_value(1), self.cx.animate.set_value(mid_x), self.cy.animate.set_value(mid_y),
                run_time=2.5,
            )
            self.play(FadeIn(labels), run_time=0.6)

        self.caption("The spelling split is written right into the data the model learns from")
        self.wait(5)
        end_credits(self)

    def window(self):
        """Visible block range (x0, x1, y0, y1) for the current zoom and center, kept inside the image."""
        z = self.zoom.get_value()
        vw, vh = self.cols / z, self.rows / z
        cx = float(np.clip(self.cx.get_value(), vw / 2, self.cols - vw / 2))
        cy = float(np.clip(self.cy.get_value(), vh / 2, self.rows - vh / 2))
        return cx - vw / 2, cx + vw / 2, cy - vh / 2, cy + vh / 2

    def to_scene(self, col, row):
        x0, x1, y0, y1 = self.window()
        corner = self.box.get_corner(UL)
        return corner + np.array([(col - x0) / (x1 - x0) * self.box.width, -(row - y0) / (y1 - y0) * self.box.height, 0])

    def view(self):
        x0, x1, y0, y1 = self.window()
        pooled = self.zoom.get_value() < POOL_BELOW
        k = 2 if pooled else 1
        pixels = self.pooled if pooled else self.full
        c0, c1 = math.floor(x0 / k), min(math.ceil(x1 / k), pixels.shape[1])
        r0, r1 = math.floor(y0 / k), min(math.ceil(y1 / k), pixels.shape[0])
        image = ImageMobject(np.ascontiguousarray(pixels[r0:r1, c0:c1]))
        image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        ul, dr = self.to_scene(c0 * k, r0 * k), self.to_scene(c1 * k, r1 * k)
        image.stretch_to_fit_width(dr[0] - ul[0]).stretch_to_fit_height(ul[1] - dr[1])
        return image.move_to((ul + dr) / 2)

    def dividers(self):
        opacity = self.show_dividers.get_value()
        lines = VGroup()
        left, right = self.box.get_left()[0], self.box.get_right()[0]
        top, bottom = self.box.get_top()[1], self.box.get_bottom()[1]
        y = self.to_scene(0, self.split)[1]
        if bottom <= y <= top:
            lines.add(DashedLine([left, y, 0], [right, y, 0]))
        for col in (self.curly_col, self.straight_col):
            x = self.to_scene(col, 0)[0]
            if left <= x <= right:
                lines.add(DashedLine([x, top, 0], [x, bottom, 0]))
        return lines.set_stroke(WHITE, width=2, opacity=opacity)


class StarfieldZoomVertical(Portrait, StarfieldZoom):
    """The same scene at 3:4 for Instagram."""
