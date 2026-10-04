"""sparse(transpose(X)): why train_multinomial_nb copies its feature matrix.

Julia stores a sparse matrix column by column (CSC). X has one row per
sentence and one column per trigram, so a sentence's entries are scattered
through that storage. Training visits sentences one at a time, so
train_multinomial_nb first builds Xt = sparse(transpose(X)): a full copy in
which each sentence is one contiguous slice that nzrange(Xt, doc) can walk.

Render from animations/:
    uv run manim -pqh scenes/sparse_transpose.py SparseTranspose          # 16:9
    uv run manim -pqh scenes/sparse_transpose.py SparseTransposeVertical  # 3:4
"""

from dataclasses import dataclass

import numpy as np
from manim import *

from style import DATA_DIR, FONT, MONO, Portrait, end_credits, is_portrait, load_json, text_block

TRIGRAMS = ["k’a", "q’i", "tz’", "ab’", "’ij", "b’a"]
# Toy nonzeros as (doc, trigram, count), 0-based.
ENTRIES = [
    (0, 0, 2), (0, 3, 1), (0, 5, 1),
    (1, 1, 1), (1, 2, 3),
    (2, 0, 1), (2, 4, 2),
    (3, 1, 2), (3, 3, 1), (3, 5, 1),
]
N_DOCS = 4
DOC_COLORS = [BLUE_C, YELLOW_C, GREEN_C, RED_C]


@dataclass
class Layout:
    """Where the matrix and its storage go. The transpose grows the matrix
    downward from grid_corner, so in portrait the storage starts below it."""

    portrait: bool
    cell: float
    grid_corner: np.ndarray  # top-left corner of X
    store_corner: np.ndarray  # top-left corner of the nzval/rowval tapes
    code_x: float  # left edge of the loop code
    text_width: float


def current_layout():
    if is_portrait():
        return Layout(True, cell=0.62, grid_corner=np.array([-1.45, 3.75, 0.0]),
                      store_corner=np.array([-2.5, -1.45, 0.0]), code_x=-3.6, text_width=config.frame_width - 0.8)
    return Layout(False, cell=0.66, grid_corner=np.array([-5.8, 2.1, 0.0]),
                  store_corner=np.array([-0.1, 0.9, 0.0]), code_x=-0.1, text_width=config.frame_width - 1.4)


def count_label(n):
    return Text(str(n), font=MONO, font_size=24, color=BLACK)


class SparseTranspose(Scene):
    def cell_center(self, row, col):
        lay = self.lay
        return lay.grid_corner + (col + 0.5) * lay.cell * RIGHT + (row + 0.5) * lay.cell * DOWN

    def slot_center(self, slot, row):
        lay = self.lay
        return lay.store_corner + (slot + 0.5) * lay.cell * RIGHT + (row + 0.5) * lay.cell * DOWN

    def tile(self, color):
        return RoundedRectangle(
            corner_radius=0.08, width=0.84 * self.lay.cell, height=0.84 * self.lay.cell,
            stroke_width=0, fill_color=color, fill_opacity=0.9,
        )

    def field_label(self, name, y):
        point = np.array([self.lay.store_corner[0], y, 0.0])
        return Text(name, font=MONO, font_size=20, color=GREY_B).next_to(point, LEFT, buff=0.2)

    def construct(self):
        stats = load_json(DATA_DIR / "feature_matrix.json")
        self.lay = lay = current_layout()
        cell_center, slot_center, tile, field_label = self.cell_center, self.slot_center, self.tile, self.field_label
        self.caption_mob = None
        n = len(ENTRIES)
        n_cols = len(TRIGRAMS)

        title = Text("sparse(transpose(X))", font=MONO, font_size=40).to_edge(UP, buff=0.35)
        subtitle = text_block(
            "why Naive Bayes training copies its feature matrix first", 26, lay.text_width, color=GREY_B,
        ).next_to(title, DOWN, buff=0.2)
        self.play(Write(title), FadeIn(subtitle, shift=0.2 * UP), run_time=1.5)
        self.wait(1)

        # The matrix X: one row per sentence, one column per trigram.
        cells = VGroup(*[
            Square(lay.cell, stroke_color=GREY_D, stroke_width=1.5).move_to(cell_center(r, c))
            for r in range(N_DOCS) for c in range(n_cols)
        ])
        matrix_name = Text("X", font=MONO, font_size=30).move_to(lay.grid_corner + np.array([-0.45, 0.4, 0]))
        doc_labels = VGroup(*[
            Text(f"doc {d + 1}", font=FONT, font_size=22, color=DOC_COLORS[d])
            .next_to(cell_center(d, 0), LEFT, buff=0.5 * lay.cell + 0.15)
            for d in range(N_DOCS)
        ])
        tri_labels = VGroup(*[
            Text(t, font=MONO, font_size=19).next_to(cell_center(0, c), UP, buff=0.5 * lay.cell + 0.12)
            for c, t in enumerate(TRIGRAMS)
        ])
        boxes = [tile(DOC_COLORS[d]).move_to(cell_center(d, t)) for d, t, _ in ENTRIES]
        nums = [count_label(k).move_to(cell_center(d, t)) for d, t, k in ENTRIES]
        filled = {(d, t) for d, t, _ in ENTRIES}
        zeros = VGroup(*[
            Text("0", font=MONO, font_size=20, color=GREY_D).move_to(cell_center(r, c))
            for r in range(N_DOCS) for c in range(n_cols) if (r, c) not in filled
        ])

        self.play(FadeOut(subtitle), run_time=0.5)
        self.caption("X: one row per sentence, one column per trigram")
        self.play(Create(cells), FadeIn(matrix_name), FadeIn(doc_labels), FadeIn(tri_labels), run_time=1.5)
        self.play(
            LaggedStart(*[FadeIn(VGroup(b, k), scale=0.6) for b, k in zip(boxes, nums)], lag_ratio=0.08),
            FadeIn(zeros),
            run_time=1.5,
        )
        self.wait(1)
        self.caption("Most entries are 0, so a sparse matrix stores only the rest")
        self.play(FadeOut(zeros), run_time=1)
        self.wait(0.5)

        # CSC storage: nonzeros column by column, each tagged with its row.
        slots = VGroup(*[
            Square(lay.cell, stroke_color=GREY_C, stroke_width=1.5).move_to(slot_center(k, r))
            for r in range(2) for k in range(n)
        ])
        brace_y = lay.store_corner[1] + 0.45
        fields = VGroup(
            field_label("colptr", brace_y),
            field_label("nzval", slot_center(0, 0)[1]),
            field_label("rowval", slot_center(0, 1)[1]),
        )
        store_title_size = 22 if lay.portrait else 24
        store_title = Text("X in memory: column by column (CSC)", font=FONT, font_size=store_title_size, color=GREY_B)
        if lay.portrait:
            store_title.move_to([0, lay.store_corner[1] + 1.25, 0])
        else:
            store_title.move_to(lay.store_corner + np.array([0, 1.25, 0]), aligned_edge=LEFT)

        self.caption("Julia stores the nonzeros one column at a time")
        self.play(FadeIn(slots), FadeIn(fields), FadeIn(store_title), run_time=1)

        csc_order = sorted(range(n), key=lambda e: (ENTRIES[e][1], ENTRIES[e][0]))
        members_of = {c: [e for e in csc_order if ENTRIES[e][1] == c] for c in range(n_cols)}
        store, rowvals = {}, {}
        col_braces = VGroup()
        col_box = None
        slot = 0
        for col, trigram in enumerate(TRIGRAMS):
            column = VGroup(*[cells[r * n_cols + col] for r in range(N_DOCS)])
            target = SurroundingRectangle(column, color=WHITE, buff=0.04)
            anims = [Create(target)] if col_box is None else [Transform(col_box, target)]
            col_box = col_box or target
            first = slot
            for e in members_of[col]:
                d, _, k = ENTRIES[e]
                store[e] = VGroup(tile(DOC_COLORS[d]), count_label(k)).move_to(slot_center(slot, 0))
                rowvals[e] = Text(str(d + 1), font=MONO, font_size=22, color=DOC_COLORS[d]).move_to(slot_center(slot, 1))
                anims += [TransformFromCopy(VGroup(boxes[e], nums[e]), store[e]), FadeIn(rowvals[e])]
                slot += 1
            brace = Brace(slots[first:slot], UP, buff=0.08)
            label = Text(trigram, font=MONO, font_size=18).next_to(brace, UP, buff=0.08)
            col_braces.add(VGroup(brace, label))
            self.play(*anims, GrowFromCenter(brace), FadeIn(label), run_time=0.9)
        self.play(FadeOut(col_box), run_time=0.5)
        self.wait(0.5)

        # Training wants one sentence at a time, which CSC scatters.
        code_corner = np.array([lay.code_x, lay.store_corner[1] - (1.5 if lay.portrait else 2.0), 0])
        loop = Text("for doc_idx in 1:n_docs", font=MONO, font_size=22).move_to(code_corner, aligned_edge=LEFT)
        row_box = SurroundingRectangle(cells[0:n_cols], color=DOC_COLORS[0], buff=0.04)
        self.caption("Training visits one sentence at a time")
        self.play(FadeIn(loop), Create(row_box), run_time=1)
        self.wait(0.5)
        self.caption("Stored by column, doc 1 is scattered: every column must be searched")
        scan, found = None, VGroup()
        slot = 0
        for col in range(n_cols):
            size = len(members_of[col])
            span = VGroup(*slots[slot:slot + size], *slots[n + slot:n + slot + size])
            target = SurroundingRectangle(span, color=WHITE, buff=0.05)
            self.play(Create(target) if scan is None else Transform(scan, target), run_time=0.4)
            scan = scan or target
            hits = [e for e in members_of[col] if ENTRIES[e][0] == 0]
            if hits:
                marks = [
                    SurroundingRectangle(VGroup(store[e], rowvals[e]), color=DOC_COLORS[0], buff=0.03, stroke_width=5)
                    for e in hits
                ]
                found.add(*marks)
                self.play(*[Create(m) for m in marks], run_time=0.3)
            slot += size
        self.play(FadeOut(scan), run_time=0.4)
        self.wait(1)

        # The transpose: flip the matrix across its diagonal...
        self.caption("sparse(transpose(X)) builds a full copy with rows and columns swapped")
        self.play(FadeOut(found), FadeOut(row_box), FadeOut(loop), *[FadeOut(k) for k in nums], run_time=0.8)
        new_doc_labels = VGroup(*[
            Text(f"doc {d + 1}", font=FONT, font_size=16 if lay.cell >= 0.66 else 14, color=DOC_COLORS[d])
            .next_to(cell_center(0, d), UP, buff=0.5 * lay.cell + 0.12)
            for d in range(N_DOCS)
        ])
        new_tri_labels = VGroup(*[
            Text(t, font=MONO, font_size=22).next_to(cell_center(c, 0), LEFT, buff=0.5 * lay.cell + 0.15)
            for c, t in enumerate(TRIGRAMS)
        ])
        diagonal = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
        self.play(
            Rotate(VGroup(cells, *boxes), angle=PI, axis=diagonal, about_point=lay.grid_corner),
            Transform(doc_labels, new_doc_labels),
            Transform(tri_labels, new_tri_labels),
            Transform(matrix_name, Text("Xt", font=MONO, font_size=30).move_to(matrix_name)),
            run_time=2.5,
        )
        t_nums = [count_label(k).move_to(cell_center(t, d)) for d, t, k in ENTRIES]
        self.play(*[FadeIn(k) for k in t_nums], run_time=0.6)
        self.wait(0.5)

        # ...and rebuild the storage so each sentence is contiguous.
        self.caption("Its storage now runs sentence by sentence")
        doc_order = sorted(range(n), key=lambda e: (ENTRIES[e][0], ENTRIES[e][1]))
        new_store_title = Text(
            "Xt in memory: column (= sentence) by column", font=FONT, font_size=store_title_size, color=GREY_B,
        )
        if lay.portrait:
            new_store_title.move_to(store_title)
        else:
            new_store_title.move_to(store_title, aligned_edge=LEFT)
        self.play(FadeOut(col_braces), Transform(store_title, new_store_title), run_time=0.8)
        self.play(*[
            anim
            for k, e in enumerate(doc_order)
            for anim in (
                store[e].animate(path_arc=-PI / 2).move_to(slot_center(k, 0)),
                rowvals[e].animate(path_arc=-PI / 2).move_to(slot_center(k, 1)),
            )
        ], run_time=2.2)
        self.play(*[
            Transform(rowvals[e], Text(TRIGRAMS[ENTRIES[e][1]], font=MONO, font_size=16).move_to(slot_center(k, 1)))
            for k, e in enumerate(doc_order)
        ], run_time=0.8)
        doc_braces = VGroup()
        slot = 0
        for d in range(N_DOCS):
            size = sum(1 for e in ENTRIES if e[0] == d)
            brace = Brace(slots[slot:slot + size], UP, buff=0.08, color=DOC_COLORS[d])
            label = Text(f"doc {d + 1}", font=FONT, font_size=18, color=DOC_COLORS[d]).next_to(brace, UP, buff=0.08)
            doc_braces.add(VGroup(brace, label))
            slot += size
        self.play(LaggedStart(*[AnimationGroup(GrowFromCenter(b), FadeIn(l)) for b, l in doc_braces], lag_ratio=0.25))
        self.wait(0.5)

        # Each sentence is now one slice that nzrange can walk.
        inner = Text("for ptr in nzrange(Xt, doc_idx)", font=MONO, font_size=22)
        code = VGroup(loop.copy(), inner).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(loop, aligned_edge=UL)
        inner.shift(0.5 * RIGHT)
        self.caption("Now each sentence is one contiguous slice")
        self.play(FadeIn(code), run_time=0.8)
        slice_box = column_box = None
        slot = 0
        for d in range(N_DOCS):
            size = sum(1 for e in ENTRIES if e[0] == d)
            span = VGroup(*slots[slot:slot + size], *slots[n + slot:n + slot + size])
            new_slice = SurroundingRectangle(span, color=DOC_COLORS[d], buff=0.05)
            new_column = Rectangle(
                width=lay.cell + 0.08, height=n_cols * lay.cell + 0.08, color=DOC_COLORS[d],
            ).move_to((cell_center(0, d) + cell_center(n_cols - 1, d)) / 2)
            if slice_box is None:
                slice_box, column_box = new_slice, new_column
                self.play(Create(slice_box), Create(column_box), run_time=0.7)
            else:
                self.play(Transform(slice_box, new_slice), Transform(column_box, new_column), run_time=0.6)
            self.wait(0.4)
            slot += size
        self.wait(1)

        # The real matrix.
        self.caption(None)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title], run_time=1)
        n_docs, n_features, nnz = stats["n_docs"], stats["n_features"], stats["nnz"]
        density = stats["density"]
        heading = Text("The real X_train", font=FONT, font_size=36)
        shape = text_block(
            f"{n_docs:,} sentences × {n_features:,} trigrams = {n_docs * n_features:,} cells", 28, lay.text_width,
        )
        bar_width = min(11.0, lay.text_width)
        bar = Rectangle(width=bar_width, height=0.5, stroke_color=GREY_B, stroke_width=2)
        bar_fill = Rectangle(
            width=bar_width * density, height=0.5, stroke_width=0, fill_color=YELLOW_C, fill_opacity=1,
        ).align_to(bar, LEFT)
        bar_label = Text(f"{nnz:,} nonzeros: {100 * density:.2f}% of cells", font=FONT, font_size=26, color=YELLOW_C)
        cost = text_block(
            "sparse(transpose(X)) copies every one of them, on a single thread, "
            "before training, and again for the test matrix before prediction.",
            26, lay.text_width,
        )
        card = VGroup(heading, shape, VGroup(bar, bar_fill), bar_label, cost).arrange(DOWN, buff=0.45)
        card.next_to(title, DOWN, buff=0.6)
        self.play(FadeIn(heading, shift=0.2 * UP), FadeIn(shape, shift=0.2 * UP), run_time=1)
        self.play(Create(bar), run_time=0.8)
        self.play(GrowFromEdge(bar_fill, LEFT), FadeIn(bar_label), run_time=1)
        self.wait(1)
        self.play(FadeIn(cost, shift=0.2 * UP), run_time=1)
        self.wait(4)
        end_credits(self)

    def caption(self, text):
        anims = []
        if self.caption_mob is not None:
            anims.append(FadeOut(self.caption_mob, shift=0.2 * UP))
        self.caption_mob = None
        if text:
            self.caption_mob = text_block(text, 28, self.lay.text_width).to_edge(DOWN, buff=0.45)
            anims.append(FadeIn(self.caption_mob, shift=0.2 * UP))
        self.play(LaggedStart(*anims, lag_ratio=0.5), run_time=0.8)


class SparseTransposeVertical(Portrait, SparseTranspose):
    """The same scene at 3:4 for Instagram."""
