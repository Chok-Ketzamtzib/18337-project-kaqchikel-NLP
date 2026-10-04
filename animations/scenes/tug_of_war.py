"""Every word pulls: the word model's score, one word at a time.

The word-unigram Naive Bayes score is a sum of one push per word. The scene
plays that sum along a number line for two Chronicle test sentences: one the
model gets right after a tug in both directions, and a close call it gets
wrong, which the trigram model, still seeing ’, scores far to the Chronicle side.

Reads animations/data/tug_of_war.json (written by export_data.jl). Only
Chronicle sentences are shown.

Render from animations/:
    uv run manim -pqh scenes/tug_of_war.py TugOfWar          # 16:9, 1920x1080
    uv run manim -pqh scenes/tug_of_war.py TugOfWarVertical  # 3:4, 1080x1440
"""

import numpy as np
from manim import *

from style import (
    APOSTROPHE_STYLE, CLASSICAL, CURLY, DATA_DIR, FONT, MODERN, Captions, Portrait, end_credits, is_portrait,
    load_json, text_width,
)

ERROR = ManimColor("#FF5A5F")
LIMIT = 3.0  # word-score units at each end of the line
TRIGRAM_LIMIT = 400.0


def signed(x, digits=2):
    return f"{x:+.{digits}f}".replace("-", "\u2212")


def line_breaks(words, font_size, max_width):
    """Group word indices into lines no wider than `max_width`."""
    lines, line = [], []
    for i, w in enumerate(words):
        trial = " ".join(words[j] for j in line + [i])
        if line and Text(trial, font=FONT, font_size=font_size).width > max_width:
            lines.append(line)
            line = []
        line.append(i)
    lines.append(line)
    return lines


def word_lines(words, breaks, font_size, **kwargs):
    """One Text per line, so words share a baseline, plus each word's glyphs in reading order."""
    lines = VGroup(*[Text(" ".join(words[i] for i in line), font=FONT, font_size=font_size, **kwargs)
                     for line in breaks]).arrange(DOWN, buff=0.32)
    per_word = VGroup()
    for mob, line in zip(lines, breaks):
        k = 0
        for i in line:
            n = sum(not ch.isspace() for ch in words[i])
            per_word.add(VGroup(*mob[k:k + n]))
            k += n
        assert k == len(mob), (words, line)
    return per_word, lines


class TugOfWar(Captions, Scene):
    def construct(self):
        data = load_json(DATA_DIR / "tug_of_war.json")
        self.portrait = portrait = is_portrait()
        self.half = 3.4 if portrait else 5.8
        self.line_y = -1.3 if portrait else -1.2
        self.sentence_y = 2.7 if portrait else 1.55

        self.title("Every word pulls")
        axis = self.axis(LIMIT, [-3, -2, -1, 0, 1, 2, 3], lambda v: signed(v, 0) if v else "0")
        ends = VGroup(
            Text("Tang/Bennett", font=FONT, font_size=24, color=MODERN)
            .next_to([-self.half, self.line_y, 0], UP, buff=1.2).align_to([-self.half, 0, 0], LEFT),
            Text("Chronicle", font=FONT, font_size=24, color=CLASSICAL)
            .next_to([self.half, self.line_y, 0], UP, buff=1.2).align_to([self.half, 0, 0], RIGHT),
        )

        good = data["correct"][0]
        sentence = self.show_sentence(good, "A Chronicle sentence from the test set")
        self.caption(f"The word model deletes punctuation before counting, {CURLY} included")
        words, sentence = self.to_tokens(sentence, good)
        self.wait(1.5)
        self.caption("Each word adds its weight: right means Chronicle, left Tang/Bennett")
        self.play(FadeIn(axis), FadeIn(ends), run_time=1.2)
        knot, score_label = self.tug(words, good, first_slow=3)
        self.verdict(knot, score_label, good["word_score"], correct=True)
        self.wait(2.5)

        close = data["close_calls"][0]
        self.play(FadeOut(sentence), FadeOut(knot), FadeOut(score_label), run_time=0.8)
        self.remove(words)
        sentence = self.show_sentence(close, "Now a close call, also from the Chronicle")
        words, sentence = self.to_tokens(sentence, close)
        knot, score_label = self.tug(words, close, first_slow=0)
        self.verdict(knot, score_label, close["word_score"], correct=False)
        self.wait(2.5)

        # The trigram model on the same sentence, on a much wider scale.
        n_curly = sum(w.count(CURLY) for w in close["words"])
        self.caption(f"The trigram model still sees its {n_curly} {CURLY}, and does not hesitate")
        wide = self.axis(TRIGRAM_LIMIT, [-400, -200, 0, 200, 400], lambda v: signed(v, 0) if v else "0")
        tri = close["trigram_score"]
        tri_dot = Dot(self.x_at(0, TRIGRAM_LIMIT), radius=0.14, color=CLASSICAL)
        tri_label = Text(f"trigram model {signed(tri, 0)}", font=FONT, font_size=24, color=CLASSICAL)
        self.play(
            ReplacementTransform(axis, wide),
            knot.animate.move_to(self.x_at(close["word_score"], TRIGRAM_LIMIT)),
            FadeOut(score_label),
            run_time=1.5,
        )
        word_tag = Text("word model", font=FONT, font_size=20, color=GREY_B).next_to(knot, DOWN, buff=0.45)
        self.play(FadeIn(word_tag), FadeIn(tri_dot), run_time=0.5)
        self.play(tri_dot.animate.move_to(self.x_at(tri, TRIGRAM_LIMIT)), run_time=1.8, rate_func=rush_from)
        tri_label.next_to(tri_dot, UP, buff=0.3)
        if tri_label.get_right()[0] > config.frame_width / 2 - 0.2:
            tri_label.align_to([config.frame_width / 2 - 0.2, 0, 0], RIGHT)
        self.play(FadeIn(tri_label, shift=0.2 * UP), run_time=0.8)
        self.wait(2)
        self.caption("Words can be a close call. The spelling almost never is")
        self.wait(4.5)
        end_credits(self)

    def x_at(self, score, limit):
        return np.array([np.clip(score / limit, -1.05, 1.05) * self.half, self.line_y, 0])

    def axis(self, limit, ticks, fmt):
        left = Line(self.x_at(-limit, limit), self.x_at(0, limit), stroke_color=MODERN, stroke_width=4)
        right = Line(self.x_at(0, limit), self.x_at(limit, limit), stroke_color=CLASSICAL, stroke_width=4)
        marks = VGroup()
        for v in ticks:
            p = self.x_at(v, limit)
            marks.add(Line(p + 0.1 * DOWN, p + 0.1 * UP, stroke_color=GREY_B, stroke_width=2 if v else 4))
            marks.add(Text(fmt(v), font=FONT, font_size=20, color=GREY_B).next_to(p, DOWN, buff=0.2))
        return VGroup(left, right, marks)

    def show_sentence(self, example, caption):
        self.breaks = line_breaks(example["words"], 34, text_width())
        _, lines = word_lines(example["words"], self.breaks, 34, **APOSTROPHE_STYLE)
        lines.move_to(UP * self.sentence_y)
        self.caption(caption)
        self.play(FadeIn(lines, shift=0.2 * UP), run_time=1.2)
        self.wait(2)
        return lines

    def to_tokens(self, lines, example):
        """Morph the sentence into the tokens the word model counts; returns the new words and lines."""
        tokens = [" ".join(t) for t in example["tokens"]]
        words, token_lines = word_lines(tokens, self.breaks, 34)
        for token_line, line in zip(token_lines, lines):
            token_line.move_to(line)
        self.play(*[ReplacementTransform(a, b) for a, b in zip(lines, token_lines)], run_time=1.5)
        return words, token_lines

    def tug(self, words, example, first_slow):
        score = 0.0
        knot = Dot(self.x_at(0, LIMIT), radius=0.14, color=WHITE)
        label = Text(signed(0.0), font=FONT, font_size=26).next_to(knot, DOWN, buff=0.55)
        self.play(FadeIn(knot, scale=0.5), FadeIn(label), words.animate.set_opacity(0.4), run_time=0.6)
        arrow_y = 0.45
        for i, (m, push) in enumerate(zip(words, example["push"])):
            slow = i < first_slow
            color = CLASSICAL if push > 0 else MODERN
            start, end = self.x_at(score, LIMIT), self.x_at(score + push, LIMIT)
            score += push
            anims = [m.animate.set_opacity(1).set_color(color)]
            arrow = None
            if abs(end[0] - start[0]) > 0.12:
                arrow = Arrow(start + arrow_y * UP, end + arrow_y * UP, buff=0, color=color, stroke_width=6,
                              max_tip_length_to_length_ratio=0.5, tip_length=0.18)
                anims.append(GrowArrow(arrow))
            value = Text(signed(push), font=FONT, font_size=22, color=color)
            value.next_to((start + end) / 2 + arrow_y * UP, UP, buff=0.1)
            self.play(*anims, FadeIn(value), run_time=0.6 if slow else 0.35)
            new_label = Text(signed(score), font=FONT, font_size=26).next_to(end, DOWN, buff=0.55)
            self.play(knot.animate.move_to(end), Transform(label, new_label),
                      *[FadeOut(x) for x in (arrow, value) if x is not None],
                      m.animate.set_opacity(0.55), run_time=0.8 if slow else 0.45)
        return knot, label

    def verdict(self, knot, label, score, correct):
        side = "Chronicle" if score >= 0 else "Tang/Bennett"
        color = CLASSICAL if score >= 0 else MODERN
        if correct:
            text = f"Score {signed(score)}: {side}. Correct"
        else:
            text = f"Score {signed(score)}: {side}. Wrong, narrowly"
        self.caption(text)
        self.play(knot.animate.set_color(color if correct else ERROR).scale(1.3), run_time=0.6)


class TugOfWarVertical(Portrait, TugOfWar):
    """The same scene at 3:4 for Instagram."""
