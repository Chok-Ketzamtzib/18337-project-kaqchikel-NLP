"""Two apostrophes, one sound.

Both corpora write the Kaqchikel glottal stop with an apostrophe, but the
Chronicle types U+2019 and Tang/Bennett types U+0027, and Tang/Bennett has no
punctuation or digits at all. A rule with no training, "say Chronicle if the
sentence has a character Tang/Bennett never uses", nearly matches the trigram
model (exclusive_char_rule in confound_check.jl).

Reads src/results/confound_check.json and src/results/train_metrics.json.
The example sentence is from the Chronicle test split (tug_of_war.json);
no Tang/Bennett text is shown.

Render from animations/:
    uv run manim -pqh scenes/confound_reveal.py ConfoundReveal          # 16:9, 1920x1080
    uv run manim -pqh scenes/confound_reveal.py ConfoundRevealVertical  # 3:4, 1080x1440
"""

from manim import *

from style import (
    APOSTROPHE_STYLE, CLASSICAL, CURLY, DATA_DIR, FONT, MODERN, MONO, RESULTS_DIR, STRAIGHT, Captions, Portrait,
    is_portrait, load_json, text_block,
)

AXIS_FROM = 0.90  # accuracy bars start at 90%, labeled on screen


def percent(x):
    return f"{100 * x:.1f}%"


def glyph_card(char, color, code, name):
    glyph = Text(char, font=MONO, font_size=220, color=color)
    box = Square(side_length=2.2, stroke_color=color, stroke_width=3)
    glyph.move_to(box)
    labels = VGroup(
        Text(code, font=MONO, font_size=26, color=color),
        Text(name, font=FONT, font_size=20, color=GREY_B),
    ).arrange(DOWN, buff=0.1).next_to(box, DOWN, buff=0.2)
    return VGroup(box, glyph, labels)


class ConfoundReveal(Captions, Scene):
    def construct(self):
        confound = load_json(RESULTS_DIR / "confound_check.json")
        train = load_json(RESULTS_DIR / "train_metrics.json")
        chron, tang = confound["orthography"]
        rule = confound["trivial_baselines"]["exclusive_char_rule"]
        regimes = {r["regime"]: r for r in confound["regimes"]}
        example = load_json(DATA_DIR / "tug_of_war.json")["correct"][0]["words"]
        portrait = is_portrait()

        self.title("Two apostrophes, one sound")

        cards = VGroup(
            glyph_card(CURLY, CLASSICAL, "U+2019", "right single quotation mark"),
            glyph_card(STRAIGHT, MODERN, "U+0027", "apostrophe"),
        ).arrange(RIGHT, buff=1.2 if portrait else 2.0).move_to(UP * (0.6 if portrait else 0.1))
        self.caption("Kaqchikel writes its glottal stop with an apostrophe")
        self.play(FadeIn(cards[0], shift=0.3 * UP), run_time=1.2)
        self.wait(1)
        self.caption("But a computer has more than one apostrophe")
        self.play(FadeIn(cards[1], shift=0.3 * UP), run_time=1.2)
        self.wait(2.5)

        # How often each corpus uses each one.
        bar_w = 6.8 if portrait else 9.0
        rows = VGroup()
        for corpus, name, color in ((chron, "Chronicle", CLASSICAL), (tang, "Tang/Bennett", MODERN)):
            n_curly, n_straight = corpus["U+2019"], corpus["U+0027"]
            share = n_curly / (n_curly + n_straight)
            gold = Rectangle(width=max(share * bar_w, 0.001), height=0.55, stroke_width=0,
                             fill_color=CLASSICAL, fill_opacity=0.95)
            blue = Rectangle(width=max((1 - share) * bar_w, 0.001), height=0.55, stroke_width=0,
                             fill_color=MODERN, fill_opacity=0.95)
            bar = VGroup(gold, blue).arrange(RIGHT, buff=0)
            label = Text(name, font=FONT, font_size=26, color=color)
            counts = Text(f"{n_curly:,} {CURLY}    {n_straight:,} {STRAIGHT}", font=FONT, font_size=22,
                          color=GREY_A, **APOSTROPHE_STYLE)
            row = VGroup(label, bar, counts).arrange(DOWN, buff=0.15)
            row.bar = bar
            rows.add(row)
        rows.arrange(DOWN, buff=0.6).move_to(UP * (0.4 if portrait else -0.2))
        for row in rows:
            row[0].align_to(row.bar, LEFT)
            row[2].align_to(row.bar, LEFT)
        small_cards = cards.copy().scale(0.3).arrange(RIGHT, buff=0.4).next_to(rows, UP, buff=0.5)
        self.caption(f"The Chronicle types {CURLY} {percent(chron['curly_share_of_apostrophes'])} of the time")
        self.play(Transform(cards, small_cards), FadeIn(rows[0][0]), GrowFromEdge(rows[0].bar, LEFT),
                  FadeIn(rows[0][2]), run_time=1.8)
        self.wait(2)
        self.caption(f"Tang/Bennett: never. Every one of its {tang['U+0027']:,} is {STRAIGHT}")
        self.play(FadeIn(rows[1][0]), GrowFromEdge(rows[1].bar, LEFT), FadeIn(rows[1][2]), run_time=1.8)
        self.wait(3)

        # Punctuation and digits.
        stats = VGroup()
        for label, key in (("commas", "comma"), ("digits", "digits"), ("all punctuation", "total_punct")):
            stats.add(VGroup(
                Text(label, font=FONT, font_size=24, color=GREY_B),
                Text(f"{chron[key]:,}", font=FONT, font_size=40, weight=BOLD, color=CLASSICAL),
                Text(f"{tang[key]:,}", font=FONT, font_size=40, weight=BOLD, color=MODERN),
            ).arrange(DOWN, buff=0.25))
        stats.arrange(RIGHT, buff=0.9 if portrait else 1.6).move_to(UP * (0.2 if portrait else -0.3))
        heads = VGroup(
            Text("Chronicle", font=FONT, font_size=22, color=CLASSICAL).next_to(stats[0][1], LEFT, buff=0.6),
            Text("Tang/Bennett", font=FONT, font_size=22, color=MODERN).next_to(stats[0][2], LEFT, buff=0.6),
        )
        table = VGroup(stats, heads).move_to(UP * (0.2 if portrait else -0.3))
        if table.width > config.frame_width - 0.6:
            table.scale_to_fit_width(config.frame_width - 0.6)
        self.caption("Tang/Bennett has no punctuation or digits at all")
        self.play(FadeOut(cards), FadeOut(rows), FadeIn(table, shift=0.2 * UP), run_time=1.5)
        self.wait(3)

        # Characters only the Chronicle uses.
        only = rule["classical_only_chars"]
        per_row = 8 if portrait else 11
        # One Text per row keeps every glyph on its baseline, so , and ’ stay distinguishable.
        rows = [Text("  ".join(only[i:i + per_row]), font=MONO, font_size=30, t2c={CURLY: CLASSICAL})
                for i in range(0, len(only), per_row)]
        for i, row in enumerate(rows):
            row.move_to([0, -0.76 * i, 0]).align_to(rows[0], LEFT)
        chips = VGroup(*[
            VGroup(Square(side_length=0.62, stroke_color=CLASSICAL, stroke_width=2).move_to([g.get_x(), row.get_y(), 0]), g)
            for row in rows for g in row
        ])
        assert len(chips) == len(only)
        chips.move_to(UP * (1.6 if portrait else 1.1))
        self.caption(f"{len(only)} characters appear in Chronicle training text and never in Tang/Bennett text")
        self.play(FadeOut(table), LaggedStart(*[FadeIn(c, scale=0.5) for c in chips], lag_ratio=0.05), run_time=2)
        self.wait(2.5)

        # The rule on a real Chronicle test sentence.
        sentence = text_block(" ".join(example), 32, config.frame_width - 1.2, buff=0.3, **APOSTROPHE_STYLE)
        sentence.next_to(chips, DOWN, buff=0.8)
        marks = VGroup()
        for line_mob, line in zip(sentence, sentence.lines):
            glyphs = [ch for ch in line if not ch.isspace()]
            assert len(glyphs) == len(line_mob), (line, len(line_mob))
            for glyph, ch in zip(line_mob, glyphs):
                if ch in only:
                    marks.add(SurroundingRectangle(glyph, buff=0.05, stroke_color=CLASSICAL, stroke_width=3))
        verdict = Text("has one of them, so: Chronicle", font=FONT, font_size=26, color=CLASSICAL)
        verdict.next_to(sentence, DOWN, buff=0.4)
        self.caption("A rule with no training: if a sentence has any of them, say Chronicle")
        self.play(FadeIn(sentence, shift=0.2 * UP), run_time=1.2)
        self.play(LaggedStart(*[GrowFromCenter(m) for m in marks], lag_ratio=0.1), run_time=1.2)
        self.play(FadeIn(verdict), run_time=0.8)
        self.wait(2.5)

        # Scoreboard.
        entries = (
            ("word model", train["word_unigram"]["test_accuracy_serial"], GREY_B),
            ("the one-line rule", rule["test_accuracy"], CLASSICAL),
            ("trigram model", regimes["A_raw"]["test_accuracy"], WHITE),
        )
        full = 6.0 if portrait else 8.0
        board = VGroup()
        for name, acc, color in entries:
            bar = Rectangle(width=(acc - AXIS_FROM) / (1 - AXIS_FROM) * full, height=0.5, stroke_width=0,
                            fill_color=color, fill_opacity=0.9)
            label = Text(name, font=FONT, font_size=24, color=color)
            value = Text(percent(acc), font=FONT, font_size=26, weight=BOLD)
            board.add(VGroup(label, bar, value))
        for i, (label, bar, value) in enumerate(board):
            bar.move_to([-full / 2, -i * 1.1, 0], aligned_edge=LEFT)
            label.next_to(bar, UP, buff=0.1, aligned_edge=LEFT)
            value.next_to(bar, RIGHT, buff=0.15)
        ends = VGroup(
            Line([-full / 2, 0.55, 0], [-full / 2, -2.55, 0], stroke_color=GREY_C, stroke_width=2),
            Line([full / 2, 0.55, 0], [full / 2, -2.55, 0], stroke_color=GREY_D, stroke_width=1.5),
        )
        ticks = VGroup(
            Text(percent(AXIS_FROM)[:-3] + "%", font=FONT, font_size=18, color=GREY_B).next_to(ends[0], DOWN, buff=0.1),
            Text("100%", font=FONT, font_size=18, color=GREY_B).next_to(ends[1], DOWN, buff=0.1),
        )
        scoreboard = VGroup(ends, ticks, board).move_to(UP * (0.2 if portrait else -0.3))
        self.caption(f"That rule gets {percent(rule['test_accuracy'])} of the test set right. "
                     "The word model: " + percent(train["word_unigram"]["test_accuracy_serial"]))
        self.play(FadeOut(chips), FadeOut(sentence), FadeOut(marks), FadeOut(verdict),
                  FadeIn(ends), FadeIn(ticks), run_time=1)
        for label, bar, value in board:
            self.play(FadeIn(label), GrowFromEdge(bar, LEFT), FadeIn(value), run_time=1)
        self.wait(2.5)
        self.caption("The trigram model mostly learned how each corpus was typed")
        self.wait(4.5)


class ConfoundRevealVertical(Portrait, ConfoundReveal):
    """The same scene at 3:4 for Instagram."""
