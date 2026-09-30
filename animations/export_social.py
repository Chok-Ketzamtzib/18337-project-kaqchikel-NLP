"""Builds animations/social/: every video formatted for each placement, in posting order.

For each post in social_posts.toml, writes NN_<slug>/ with
    instagram_post_3x4.mp4       1080x1440, the portrait render
    instagram_reel_9x16.mp4      1080x1920, the portrait render on a taller black canvas
    linkedin_post_16x9.mp4       1920x1080, the landscape render
    linkedin_vertical_9x16.mp4   1080x1920, the same file as the Reel
    cover_*.png                  one cover per aspect ratio
    caption_instagram.txt, caption_linkedin.txt
    preview.png                  the four placements, mocked up on phones
and, for the whole series, schedule.md, schedule.csv, and mockups/.

The mockups are drawn with PIL and only approximate each app's layout.

Usage, from animations/ after rendering every scene at -qh:
    uv run python export_social.py
    uv run python export_social.py --no-video   # everything except the video files
"""

import argparse
import csv
import datetime as dt
import shutil
import tomllib
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
VIDEOS = HERE / "media" / "videos"
OUT = HERE / "social"
FONTS = Path("C:/Windows/Fonts")

REEL_W, REEL_H = 1080, 1920
# The 3:4 frame sits 100 px from the top of the Reel. The scenes' captions then
# end 380 px above the bottom, clear of the handle and caption the app overlays there.
REEL_TOP = 100

GOLD, BLUE = (242, 184, 75), (79, 184, 230)
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def font(size, weight="regular"):
    name = {"regular": "segoeui.ttf", "semibold": "seguisb.ttf", "bold": "segoeuib.ttf",
            "symbol": "seguisym.ttf"}[weight]
    return ImageFont.truetype(str(FONTS / name), size)


# ---------------------------------------------------------------- videos


def renders(scene, slug):
    return (VIDEOS / slug / "1440p60" / f"{scene}Vertical.mp4", VIDEOS / slug / "1080p60" / f"{scene}.mp4")


def pad_to_reel(src, dst):
    """Re-encode a 1080x1440 yuv420p video onto a black 1080x1920 canvas, plane by plane."""
    with av.open(str(src)) as inp, av.open(str(dst), "w", options={"movflags": "+faststart"}) as out:
        vin = inp.streams.video[0]
        rate = Fraction(vin.average_rate).limit_denominator(1001)
        vout = out.add_stream("libx264", rate=rate)
        vout.width, vout.height, vout.pix_fmt = REEL_W, REEL_H, "yuv420p"
        vout.options = {"crf": "18", "preset": "medium", "profile": "high"}
        black = None
        for i, frame in enumerate(inp.decode(vin)):
            assert frame.format.name == "yuv420p" and (frame.width, frame.height) == (1080, 1440), frame
            planes = frame.to_ndarray()  # Y (1440 rows) then U and V (360 rows of 1080 each)
            y, u, v = planes[:1440], planes[1440:1800].reshape(720, 540), planes[1800:].reshape(720, 540)
            if black is None:
                black = (int(y[0, 0]), int(u[0, 0]), int(v[0, 0]))
            canvas_y = np.full((REEL_H, REEL_W), black[0], np.uint8)
            canvas_u = np.full((REEL_H // 2, REEL_W // 2), black[1], np.uint8)
            canvas_v = np.full((REEL_H // 2, REEL_W // 2), black[2], np.uint8)
            canvas_y[REEL_TOP:REEL_TOP + 1440] = y
            canvas_u[REEL_TOP // 2:REEL_TOP // 2 + 720] = u
            canvas_v[REEL_TOP // 2:REEL_TOP // 2 + 720] = v
            stacked = np.concatenate([canvas_y, canvas_u.reshape(-1, REEL_W), canvas_v.reshape(-1, REEL_W)])
            new = av.VideoFrame.from_ndarray(stacked, format="yuv420p")
            new.pts, new.time_base = i, 1 / rate
            for packet in vout.encode(new):
                out.mux(packet)
        for packet in vout.encode():
            out.mux(packet)


def frame_at(path, seconds):
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        c.seek(int(seconds / stream.time_base), stream=stream)
        for frame in c.decode(stream):
            if frame.time >= seconds - 1 / 120:
                return frame.to_image()
    raise ValueError(f"{path} has no frame at {seconds}s")


def duration(path):
    with av.open(str(path)) as c:
        return float(c.duration) / av.time_base


def reel_frame(image_3x4):
    canvas = Image.new("RGB", (REEL_W, REEL_H), (0, 0, 0))
    canvas.paste(image_3x4, (0, REEL_TOP))
    return canvas


# ---------------------------------------------------------------- covers


def wrap(draw, text, fnt, max_w):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and draw.textlength(trial, font=fnt) > max_w:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line]


def balanced_wrap(draw, text, fnt, max_w):
    """As few lines as greedy wrapping needs, but with their lengths evened out."""
    n = len(wrap(draw, text, fnt, max_w))
    width = draw.textlength(text, font=fnt) / n
    while len(lines := wrap(draw, text, fnt, width)) > n:
        width += 8
    return lines


def cover(frame, title, part, total, series, center_y=None):
    """The frame blurred and dimmed behind a dark panel carrying the post title."""
    img = frame.convert("RGB").filter(ImageFilter.GaussianBlur(4))
    img = Image.blend(img, Image.new("RGB", img.size, (0, 0, 0)), 0.62)
    w, h = img.size
    size = int(min(w, h) * 0.088)
    probe = ImageDraw.Draw(img)
    title_font, tag_font = font(size, "bold"), font(int(size * 0.4), "semibold")
    lines = balanced_wrap(probe, title, title_font, w * 0.78)
    line_h = int(size * 1.2)
    block_h = int(size * 0.85) + line_h * len(lines) + 30
    block_w = max(probe.textlength(line, font=title_font) for line in lines)
    cy = center_y if center_y is not None else h / 2
    y = int(cy - block_h / 2)

    panel = Image.new("RGBA", img.size, (0, 0, 0, 0))
    pad = int(size * 0.55)
    ImageDraw.Draw(panel).rounded_rectangle(
        [w / 2 - block_w / 2 - pad, y - pad, w / 2 + block_w / 2 + pad, y + block_h + pad * 0.7],
        radius=int(size * 0.35), fill=(8, 8, 8, 225), outline=(70, 70, 70, 255), width=2)
    img = Image.alpha_composite(img.convert("RGBA"), panel).convert("RGB")

    draw = ImageDraw.Draw(img)
    draw.text((w / 2, y), f"{series.upper()}  ·  {part}/{total}", font=tag_font, fill=GOLD, anchor="mt")
    y += int(size * 0.85)
    for line in lines:
        draw.text((w / 2, y), line, font=title_font, fill=(255, 255, 255), anchor="mt")
        y += line_h
    bar_w = int(size * 1.2)
    draw.rectangle([w / 2 - bar_w, y + 12, w / 2, y + 19], fill=GOLD)
    draw.rectangle([w / 2, y + 12, w / 2 + bar_w, y + 19], fill=BLUE)
    return img


# ---------------------------------------------------------------- mockups

PHONE_W, PHONE_H = 480, 1000
SCREEN_PAD = 14


def phone(screen):
    """Put a screen image (PHONE_W - 2*pad wide) inside a phone outline."""
    body = Image.new("RGBA", (PHONE_W, PHONE_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(body)
    d.rounded_rectangle([0, 0, PHONE_W - 1, PHONE_H - 1], radius=64, fill=(12, 12, 12), outline=(70, 70, 70), width=3)
    mask = Image.new("L", screen.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, screen.width - 1, screen.height - 1], radius=52, fill=255)
    body.paste(screen, (SCREEN_PAD, SCREEN_PAD), mask)
    d.rounded_rectangle([PHONE_W / 2 - 55, 26, PHONE_W / 2 + 55, 52], radius=13, fill=(0, 0, 0))
    return body


def screen_size():
    return PHONE_W - 2 * SCREEN_PAD, PHONE_H - 2 * SCREEN_PAD


def status_bar(d, w, color):
    d.text((34, 22), "9:41", font=font(20, "semibold"), fill=color)
    d.rounded_rectangle([w - 62, 26, w - 30, 40], radius=4, outline=color, width=2)


def avatar(d, xy, r, ring=None):
    x, y = xy
    if ring:
        d.ellipse([x - r - 4, y - r - 4, x + r + 4, y + r + 4], fill=ring)
    d.ellipse([x - r, y - r, x + r, y + r], fill=(60, 60, 60), outline=(0, 0, 0), width=2)
    d.text((x, y), "K", font=font(int(r * 1.1), "bold"), fill=GOLD, anchor="mm")


def fit(draw, text, fnt, max_w):
    """The longest run of whole words from the start of `text` that fits in `max_w`."""
    out = ""
    for word in text.split():
        trial = f"{out} {word}".strip()
        if draw.textlength(trial, font=fnt) > max_w:
            break
        out = trial
    return out


def first_line(text, draw, fnt, max_w):
    """The caption's opening words, cut to fit one line with a trailing ellipsis."""
    flat = " ".join(text.split())
    line = fit(draw, flat, fnt, max_w - draw.textlength(" …", font=fnt))
    return line if line == flat else line + " …"


def ig_feed_screen(media_3x4, handle, caption):
    w, h = screen_size()
    s = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(s)
    status_bar(d, w, (255, 255, 255))
    d.text((22, 62), "Instagram", font=font(30, "semibold"), fill=(255, 255, 255))
    avatar(d, (40, 134), 17, ring=(214, 41, 118))
    d.text((68, 134), handle, font=font(18, "semibold"), fill=(255, 255, 255), anchor="lm")
    d.text((w - 24, 134), "···", font=font(20, "bold"), fill=(255, 255, 255), anchor="rm")
    media = media_3x4.resize((w, int(w * 4 / 3)))
    s.paste(media, (0, 164))
    y = 164 + media.height + 16
    icons = font(28, "symbol")
    for i, glyph in enumerate(("\u2661", "\u25cb", "\u27a4")):
        d.text((22 + i * 46, y), glyph, font=icons, fill=(255, 255, 255))
    d.text((w - 22, y), "\u2690", font=icons, fill=(255, 255, 255), anchor="ra")
    y += 46
    body = font(17)
    line = first_line(caption, d, body, w - 44 - d.textlength(handle + " ", font=font(17, "semibold")))
    d.text((22, y), handle, font=font(17, "semibold"), fill=(255, 255, 255))
    d.text((22 + d.textlength(handle + " ", font=font(17, "semibold")), y), line, font=body, fill=(230, 230, 230))
    d.text((22, y + 26), "more", font=body, fill=(150, 150, 150))
    nav_bar(d, w, h, (255, 255, 255), (0, 0, 0))
    return s


def nav_bar(d, w, h, fg, bg):
    d.rectangle([0, h - 70, w, h], fill=bg)
    for i in range(5):
        x = w * (i + 0.5) / 5
        d.rounded_rectangle([x - 12, h - 52, x + 12, h - 28], radius=6, outline=fg, width=2)


def ig_reel_screen(reel_9x16, handle, caption):
    w, h = screen_size()
    s = Image.new("RGB", (w, h), (0, 0, 0))
    media = reel_9x16.resize((w, int(w * 16 / 9)))
    top = 52
    s.paste(media, (0, top))
    d = ImageDraw.Draw(s)
    status_bar(d, w, (255, 255, 255))
    d.text((22, top + 18), "Reels", font=font(26, "bold"), fill=(255, 255, 255))
    icons, small = font(30, "symbol"), font(15, "semibold")
    y = top + media.height - 330
    for glyph, label in (("\u2661", "1,204"), ("\u25cb", "86"), ("\u27a4", "310"), ("\u2026", "")):
        d.text((w - 36, y), glyph, font=icons, fill=(255, 255, 255), anchor="mm")
        if label:
            d.text((w - 36, y + 28), label, font=small, fill=(255, 255, 255), anchor="mm")
        y += 72
    base = top + media.height - 118
    avatar(d, (38, base), 16)
    d.text((64, base), handle, font=font(17, "semibold"), fill=(255, 255, 255), anchor="lm")
    fx = 64 + d.textlength(handle, font=font(17, "semibold")) + 14
    d.rounded_rectangle([fx, base - 14, fx + 70, base + 14], radius=8, outline=(255, 255, 255), width=2)
    d.text((fx + 35, base), "Follow", font=font(15, "semibold"), fill=(255, 255, 255), anchor="mm")
    body = font(16)
    d.text((22, base + 26), first_line(caption, d, body, w - 110), font=body, fill=(240, 240, 240))
    nav_bar(d, w, h, (255, 255, 255), (0, 0, 0))
    return s


def li_feed_screen(media_16x9, name, headline, caption):
    w, h = screen_size()
    s = Image.new("RGB", (w, h), (244, 242, 238))
    d = ImageDraw.Draw(s)
    status_bar(d, w, (0, 0, 0))
    d.rectangle([0, 50, w, 110], fill=(255, 255, 255))
    avatar(d, (34, 80), 16)
    d.rounded_rectangle([64, 64, w - 70, 96], radius=6, fill=(237, 243, 248))
    d.text((80, 80), "Search", font=font(17), fill=(90, 90, 90), anchor="lm")
    top = 122
    d.rectangle([0, top, w, h], fill=(255, 255, 255))
    avatar(d, (42, top + 40), 24)
    d.text((78, top + 16), name, font=font(18, "semibold"), fill=(0, 0, 0))
    d.text((78, top + 40), headline, font=font(14), fill=(100, 100, 100))
    d.text((78, top + 60), "1d · Edited", font=font(14), fill=(100, 100, 100))
    body = font(16)
    y = top + 96
    lines = wrap(d, " ".join(caption.split()), body, w - 40)
    for i, line in enumerate(lines[:3]):
        if i == 2 and len(lines) > 3:
            line = fit(d, line, body, w - 40 - d.textlength(" …more", font=body))
            d.text((20 + d.textlength(line, font=body), y), " …more", font=body, fill=(100, 100, 100))
        d.text((20, y), line, font=body, fill=(30, 30, 30))
        y += 24
    y += 12
    media = media_16x9.resize((w, int(w * 9 / 16)))
    s.paste(media, (0, y))
    y += media.height + 12
    d.text((20, y), "\u25cf\u25cf 128 · 14 comments · 9 reposts", font=font(14), fill=(100, 100, 100))
    y += 32
    d.line([20, y, w - 20, y], fill=(225, 225, 225), width=1)
    for i, label in enumerate(("Like", "Comment", "Repost", "Send")):
        d.text((w * (i + 0.5) / 4, y + 26), label, font=font(15, "semibold"), fill=(90, 90, 90), anchor="mm")
    nav_bar(d, w, h, (90, 90, 90), (255, 255, 255))
    return s


def li_vertical_screen(reel_9x16, name, headline, caption):
    w, h = screen_size()
    s = Image.new("RGB", (w, h), (0, 0, 0))
    media = reel_9x16.resize((w, int(w * 16 / 9)))
    top = 60
    s.paste(media, (0, top))
    d = ImageDraw.Draw(s)
    status_bar(d, w, (255, 255, 255))
    d.text((24, top + 16), "\u2715", font=font(24, "symbol"), fill=(255, 255, 255))
    y = top + media.height - 300
    for label in ("Like", "Comment", "Repost", "Send"):
        d.ellipse([w - 58, y - 20, w - 18, y + 20], outline=(255, 255, 255), width=2)
        d.text((w - 38, y + 34), label, font=font(13, "semibold"), fill=(255, 255, 255), anchor="mm")
        y += 72
    base = top + media.height - 110
    avatar(d, (40, base), 18)
    d.text((68, base - 10), name, font=font(17, "semibold"), fill=(255, 255, 255), anchor="lm")
    d.text((68, base + 12), headline, font=font(13), fill=(210, 210, 210), anchor="lm")
    body = font(15)
    d.text((22, base + 34), first_line(caption, d, body, w - 110), font=body, fill=(240, 240, 240))
    nav_bar(d, w, h, (255, 255, 255), (0, 0, 0))
    return s


def labeled(panels, labels, bg=(30, 30, 30), title=None):
    gap, top = 56, 150 if title else 90
    width = sum(p.width for p in panels) + gap * (len(panels) + 1)
    height = max(p.height for p in panels) + top + 50
    sheet = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(sheet)
    if title:
        d.text((gap, 40), title, font=font(40, "bold"), fill=(255, 255, 255))
    x = gap
    for panel, label in zip(panels, labels):
        d.text((x + panel.width / 2, top - 30), label, font=font(24, "semibold"), fill=(220, 220, 220), anchor="mm")
        sheet.paste(panel, (x, top), panel if panel.mode == "RGBA" else None)
        x += panel.width + gap
    return sheet


def ig_profile_screens(series, posts_newest_first):
    w, h = screen_size()
    screens = []
    for tab in ("grid", "reels"):
        s = Image.new("RGB", (w, h), (0, 0, 0))
        d = ImageDraw.Draw(s)
        status_bar(d, w, (255, 255, 255))
        d.text((22, 62), series["handle"], font=font(24, "bold"), fill=(255, 255, 255))
        avatar(d, (66, 160), 44, ring=(214, 41, 118))
        for i, (n, label) in enumerate(((len(posts_newest_first), "posts"), ("1,024", "followers"), ("180", "following"))):
            x = 190 + i * 88
            d.text((x, 146), str(n), font=font(20, "bold"), fill=(255, 255, 255), anchor="mm")
            d.text((x, 172), label, font=font(14), fill=(220, 220, 220), anchor="mm")
        d.text((22, 222), series["display_name"], font=font(17, "semibold"), fill=(255, 255, 255))
        d.text((22, 246), series["name"], font=font(16), fill=(220, 220, 220))
        d.text((22, 268), series["headline"], font=font(16), fill=(220, 220, 220))
        tab_y = 316
        for i in range(3):
            x = w * (i + 0.5) / 3
            active = (tab == "grid" and i == 0) or (tab == "reels" and i == 1)
            d.rounded_rectangle([x - 11, tab_y - 11, x + 11, tab_y + 11], radius=4,
                                outline=(255, 255, 255) if active else (120, 120, 120), width=2)
            if active:
                d.line([x - w / 6, tab_y + 26, x + w / 6, tab_y + 26], fill=(255, 255, 255), width=2)
        grid_top = tab_y + 30
        tile_w = (w - 4) // 3
        tile_h = int(tile_w * (4 / 3 if tab == "grid" else 16 / 9))
        for k, post in enumerate(posts_newest_first):
            r, c = divmod(k, 3)
            y = grid_top + r * (tile_h + 2)
            if y >= h - 70:
                break
            img = post["cover_3x4"] if tab == "grid" else post["cover_9x16"]
            tile = img.resize((tile_w, tile_h))
            s.paste(tile.crop((0, 0, tile_w, min(tile_h, h - 70 - y))), (c * (tile_w + 2), y))
        nav_bar(d, w, h, (255, 255, 255), (0, 0, 0))
        screens.append(s)
    return screens


def li_activity(series, posts):
    """A desktop-width list of the LinkedIn posts, oldest first."""
    w = 1100
    row_h = 200
    sheet = Image.new("RGB", (w, 120 + row_h * len(posts) + 40), (244, 242, 238))
    d = ImageDraw.Draw(sheet)
    d.text((40, 40), f"{series['display_name']} · LinkedIn activity, in posting order", font=font(30, "bold"),
           fill=(0, 0, 0))
    for i, post in enumerate(posts):
        y = 110 + i * row_h
        d.rounded_rectangle([30, y, w - 30, y + row_h - 16], radius=10, fill=(255, 255, 255), outline=(220, 220, 220))
        thumb = post["cover_16x9"].resize((300, 169))
        sheet.paste(thumb, (46, y + 8))
        d.text((370, y + 20), f"{post['date']:%a %b %d} · {series['linkedin_time']}", font=font(18, "semibold"),
               fill=(10, 102, 194))
        d.text((370, y + 48), f"{i + 1}/{len(posts)}  {post['title']}", font=font(24, "bold"), fill=(0, 0, 0))
        for k, line in enumerate(wrap(d, " ".join(post["linkedin"].split()), font(17), w - 440)[:3]):
            d.text((370, y + 86 + k * 24), line, font=font(17), fill=(60, 60, 60))
    return sheet


def timeline(series, posts):
    """One row per post in order: date, title, and a thumbnail per placement."""
    row_h, thumb_h = 250, 210
    cols = [("Instagram post", 3 / 4, "cover_3x4"), ("Instagram Reel", 9 / 16, "cover_9x16"),
            ("LinkedIn post", 16 / 9, "cover_16x9"), ("LinkedIn vertical", 9 / 16, "cover_9x16")]
    text_w, gap = 520, 36
    width = text_w + sum(int(thumb_h * a) for _, a, _ in cols) + gap * (len(cols) + 1)
    sheet = Image.new("RGB", (width, 170 + row_h * len(posts)), (22, 22, 22))
    d = ImageDraw.Draw(sheet)
    d.text((40, 36), f"{series['name']}: posting order", font=font(38, "bold"), fill=(255, 255, 255))
    x = text_w + gap
    for label, aspect, _ in cols:
        tw = int(thumb_h * aspect)
        d.text((x + tw / 2, 125), label, font=font(20, "semibold"), fill=(200, 200, 200), anchor="mm")
        x += tw + gap
    for i, post in enumerate(posts):
        y = 160 + i * row_h
        d.line([40, y - 10, width - 40, y - 10], fill=(60, 60, 60), width=1)
        d.text((40, y + 20), f"{i + 1}", font=font(54, "bold"), fill=GOLD)
        d.text((120, y + 26), f"{post['date']:%a %b %d, %Y}", font=font(20, "semibold"), fill=(170, 170, 170))
        for k, line in enumerate(wrap(d, post["title"], font(28, "bold"), text_w - 130)[:2]):
            d.text((120, y + 56 + k * 36), line, font=font(28, "bold"), fill=(255, 255, 255))
        d.text((120, y + 140), f"Instagram {series['instagram_time']} · LinkedIn {series['linkedin_time']}",
               font=font(17), fill=(150, 150, 150))
        d.text((120, y + 166), f"{post['duration']:.0f} s · {post['folder']}/", font=font(17), fill=(150, 150, 150))
        x = text_w + gap
        for _, aspect, key in cols:
            tw = int(thumb_h * aspect)
            sheet.paste(post[key].resize((tw, thumb_h)), (x, y + 10))
            x += tw + gap
    return sheet


# ---------------------------------------------------------------- schedule


def long_date(day):
    return f"{day:%B} {day.day}, {day.year}"


def posting_days(first, weekdays, n):
    wanted = {WEEKDAYS.index(w) for w in weekdays}
    days, day = [], first
    while len(days) < n:
        if day.weekday() in wanted:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


def caption_files(folder, post, part, total, series):
    footer = f"Part {part} of {total} · MIT 18.337 · made with Julia and Manim"
    (folder / "caption_instagram.txt").write_text(
        f"{post['instagram'].strip()}\n\n{footer}\n\n{series['instagram_tags']}\n", encoding="utf-8")
    (folder / "caption_linkedin.txt").write_text(
        f"{post['linkedin'].strip()}\n\n{footer}\n\n{series['linkedin_tags']}\n", encoding="utf-8")


SPECS = """
## Formats

| Placement | File | Size | Notes |
|---|---|---|---|
| Instagram feed post | `instagram_post_3x4.mp4` | 1080×1440 (3:4) | Fills the 3:4 profile grid tile without cropping. |
| Instagram Reel | `instagram_reel_9x16.mp4` | 1080×1920 (9:16) | The 3:4 video on a black canvas, 100 px from the top, so the scene's own captions end above the handle and caption the app draws at the bottom. |
| LinkedIn feed post | `linkedin_post_16x9.mp4` | 1920×1080 (16:9) | The landscape render; plays inline in the feed. |
| LinkedIn vertical video | `linkedin_vertical_9x16.mp4` | 1080×1920 (9:16) | Same file as the Reel, for LinkedIn's full-screen vertical video viewer. |

All files are H.264, 60 fps, with no audio track. Adding a quiet track in each app may help reach.

Instagram publishes every video as a Reel, so uploading both Instagram files of one video puts
it on your profile twice; the same goes for the two LinkedIn files. On each platform, pick one
file per post. On Instagram, use the 3:4 post if the grid look matters most, or the 9:16 Reel
(with its grid crop set to the middle) if reach in the Reels tab does. Each Reel cover keeps its
title inside that middle 3:4.

Covers: upload `cover_*.png` as the video's thumbnail when posting. Platform limits and
recommended sizes change; check each app's current guidance before posting.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-video", action="store_true", help="skip copying and re-encoding video files")
    args = parser.parse_args()

    plan = tomllib.loads((HERE / "social_posts.toml").read_text(encoding="utf-8"))
    series, posts = plan["series"], plan["post"]
    total = len(posts)
    days = posting_days(series["first_day"], series["weekdays"], total)
    OUT.mkdir(exist_ok=True)
    (OUT / "mockups").mkdir(exist_ok=True)

    for part, (post, day) in enumerate(zip(posts, days), start=1):
        portrait, landscape = renders(post["scene"], post["slug"])
        for path in (portrait, landscape):
            if not path.exists():
                raise SystemExit(f"missing render: {path} (render it with manim -qh first)")
        folder = OUT / f"{part:02d}_{post['slug']}"
        folder.mkdir(exist_ok=True)
        post.update(date=day, folder=folder.name, duration=duration(landscape))
        print(f"{folder.name}: {day:%a %Y-%m-%d}, {post['duration']:.1f} s")

        if not args.no_video:
            shutil.copyfile(portrait, folder / "instagram_post_3x4.mp4")
            shutil.copyfile(landscape, folder / "linkedin_post_16x9.mp4")
            reel = folder / "instagram_reel_9x16.mp4"
            pad_to_reel(portrait, reel)
            shutil.copyfile(reel, folder / "linkedin_vertical_9x16.mp4")

        t = post["cover_at"]
        tall, wide = frame_at(portrait, t), frame_at(landscape, t)
        post["cover_3x4"] = cover(tall, post["title"], part, total, series["name"])
        post["cover_9x16"] = cover(reel_frame(tall), post["title"], part, total, series["name"],
                                   center_y=REEL_TOP + 720)
        post["cover_16x9"] = cover(wide, post["title"], part, total, series["name"])
        post["cover_3x4"].save(folder / "cover_instagram_post_3x4.png")
        post["cover_9x16"].save(folder / "cover_reel_9x16.png")
        post["cover_16x9"].save(folder / "cover_linkedin_16x9.png")
        caption_files(folder, post, part, total, series)

        # What each placement looks like mid-video.
        mid_tall, mid_wide = frame_at(portrait, post["duration"] * 0.45), frame_at(landscape, post["duration"] * 0.45)
        panels = [
            phone(ig_feed_screen(mid_tall, series["handle"], post["instagram"])),
            phone(ig_reel_screen(reel_frame(mid_tall), series["handle"], post["instagram"])),
            phone(li_feed_screen(mid_wide, series["display_name"], series["headline"], post["linkedin"])),
            phone(li_vertical_screen(reel_frame(mid_tall), series["display_name"], series["headline"],
                                     post["linkedin"])),
        ]
        labeled(panels, ["Instagram · feed post (3:4)", "Instagram · Reel (9:16)", "LinkedIn · feed post (16:9)",
                         "LinkedIn · vertical video (9:16)"],
                title=f"{part}/{total}  {post['title']}  ·  {day:%a %b %d}").save(folder / "preview.png")

    grid, reels = ig_profile_screens(series, list(reversed(posts)))
    labeled([phone(grid), phone(reels)], ["Posts grid after all 9 (newest first)", "Reels tab"],
            title="Instagram profile").save(OUT / "mockups" / "instagram_profile.png")
    li_activity(series, posts).save(OUT / "mockups" / "linkedin_activity.png")
    timeline(series, posts).save(OUT / "mockups" / "timeline.png")

    with open(OUT / "schedule.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["part", "date", "weekday", "title", "instagram_time", "linkedin_time", "folder",
                         "seconds", "instagram_post", "instagram_reel", "linkedin_post", "linkedin_vertical"])
        for part, post in enumerate(posts, start=1):
            writer.writerow([part, post["date"].isoformat(), f"{post['date']:%a}", post["title"],
                             series["instagram_time"], series["linkedin_time"], post["folder"],
                             f"{post['duration']:.1f}", "instagram_post_3x4.mp4", "instagram_reel_9x16.mp4",
                             "linkedin_post_16x9.mp4", "linkedin_vertical_9x16.mp4"])

    rows = "\n".join(
        f"| {part} | {post['date']:%a %b %d} | {post['title']} | {post['duration']:.0f} s | "
        f"[{post['folder']}]({post['folder']}/preview.png) |"
        for part, post in enumerate(posts, start=1)
    )
    (OUT / "schedule.md").write_text(
        f"# {series['name']}: posting schedule\n\n"
        f"{total} posts, {' and '.join(series['weekdays'])} from {long_date(days[0])} to {long_date(days[-1])}. "
        f"Instagram at {series['instagram_time']}, LinkedIn at {series['linkedin_time']}, local time.\n"
        "The series builds an argument, so it reads best in this order.\n\n"
        "| # | Date | Title | Length | Preview |\n|---|---|---|---|---|\n"
        f"{rows}\n\n"
        "Overview mockups: [timeline](mockups/timeline.png) (every placement, in order), "
        "[Instagram profile](mockups/instagram_profile.png), "
        "[LinkedIn activity](mockups/linkedin_activity.png).\n"
        f"{SPECS}",
        encoding="utf-8",
    )
    print(f"wrote {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
