#!/usr/bin/env python3
"""Build LateAct's web-ready showcase from frozen experiment artifacts.

This script performs only lightweight composition and plotting. It never imports
the model runtime and never launches inference.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path(
    os.environ.get("LATEACT_ARTIFACT_ROOT", ROOT.parent / "artifacts" / "lateact")
)
GATE0 = ARTIFACTS / "gate0-20260823"
GATE1 = ARTIFACTS / "gate1"
GATE2 = ARTIFACTS / "gate2" / "primary"
CASE_DIR = GATE1 / "0005"
CASE_DIRECTION = "right_to_left"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

BG = "#F5F7FA"
INK = "#14202B"
MUTED = "#5B6874"
GRID = "#DCE3E8"
OLD = "#E26D5A"
NEW = "#238C72"
LATE = "#2D68C4"
RESTART = "#7A62B3"
AMBER = "#EAA42A"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def text(draw: ImageDraw.ImageDraw, xy, value, size, fill=INK, bold=False, anchor=None):
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def rounded(draw: ImageDraw.ImageDraw, box, radius=20, fill="white", outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return frames, fps


def fit_frame(frame: np.ndarray, size: tuple[int, int]) -> Image.Image:
    im = Image.fromarray(frame)
    return im.resize(size, Image.Resampling.LANCZOS)


def case_paths() -> dict[str, Path]:
    stem = CASE_DIRECTION
    return {
        "old": CASE_DIR / f"{stem}_old.mp4",
        "direct": CASE_DIR / f"{stem}_direct_late_bind.mp4",
        "lateact": CASE_DIR / f"{stem}_minimal_rollback.mp4",
        "restart": CASE_DIR / f"{stem}_full_restart.mp4",
        "new": CASE_DIR / f"{stem}_new_oracle.mp4",
    }


def copy_sources(paths: dict[str, Path]) -> None:
    out = ROOT / "source_clips"
    out.mkdir(exist_ok=True)
    for key, src in paths.items():
        shutil.copy2(src, out / f"canonical_{key}.mp4")
    frozen = ROOT / "frozen_sources"
    frozen.mkdir(exist_ok=True)
    for src, name in [
        (GATE0 / "gate0_summary.json", "gate0_summary.json"),
        (GATE1 / "gate1_summary.json", "gate1_summary.json"),
        (GATE2 / "mouse_summary.json", "gate2_mouse_summary.json"),
    ]:
        shutil.copy2(src, frozen / name)


def panel_frame(frames: dict[str, list[np.ndarray]], idx: int) -> Image.Image:
    w, h = 320, 176
    labels = [
        ("direct", "DIRECT LATE BIND", "response 0.191", OLD),
        ("lateact", "LATEACT", "response 0.997", NEW),
        ("restart", "FULL RESTART", "response 1.000", RESTART),
        ("new", "NEW ORACLE", "reference", NEW),
    ]
    canvas = Image.new("RGB", (1280, 318), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (38, 25), "Action update arrives after the commitment boundary", 26, INK, True)
    text(draw, (1240, 27), "OLD yaw  →  NEW yaw", 18, MUTED, False, "ra")
    for col, (key, title, metric, color) in enumerate(labels):
        x = col * w
        frame = fit_frame(frames[key][idx % len(frames[key])], (w, h))
        canvas.paste(frame, (x, 76))
        draw.rectangle((x, 76, x + w - 1, 81), fill=color)
        if col:
            draw.line((x, 76, x, 252), fill="white", width=3)
        text(draw, (x + 16, 274), title, 17, color, True)
        text(draw, (x + w - 16, 275), metric, 14, MUTED, False, "ra")
    text(draw, (38, 306), "Same prefix, scene, sampler, and coupled noise · Matrix-Game 2.0 · validated Gate 1 case 0005", 12, MUTED, False, "lm")
    return canvas


def make_demo(paths: dict[str, Path]) -> dict[str, list[np.ndarray]]:
    videos = {k: read_video(v)[0] for k, v in paths.items()}
    n = min(len(videos[k]) for k in ("direct", "lateact", "restart", "new"))
    # Two synchronized passes make the comparison long enough to scan on a web page.
    writer = imageio.get_writer(
        ROOT / "lateact_demo.mp4",
        fps=12,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        ffmpeg_log_level="warning",
        macro_block_size=None,
    )
    for _ in range(2):
        for i in range(n):
            writer.append_data(np.asarray(panel_frame(videos, i)))
    writer.close()
    return videos


def make_thumbnail(videos: dict[str, list[np.ndarray]]) -> None:
    canvas = Image.new("RGB", (1200, 675), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (50, 38), "LateAct restores asynchronous control", 37, INK, True)
    text(draw, (50, 84), "Representative validated Gate 1 case · case-specific responses", 18, LATE, True)
    text(draw, (1150, 84), "Identical context + noise", 16, MUTED, False, "ra")
    specs = [
        ("direct", ("DIRECT LATE", "BIND"), "0.191 response", "retains OLD yaw", OLD),
        ("lateact", ("LATEACT",), "0.997 response", "recovers NEW yaw", NEW),
        ("restart", ("FULL RESTART",), "1.000 response", "recovers NEW yaw", RESTART),
        ("new", ("NEW ORACLE",), "reference", "target NEW yaw", NEW),
    ]
    idx = min(len(videos["direct"]) - 1, 20)
    for j, (key, label_lines, score, outcome, color) in enumerate(specs):
        row, col = divmod(j, 2)
        x, y = 50 + col * 565, 120 + row * 245
        rounded(draw, (x, y, x + 535, y + 220), 14, "white", GRID, 2)
        frame = fit_frame(videos[key][idx], (350, 198))
        canvas.paste(frame, (x + 10, y + 10))
        draw.rectangle((x + 10, y + 10, x + 16, y + 208), fill=color)
        tx = x + 380
        title_y = y + 50 if len(label_lines) == 1 else y + 38
        for line_no, label in enumerate(label_lines):
            text(draw, (tx, title_y + line_no * 24), label, 16, color, True)
        text(draw, (tx, y + 108), score, 15, INK, True)
        text(draw, (tx, y + 133), "case-specific", 12, MUTED)
        text(draw, (tx, y + 174), outcome, 14, OLD if key == "direct" else NEW)
    text(draw, (50, 636), "Matrix-Game 2.0 mouse-yaw · Gate 1 scene 0005, right-to-left · identical prefix, context, sampler, and noise", 14, MUTED)
    canvas.save(ROOT / "demo_thumbnail.png", optimize=True)


def mechanism_svg() -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
<defs><style>
text{{font-family:DejaVu Sans,Arial,sans-serif;fill:{INK}}}.h{{font-weight:700}}.small{{font-size:23px}}.label{{font-size:27px;font-weight:700}}
</style>
<marker id="arrow-neutral" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{MUTED}"/></marker>
<marker id="arrow-lateact" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{NEW}"/></marker>
<marker id="arrow-restart" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{RESTART}"/></marker></defs>
<rect width="1600" height="900" fill="{BG}"/>
<text x="80" y="78" class="h" font-size="45">LateAct rolls back to the latest pre-commitment state</text>
<text x="80" y="118" font-size="23" fill="{MUTED}">A new action arrives after NFE2; the latest safe state is the input to NFE2.</text>
<line x1="390" y1="198" x2="1370" y2="198" stroke="{GRID}" stroke-width="8" stroke-linecap="round"/>
<line x1="390" y1="198" x2="710" y2="198" stroke="{NEW}" stroke-width="8" stroke-linecap="round"/>
<line x1="720" y1="142" x2="720" y2="818" stroke="{AMBER}" stroke-width="4" stroke-dasharray="12 10"/>
<text x="720" y="172" text-anchor="middle" font-size="20" font-weight="700" fill="{AMBER}">COMMITMENT BOUNDARY</text>
<g font-size="25" font-weight="700"><text x="390" y="237" text-anchor="middle">PREFIX</text><text x="585" y="237" text-anchor="middle">NFE1</text><text x="890" y="237" text-anchor="middle">NFE2</text><text x="1195" y="237" text-anchor="middle">NFE3</text></g>
<rect x="950" y="128" width="350" height="55" rx="27" fill="{OLD}"/><text x="1125" y="164" text-anchor="middle" font-size="22" font-weight="700" fill="white">NEW ACTION ARRIVES</text>

<text x="80" y="354" class="label" fill="{OLD}">DIRECT LATE BIND</text>
<rect x="390" y="305" width="195" height="85" rx="18" fill="#F4D5D0"/><text x="488" y="358" text-anchor="middle" class="small h">NFE1 · OLD</text>
<path d="M585 347 H792" stroke="{MUTED}" stroke-width="4" marker-end="url(#arrow-neutral)"/>
<rect x="805" y="305" width="195" height="85" rx="18" fill="#F4D5D0"/><text x="902" y="358" text-anchor="middle" class="small h">NFE2 · OLD</text>
<path d="M1000 347 H1115" stroke="{MUTED}" stroke-width="4" marker-end="url(#arrow-neutral)"/>
<rect x="1128" y="305" width="250" height="85" rx="18" fill="#F4D5D0"/><text x="1253" y="344" text-anchor="middle" class="small h">NFE3 · NEW</text><text x="1253" y="372" text-anchor="middle" font-size="18" fill="{OLD}">too late; path is committed</text>

<text x="80" y="543" class="label" fill="{NEW}">LATEACT</text>
<rect x="390" y="494" width="195" height="85" rx="18" fill="#DDE7F7"/><text x="488" y="547" text-anchor="middle" class="small h">NFE1 · OLD</text>
<path d="M585 536 H684" stroke="{NEW}" stroke-width="4" marker-end="url(#arrow-lateact)"/>
<circle cx="720" cy="536" r="26" fill="{NEW}"/><path d="M710 536 l8 8 15 -18" fill="none" stroke="white" stroke-width="5"/>
<text x="720" y="588" text-anchor="middle" font-size="18" font-weight="700" fill="{NEW}">RESTORE</text>
<path d="M748 536 H792" stroke="{NEW}" stroke-width="4" marker-end="url(#arrow-lateact)"/>
<rect x="805" y="494" width="195" height="85" rx="18" fill="#D9EEE8"/><text x="902" y="547" text-anchor="middle" class="small h">NFE2 · NEW</text>
<path d="M1000 536 H1115" stroke="{NEW}" stroke-width="4" marker-end="url(#arrow-lateact)"/>
<rect x="1128" y="494" width="250" height="85" rx="18" fill="#D9EEE8"/><text x="1253" y="547" text-anchor="middle" class="small h">NFE3 · NEW</text>
<text x="1405" y="544" font-size="20" font-weight="700" fill="{NEW}">recompute suffix</text>

<text x="80" y="731" class="label" fill="{RESTART}">FULL RESTART</text>
<circle cx="342" cy="724" r="27" fill="{RESTART}"/><text x="342" y="734" text-anchor="middle" font-size="31" font-weight="700" fill="white">↺</text>
<text x="342" y="774" text-anchor="middle" font-size="16" font-weight="700" fill="{RESTART}">BLOCK START</text>
<path d="M369 724 H377" stroke="{RESTART}" stroke-width="4" marker-end="url(#arrow-restart)"/>
<rect x="390" y="682" width="195" height="85" rx="18" fill="#E9E2F4"/><text x="488" y="735" text-anchor="middle" class="small h">NFE1 · NEW</text>
<path d="M585 724 H792" stroke="{RESTART}" stroke-width="4" marker-end="url(#arrow-restart)"/>
<rect x="805" y="682" width="195" height="85" rx="18" fill="#E9E2F4"/><text x="902" y="735" text-anchor="middle" class="small h">NFE2 · NEW</text>
<path d="M1000 724 H1115" stroke="{RESTART}" stroke-width="4" marker-end="url(#arrow-restart)"/>
<rect x="1128" y="682" width="250" height="85" rx="18" fill="#E9E2F4"/><text x="1253" y="735" text-anchor="middle" class="small h">NFE3 · NEW</text>
<text x="80" y="843" font-size="30" font-weight="700">Keep reusable compute; roll back only across the commitment boundary.</text>
</svg>'''


def svg_to_png_via_pillow_fallback(svg_path: Path, png_path: Path, draw_fn) -> None:
    # The project environment intentionally has no SVG rasterizer. The PNG is
    # drawn from the same geometry; the SVG remains the vector source.
    draw_fn(png_path)


def make_mechanism_png(path: Path) -> None:
    im = Image.new("RGB", (1600, 900), BG)
    d = ImageDraw.Draw(im)
    text(d, (80, 48), "LateAct rolls back to the latest pre-commitment state", 44, INK, True)
    text(d, (80, 104), "A new action arrives after NFE2; the latest safe state is the input to NFE2.", 23, MUTED)
    d.line((390, 198, 1370, 198), fill=GRID, width=8)
    d.line((390, 198, 710, 198), fill=NEW, width=8)
    d.line((720, 142, 720, 818), fill=AMBER, width=4)
    text(d, (720, 160), "COMMITMENT BOUNDARY", 18, AMBER, True, "mm")
    for x, label in [(390, "PREFIX"), (585, "NFE1"), (890, "NFE2"), (1195, "NFE3")]:
        text(d, (x, 228), label, 24, INK, True, "mm")
    rounded(d, (950, 128, 1300, 183), 27, OLD)
    text(d, (1125, 156), "NEW ACTION ARRIVES", 21, "white", True, "mm")
    rows = [
        (347, "DIRECT LATE BIND", OLD, [("NFE1 · OLD", "#F4D5D0"), ("NFE2 · OLD", "#F4D5D0"), ("NFE3 · NEW\ntoo late", "#F4D5D0")]),
        (536, "LATEACT", NEW, [("NFE1 · OLD", "#DDE7F7"), ("NFE2 · NEW", "#D9EEE8"), ("NFE3 · NEW", "#D9EEE8")]),
        (724, "FULL RESTART", RESTART, [("NFE1 · NEW", "#E9E2F4"), ("NFE2 · NEW", "#E9E2F4"), ("NFE3 · NEW", "#E9E2F4")]),
    ]
    xs = [(390, 585), (805, 1000), (1128, 1378)]
    for cy, label, color, boxes in rows:
        text(d, (80, cy), label, 26, color, True, "lm")
        for i, ((val, fill), (x0, x1)) in enumerate(zip(boxes, xs)):
            rounded(d, (x0, cy - 42, x1, cy + 43), 18, fill)
            parts = val.split("\n")
            text(d, ((x0 + x1) // 2, cy - (10 if len(parts) == 2 else 0)), parts[0], 22, INK, True, "mm")
            if len(parts) == 2:
                text(d, ((x0 + x1) // 2, cy + 21), parts[1], 16, color, False, "mm")
            if i < 2:
                d.line((x1 + 8, cy, xs[i + 1][0] - 12, cy), fill=color if label != "DIRECT LATE BIND" else MUTED, width=4)
                d.polygon([(xs[i + 1][0] - 12, cy), (xs[i + 1][0] - 25, cy - 8), (xs[i + 1][0] - 25, cy + 8)], fill=color if label != "DIRECT LATE BIND" else MUTED)
    d.ellipse((694, 510, 746, 562), fill=NEW)
    d.line((708, 536, 718, 546, 734, 522), fill="white", width=5)
    text(d, (720, 582), "RESTORE", 17, NEW, True, "mm")
    text(d, (1405, 536), "recompute suffix", 18, NEW, True, "lm")
    # Explicit block-start restart icon replaces the former improvised return arc.
    d.ellipse((315, 697, 369, 751), fill=RESTART)
    text(d, (342, 724), "↺", 31, "white", True, "mm")
    text(d, (342, 771), "BLOCK START", 15, RESTART, True, "mm")
    d.line((370, 724, 378, 724), fill=RESTART, width=4)
    d.polygon([(378, 724), (366, 716), (366, 732)], fill=RESTART)
    text(d, (80, 825), "Keep reusable compute; roll back only across the commitment boundary.", 29, INK, True)
    im.save(path, optimize=True)


def make_mechanism() -> None:
    svg = ROOT / "lateact_mechanism.svg"
    svg.write_text(mechanism_svg(), encoding="utf-8")
    svg_to_png_via_pillow_fallback(svg, ROOT / "lateact_mechanism.png", make_mechanism_png)


def curve_data() -> tuple[list[list[float]], list[float]]:
    data = json.loads((GATE0 / "gate0_summary.json").read_text())
    curves = []
    for scene in sorted(data["curves"]):
        for direction in ("left_to_right", "right_to_left"):
            curves.append(data["curves"][scene][direction]["new_action_fraction_s0_to_s3"])
    med = np.median(np.asarray(curves), axis=0).tolist()
    return curves, med


def curve_geometry(curves, med):
    x0, y0, x1, y1 = 180, 175, 1320, 700
    px = lambda i: x0 + i * (x1 - x0) / 3
    py = lambda v: y1 - max(0, min(1.1, v)) * (y1 - y0) / 1.1
    return x0, y0, x1, y1, px, py


def make_curve() -> None:
    curves, med = curve_data()
    x0, y0, x1, y1, px, py = curve_geometry(curves, med)
    im = Image.new("RGB", (1400, 900), "white")
    d = ImageDraw.Draw(im)
    text(d, (70, 40), "Mouse-yaw action response commits sharply during denoising", 34, INK, True)
    text(d, (70, 91), "Gate 0 · thin lines: 16 validated pairs · bold blue: median trend", 18, MUTED)
    for val in np.arange(0, 1.01, 0.2):
        y = py(float(val))
        d.line((x0, y, x1, y), fill=GRID, width=1)
        text(d, (x0 - 22, y), f"{val:.1f}", 16, MUTED, False, "rm")
    d.line((x0, y0, x0, y1), fill=INK, width=2)
    d.line((x0, y1, x1, y1), fill=INK, width=2)
    for i, label in enumerate(["0\nNEW oracle", "1", "2", "3\nOLD oracle"]):
        x = px(i)
        d.line((x, y1, x, y1 + 8), fill=INK, width=2)
        parts = label.split("\n")
        text(d, (x, y1 + 28), parts[0], 18, INK, True, "mm")
        if len(parts) > 1:
            text(d, (x, y1 + 54), parts[1], 14, MUTED, False, "mm")
    text(d, ((x0 + x1) // 2, 810), "OLD-action NFEs completed before switching to NEW", 19, INK, True, "mm")
    ylabel = Image.new("RGBA", (390, 50), (255, 255, 255, 0))
    yd = ImageDraw.Draw(ylabel)
    text(yd, (195, 25), "Normalized NEW-action response", 18, INK, True, "mm")
    ylabel = ylabel.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    im.paste(ylabel, (42, int((y0 + y1 - ylabel.height) / 2)), ylabel)
    # Faint individual curves establish dispersion without obscuring the median.
    for curve in curves:
        pts = [(px(i), py(v)) for i, v in enumerate(curve)]
        d.line(pts, fill="#B9C7D3", width=2)
    pts = [(px(i), py(v)) for i, v in enumerate(med)]
    d.line(pts, fill=LATE, width=7, joint="curve")
    for x, y in pts:
        d.ellipse((x - 9, y - 9, x + 9, y + 9), fill="white", outline=LATE, width=5)
    for i, display in [(1, "0.997"), (2, "0.218")]:
        x, y = pts[i]
        rounded(d, (x - 58, y - 60, x + 58, y - 20), 15, LATE)
        text(d, (x, y - 40), display, 18, "white", True, "mm")
    # The paired result is separated from both linework and callouts.
    cx, cy = (px(1) + px(2)) / 2, (py(med[1]) + py(med[2])) / 2
    rounded(d, (cx - 132, cy - 21, cx + 132, cy + 21), 16, "#FFF4DF")
    text(d, (cx, cy), "paired median drop  0.774", 17, AMBER, True, "mm")
    text(d, (1315, 856), "Source: frozen Gate 0", 13, MUTED, False, "ra")
    im.save(ROOT / "commitment_curve.png", optimize=True)
    # Standalone SVG with the same full-scale axes and source data.
    path_bits = []
    for curve in curves:
        points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(curve))
        path_bits.append(f'<polyline points="{points}" fill="none" stroke="#B9C7D3" stroke-width="2"/>')
    med_points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(med))
    y_grid = "".join(f'<line x1="{x0}" y1="{py(float(v)):.1f}" x2="{x1}" y2="{py(float(v)):.1f}" stroke="{GRID}"/><text x="{x0-22}" y="{py(float(v))+6:.1f}" text-anchor="end" font-size="16" fill="{MUTED}">{v:.1f}</text>' for v in np.arange(0,1.01,.2))
    x_ticks = "".join(f'<text x="{px(i):.1f}" y="734" text-anchor="middle" font-size="18" font-weight="700">{i}</text>' for i in range(4))
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="white" stroke="{LATE}" stroke-width="5"/>' for x,y in pts)
    cx, cy = (px(1) + px(2)) / 2, (py(med[1]) + py(med[2])) / 2
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900"><rect width="1400" height="900" fill="white"/><g font-family="DejaVu Sans,Arial,sans-serif" fill="{INK}"><text x="70" y="68" font-size="34" font-weight="700">Mouse-yaw action response commits sharply during denoising</text><text x="70" y="112" font-size="18" fill="{MUTED}">Gate 0 · thin lines: 16 validated pairs · bold blue: median trend</text>{y_grid}<text x="54" y="{(y0+y1)/2}" text-anchor="middle" font-size="18" font-weight="700" transform="rotate(-90 54 {(y0+y1)/2})">Normalized NEW-action response</text><line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{INK}" stroke-width="2"/><line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{INK}" stroke-width="2"/>{''.join(path_bits)}<polyline points="{med_points}" fill="none" stroke="{LATE}" stroke-width="7"/>{dots}{x_ticks}<text x="{px(0)}" y="760" text-anchor="middle" font-size="14" fill="{MUTED}">NEW oracle</text><text x="{px(3)}" y="760" text-anchor="middle" font-size="14" fill="{MUTED}">OLD oracle</text><text x="{(x0+x1)/2}" y="810" text-anchor="middle" font-size="19" font-weight="700">OLD-action NFEs completed before switching to NEW</text><rect x="{px(1)-58}" y="{py(med[1])-60}" width="116" height="40" rx="15" fill="{LATE}"/><text x="{px(1)}" y="{py(med[1])-33}" text-anchor="middle" font-size="18" font-weight="700" fill="white">0.997</text><rect x="{px(2)-58}" y="{py(med[2])-60}" width="116" height="40" rx="15" fill="{LATE}"/><text x="{px(2)}" y="{py(med[2])-33}" text-anchor="middle" font-size="18" font-weight="700" fill="white">0.218</text><rect x="{cx-132}" y="{cy-21}" width="264" height="42" rx="16" fill="#FFF4DF"/><text x="{cx}" y="{cy+6}" text-anchor="middle" font-size="17" font-weight="700" fill="{AMBER}">paired median drop  0.774</text><text x="1315" y="856" text-anchor="end" font-size="13" fill="{MUTED}">Source: frozen Gate 0</text></g></svg>'''
    (ROOT / "commitment_curve.svg").write_text(svg, encoding="utf-8")


def make_rollback() -> None:
    initial, exact, logical = 649_330_080, 339_360, 337_928
    im = Image.new("RGB", (1400, 820), BG)
    d = ImageDraw.Draw(im)
    text(d, (70, 48), "Exact rollback state, reduced to the necessary runtime state", 36, INK, True)
    text(d, (70, 98), "Bytes per checkpoint · logarithmic bar length", 19, MUTED)
    values = [
        ("Initial exact implementation", initial, OLD, "649,330,080 B  (~649 MB)"),
        ("Final exact rollback state", exact, LATE, "339,360 B  (~339 KB)"),
        ("Measured logical minimum", logical, NEW, "337,928 B"),
    ]
    bx, maxw = 510, 760
    logmax = math.log10(initial)
    logmin = 3
    for i, (label, value, color, display) in enumerate(values):
        y = 210 + i * 155
        text(d, (70, y + 25), label, 22, INK, i == 1)
        width = 130 + (math.log10(value) - logmin) / (logmax - logmin) * (maxw - 130)
        rounded(d, (bx, y, bx + width, y + 70), 18, color)
        text(d, (bx + 22, y + 35), display, 20, "white", True, "lm")
    rounded(d, (70, 665, 1330, 765), 18, "white", GRID, 2)
    text(d, (105, 700), "1,913× smaller", 30, LATE, True)
    text(d, (390, 700), "than the initial exact implementation", 20, MUTED)
    text(d, (105, 738), "+1,432 B (0.424%)", 24, NEW, True)
    text(d, (390, 738), "above the measured logical minimum", 20, MUTED)
    text(d, (1315, 790), "Source: frozen Gate 1 + Gate 2 primary", 13, MUTED, False, "ra")
    im.save(ROOT / "rollback_state.png", optimize=True)


def make_cover(videos: dict[str, list[np.ndarray]]) -> None:
    im = Image.new("RGB", (1200, 630), "#101A24")
    d = ImageDraw.Draw(im)
    # Actual model outputs: direct vs LateAct vs NEW, with a dark research-card treatment.
    idx = min(len(videos["direct"]) - 1, 20)
    frame_w, frame_h = 320, 176
    for i, key in enumerate(("direct", "lateact", "new")):
        x = 160 + i * 315
        fr = fit_frame(videos[key][idx], (frame_w, frame_h))
        im.paste(fr, (x, 310))
        d.rectangle((x, 310, x + frame_w, 316), fill=OLD if key == "direct" else NEW)
        label = {"direct": "DIRECT: OLD", "lateact": "LATEACT: NEW", "new": "NEW ORACLE"}[key]
        text(d, (x + frame_w / 2, 513), label, 17, "#F7FAFC", True, "mm")
    text(d, (80, 62), "LateAct", 67, "white", True)
    text(d, (82, 148), "Asynchronous Control for", 31, "#CDD7E0", True)
    text(d, (82, 190), "Interactive Video World Models", 31, "#CDD7E0", True)
    # Compact mechanism cue.
    d.line((720, 115, 1070, 115), fill="#475869", width=7)
    for x, lab in [(720, "NFE1"), (895, "NFE2"), (1070, "NFE3")]:
        d.ellipse((x - 9, 106, x + 9, 124), fill=LATE if x < 895 else "#475869")
        text(d, (x, 146), lab, 15, "#CDD7E0", True, "mm")
    d.line((810, 90, 810, 169), fill=AMBER, width=3)
    text(d, (810, 75), "rollback", 15, AMBER, True, "mm")
    rounded(d, (80, 558, 1120, 602), 20, "#1B2A38")
    text(d, (600, 580), "Restore the latest pre-commitment state · recompute only the suffix", 19, "#D9E5EC", True, "mm")
    im.save(ROOT / "og_cover.png", optimize=True)


def main() -> None:
    paths = case_paths()
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen source clips: " + ", ".join(missing))
    copy_sources(paths)
    videos = make_demo(paths)
    make_thumbnail(videos)
    make_mechanism()
    make_curve()
    make_rollback()
    make_cover(videos)
    print("Built showcase assets from frozen artifacts; no inference was run.")


if __name__ == "__main__":
    main()
