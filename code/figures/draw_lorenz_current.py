"""Draw the manuscript's single-panel Lorenz figure from frozen empirical coordinates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, default=ROOT.parent)
args = parser.parse_args()
source = args.source.resolve()
data_path = source / "Source_Data/Fig1_concentration/concentration_rank_plot_data.csv"
original_builder = source / "reproducibility/build_reviewed_main_figures.py"
out = source / "figures"
out.mkdir(parents=True, exist_ok=True)
image_out = out
image_out.mkdir(parents=True, exist_ok=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

with data_path.open(encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))

font = "Helvetica"
if Path("C:/Windows/Fonts/arial.ttf").exists():
    pdfmetrics.registerFont(TTFont("FigureArial", "C:/Windows/Fonts/arial.ttf"))
    font = "FigureArial"

pdf_path = out / "concentration_rank.pdf"
c = canvas.Canvas(str(pdf_path), pagesize=(530, 486), invariant=1, pageCompression=1)
c.setTitle("Monetary concentration of events and wallet participation")
c.setSubject("Standalone Figure 1; frozen empirical Lorenz curves, unchanged results")
ink, grid = "#17324D", "#E7EAEE"
left, bottom, side = 82, 65, 400

def xy(x, y):
    return left + side * x, bottom + side * y

def text(x, y, value, size=13, color=ink, align="left"):
    c.setFont(font, size)
    c.setFillColor(HexColor(color))
    getattr(c, {"left": "drawString", "center": "drawCentredString", "right": "drawRightString"}[align])(x, y, value)

def line(x1, y1, x2, y2, color, width=1, dash=None):
    c.saveState()
    c.setStrokeColor(HexColor(color))
    c.setLineWidth(width)
    if dash:
        c.setDash(dash)
    c.line(x1, y1, x2, y2)
    c.restoreState()

for v in np.linspace(0, 1, 6):
    x, y = xy(v, v)
    line(x, bottom, x, bottom + side, grid, .6)
    line(left, y, left + side, y, grid, .6)
    text(x, bottom - 22, f"{v:.0%}", align="center")
    text(left - 12, y - 4, f"{v:.0%}", align="right")
line(*xy(0, 0), *xy(1, 1), "#8A8A8A", 1.2, [5, 4])
line(left, bottom, left, bottom + side, ink, .9)
line(left, bottom, left + side, bottom, ink, .9)

# Ginis are the original full-array values from the source figure builder.
# They must not be recomputed by numerical integration of thinned display points.
definitions = [
    ("Events", "Events", "#235789", .932587391694129, 764378, 438, 340),
    ("Wallets", "Wallet addresses", "#AD5318", .9623243404979402, 3106517, 411, 275),
]
audit = {"source_release": source.name, "inputs_sha256": {
    str(data_path.relative_to(source)): sha(data_path),
    str(original_builder.relative_to(source)): sha(original_builder),
}, "change": "Standalone Lorenz plot; rank-size panel removed; empirical coordinates and full-array statistics unchanged.",
    "populations": {}, "curve_interpolation": "Straight segments between retained empirical coordinates, with (0,0) added; no smoothing or fitting."}
for population, label, color, gini, expected_n, legend_y, annotation_y in definitions:
    selected = [row for row in rows if row["population"] == population]
    indices = np.array([int(row["rank"]) for row in selected])
    fractions = np.array([float(row["population_fraction"]) for row in selected])
    shares = np.array([float(row["cumulative_volume_ascending"]) for row in selected])
    n = int(indices[-1])
    assert n == expected_n
    assert indices[0] == 1 and np.all(np.diff(indices) > 0)
    assert np.allclose(fractions, indices / n, rtol=1e-12, atol=1e-15)
    assert np.all(np.diff(shares) >= 0) and np.all(shares <= fractions + 1e-12)
    assert shares[0] >= 0 and math.isclose(shares[-1], 1.0)
    p = c.beginPath()
    p.moveTo(*xy(0, 0))
    for x, y in zip(fractions, shares):
        p.lineTo(*xy(x, y))
    c.setStrokeColor(HexColor(color))
    c.setLineWidth(2.1)
    c.drawPath(p)
    line(99, legend_y, 128, legend_y, color, 2.1)
    text(138, legend_y - 4, f"{label} (G = {gini:.3f})", 13)

    anchor_index = n - math.ceil(.01 * n)
    matches = np.flatnonzero(indices == anchor_index)
    assert len(matches) == 1
    j = int(matches[0])
    anchor = xy(float(fractions[j]), float(shares[j]))
    c.setFillColor(HexColor(color))
    c.circle(*anchor, 3.0, fill=1, stroke=0)
    annotation = f"Bottom 99%: {100 * shares[j]:.1f}%"
    # Labels sit wholly above the equality diagonal, without obscuring it.
    ax, ay = 155, annotation_y
    width = pdfmetrics.stringWidth(annotation, font, 13)
    text(ax, ay, annotation, 13, color)
    line(ax + width - 5, ay - 8, *anchor, color, .8)
    audit["populations"][population] = {
        "n": n, "retained_display_coordinates": len(selected),
        "gini_full_array_inherited": gini,
        "anchor_population_fraction": float(fractions[j]),
        "bottom_99_share": float(shares[j]), "top_1_share": float(1 - shares[j]),
        "monotone_below_equality_and_endpoints_verified": True,
        "coordinate_sha256": hashlib.sha256(json.dumps(list(zip(fractions.tolist(), shares.tolist())), separators=(",", ":")).encode()).hexdigest(),
    }

line(99, 384, 128, 384, "#8A8A8A", 1.2, [5, 4])
text(138, 380, "Equality", 13)
text(left + side / 2, 17, "Cumulative fraction of events or wallets", 13, align="center")
c.saveState()
c.translate(23, bottom + side / 2)
c.rotate(90)
text(0, 0, "Cumulative share of total amount", 13, align="center")
c.restoreState()
c.showPage()
c.save()

reader = PdfReader(str(pdf_path))
assert len(reader.pages) == 1
page_text = reader.pages[0].extract_text()
assert "rank-size" not in page_text.lower()
assert "Events (G = 0.933)" in page_text
assert "Wallet addresses (G = 0.962)" in page_text
assert "Bottom 99%: 36.0%" in page_text and "Bottom 99%: 20.8%" in page_text
poppler = shutil.which("pdftoppm")
if poppler:
    png_base = image_out / "concentration_rank"
    subprocess.run([poppler, "-r", "160", "-png", "-singlefile", str(pdf_path), str(png_base)], check=True, capture_output=True)
print(json.dumps({"file": pdf_path.name, "populations": audit["populations"]}, indent=2))
