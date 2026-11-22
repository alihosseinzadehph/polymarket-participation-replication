"""Build the balanced two-panel complete-market subject-network figure.

Panel a aggregates the frozen 21-subject amount coallocation matrix into
seven broad subject families and displays every positive off-diagonal pair.
Panel b displays every subject pair retained by the frozen BiCM-Holm audit.
The two panels use equal drawing regions, common typography, and a shared
family colour registry.  Layout is descriptive and carries no distance or
hierarchy interpretation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from reportlab.lib.colors import Color, HexColor, white
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "Source_Data/Fig5_subject_networks"
NETWORK_INPUT = SOURCE_DIR / "subject_networks.json"
BICM_INPUT = SOURCE_DIR / "subject_bicm_full.json"
OUTPUT = ROOT / "figures/subject_networks_journal.pdf"
PROVENANCE = ROOT / "SUBJECT_NETWORKS_JOURNAL_PROVENANCE_20261004.json"

BROAD_LABELS = [
    "Finance/economy",
    "Crypto",
    "Culture",
    "Other/meta",
    "Politics",
    "Sports",
    "Weather",
]

DETAILED_LABELS = [
    "Commodities",
    "Altcoins/DeFi",
    "Major crypto",
    "Intraday crypto",
    "Macro economy",
    "Entertainment",
    "Stocks/technology",
    "Mentions/trending",
    "Other",
    "Global politics",
    "US federal politics",
    "US state/local politics",
    "Baseball",
    "Basketball",
    "Esports",
    "Hockey",
    "NFL/CFB",
    "Other sports",
    "Soccer",
    "Tennis",
    "Weather/climate",
]

# Zero-based broad-family assignment for the fixed 21-subject taxonomy.
BROAD_INDEX = [0, 1, 1, 1, 0, 2, 0, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6]
BROAD_COLORS = [
    "#8663A8",
    "#148C83",
    "#BE607C",
    "#6B7786",
    "#C7832C",
    "#4179B0",
    "#AA8B18",
]

INK = "#17324D"
MUTED = "#536B7F"
LIGHT_EDGE = "#A8B7C5"
PERSISTENT_EDGE = "#1D4E89"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    value: str,
    size: float = 9.0,
    color: str = INK,
    align: str = "left",
    bold: bool = False,
) -> None:
    pdf.setFillColor(HexColor(color))
    pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    if align == "right":
        pdf.drawRightString(x, y, value)
    elif align == "center":
        pdf.drawCentredString(x, y, value)
    else:
        pdf.drawString(x, y, value)


def circle_positions(cx: float, cy: float, radius: float, n: int) -> list[tuple[float, float]]:
    return [
        (
            cx + radius * math.cos(math.pi / 2 - 2 * math.pi * index / n),
            cy + radius * math.sin(math.pi / 2 - 2 * math.pi * index / n),
        )
        for index in range(n)
    ]


def aggregate_amount(matrix: list[list[float]]) -> list[list[float]]:
    result = [[0.0 for _ in BROAD_LABELS] for _ in BROAD_LABELS]
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            result[BROAD_INDEX[i]][BROAD_INDEX[j]] += float(value)
    return result


def draw_panel_a(pdf: canvas.Canvas) -> dict[str, float | int]:
    source = json.loads(NETWORK_INPUT.read_text(encoding="utf-8"))
    matrix = aggregate_amount(source["joint_amount"])
    strengths = [sum(row) for row in matrix]
    positions = circle_positions(250.0, 400.0, 135.0, len(BROAD_LABELS))

    off_diagonal = [
        matrix[i][j]
        for i in range(len(matrix))
        for j in range(i + 1, len(matrix))
        if matrix[i][j] > 0
    ]
    if len(off_diagonal) != 21:
        raise RuntimeError(f"expected 21 positive broad-family pairs, found {len(off_diagonal)}")
    max_edge = max(off_diagonal)

    # Weaker links are drawn first, so strong relationships remain legible.
    pairs = sorted(
        (matrix[i][j], i, j)
        for i in range(len(matrix))
        for j in range(i + 1, len(matrix))
        if matrix[i][j] > 0
    )
    for weight, i, j in pairs:
        fraction = math.sqrt(weight / max_edge)
        shade = 0.76 - 0.46 * fraction
        pdf.setStrokeColor(Color(shade, shade + 0.035, min(1.0, shade + 0.075)))
        pdf.setLineWidth(0.65 + 4.8 * fraction)
        pdf.line(*positions[i], *positions[j])

    max_strength = max(strengths)
    for index, (x, y) in enumerate(positions):
        radius = 11.0 + 19.0 * math.sqrt(strengths[index] / max_strength)
        pdf.setFillColor(HexColor(BROAD_COLORS[index]))
        pdf.setStrokeColor(white)
        pdf.setLineWidth(1.25)
        pdf.circle(x, y, radius, fill=1, stroke=1)
        text(pdf, x, y - 4.8, str(index + 1), 13.5, "#FFFFFF", "center", True)

        angle = math.pi / 2 - 2 * math.pi * index / len(BROAD_LABELS)
        label_x = 250.0 + 178.0 * math.cos(angle)
        label_y = 400.0 + 178.0 * math.sin(angle) - 5.0
        # Inward anchoring keeps the side labels inside their panel.
        cosine = math.cos(angle)
        align = "center" if abs(cosine) < 0.25 else ("left" if cosine > 0 else "right")
        text(pdf, label_x, label_y, BROAD_LABELS[index], 14.0, INK, align, True)

    return {
        "positive_pairs": len(off_diagonal),
        "matrix_total": sum(strengths),
        "sports_marginal": strengths[5],
        "politics_marginal": strengths[4],
    }


def draw_panel_b(pdf: canvas.Canvas) -> dict[str, float | int]:
    network = json.loads(NETWORK_INPUT.read_text(encoding="utf-8"))
    bicm = json.loads(BICM_INPUT.read_text(encoding="utf-8"))
    shared = network["shared"]
    popularity = [float(shared[index][index]) for index in range(len(DETAILED_LABELS))]
    enriched = [row for row in bicm["rows"] if row["enriched_holm_005"]]
    persistent = [row for row in enriched if row["latest_four_ratios_all_above_one"]]
    if len(enriched) != 75 or len(persistent) != 22:
        raise RuntimeError("frozen BiCM edge counts do not match the manuscript registry")

    centre_x, centre_y, radius = 750.0, 400.0, 135.0
    positions = circle_positions(centre_x, centre_y, radius, len(DETAILED_LABELS))
    maximum_ratio = max(float(row["observed_expected_ratio"]) for row in enriched)

    for row in sorted(enriched, key=lambda item: bool(item["latest_four_ratios_all_above_one"])):
        a, b = int(row["a"]), int(row["b"])
        ratio = float(row["observed_expected_ratio"])
        fraction = (ratio - 1.0) / (maximum_ratio - 1.0)
        is_persistent = bool(row["latest_four_ratios_all_above_one"])
        pdf.setStrokeColor(HexColor(PERSISTENT_EDGE if is_persistent else "#9CC7D8"))
        pdf.setLineWidth(0.55 + 3.00 * fraction)
        pdf.line(*positions[a], *positions[b])

    max_popularity = max(popularity)
    for index, (x, y) in enumerate(positions):
        node_radius = 9.0 + 14.0 * math.sqrt(popularity[index] / max_popularity)
        pdf.setFillColor(HexColor(BROAD_COLORS[BROAD_INDEX[index]]))
        pdf.setStrokeColor(white)
        pdf.setLineWidth(1.1)
        pdf.circle(x, y, node_radius, fill=1, stroke=1)
        text(pdf, x, y - 4.3, str(index + 1), 12.0, "#FFFFFF", "center", True)

    return {
        "enriched_pairs": len(enriched),
        "persistent_pairs": len(persistent),
        "maximum_observed_expected_ratio": maximum_ratio,
    }


def build() -> None:
    page_width, page_height = 1000.0, 650.0
    pdf = canvas.Canvas(str(OUTPUT), pagesize=(page_width, page_height), pageCompression=1)
    pdf.setTitle("Complete-market subject organization")
    pdf.setAuthor("Polymarket integrated manuscript")
    pdf.setFillColor(white)
    pdf.rect(0, 0, page_width, page_height, fill=1, stroke=0)

    # The panels have identical 500 x 650 point canvases and aligned headings.
    text(pdf, 18, 628, "a", 21.0, INK, bold=True)
    text(pdf, 56, 628, "Amount-weighted coallocation", 19.0, INK, bold=True)
    text(pdf, 250, 600, "7 broad subjects; all 21 positive pairs", 13.0, MUTED, "center")

    text(pdf, 518, 628, "b", 21.0, INK, bold=True)
    text(pdf, 556, 628, "Degree-constrained excess overlap", 19.0, INK, bold=True)
    text(pdf, 750, 600, "21 detailed subjects; 75 of 210 pairs retained", 13.0, MUTED, "center")

    pdf.setStrokeColor(HexColor("#DCE3E9"))
    pdf.setLineWidth(0.6)
    pdf.line(500, 180, 500, 634)

    panel_a = draw_panel_a(pdf)
    panel_b = draw_panel_b(pdf)

    # Compact legends use the same baseline and visual weight.
    pdf.setStrokeColor(HexColor("#52697D"))
    pdf.setLineWidth(3.2)
    pdf.line(76, 184, 126, 184)
    text(pdf, 138, 179.5, "Width: coallocation", 12.5, MUTED)
    pdf.setFillColor(HexColor(BROAD_COLORS[5]))
    pdf.setStrokeColor(white)
    pdf.circle(321, 184, 10, fill=1, stroke=1)
    text(pdf, 339, 179.5, "Size: total subject share", 12.5, MUTED)

    pdf.setStrokeColor(HexColor("#9CC7D8"))
    pdf.setLineWidth(2.6)
    pdf.line(548, 184, 592, 184)
    text(pdf, 604, 179.5, "BiCM-Holm", 12.5, MUTED)
    pdf.setStrokeColor(HexColor(PERSISTENT_EDGE))
    pdf.setLineWidth(3.4)
    pdf.line(709, 184, 753, 184)
    text(pdf, 765, 179.5, "Also persistent in latest 4 quarters", 12.5, MUTED)

    # A compact three-column key keeps panel-b labels legible after journal scaling.
    text(pdf, 22, 151, "Panel b subject key", 13.5, INK, bold=True)
    for index, label in enumerate(DETAILED_LABELS):
        column = index // 7
        row = index % 7
        x = 28.0 + 324.0 * column
        y = 127.0 - 16.0 * row
        pdf.setFillColor(HexColor(BROAD_COLORS[BROAD_INDEX[index]]))
        pdf.setStrokeColor(white)
        pdf.circle(x, y + 3.0, 4.8, fill=1, stroke=1)
        text(pdf, x + 11.0, y, f"{index + 1}. {label}", 12.5, INK)

    pdf.showPage()
    pdf.save()

    provenance = {
        "generated_by": "reproducibility/build_subject_networks_journal.py",
        "layout": (
            "Two equal 500 x 650 point panels. Nodes use fixed taxonomy-ordered circular positions; "
            "position and distance have no statistical meaning. Colours map the same seven broad "
            "families across panels."
        ),
        "panel_a": {
            "definition": (
                "The 21-subject amount-weighted coallocation matrix is aggregated to seven fixed broad "
                "families. Every positive off-diagonal pair is drawn. Edge width is monotone in the "
                "square root of coallocation; node radius equals 11 + 19*sqrt(s/max(s)), "
                "where s is the full row sum INCLUDING its diagonal, i.e. broad-family monetary share. "
                "It is not the strength of the visible off-diagonal graph."
            ),
            **panel_a,
        },
        "panel_b": {
            "definition": (
                "Every pair with enriched_holm_005=true in the frozen BiCM audit is drawn. Edge width "
                "is monotone in observed/expected overlap; dark edges also exceed expectation in each "
                "of the latest four complete quarters. Node area is monotone in observed active-wallet count."
            ),
            **panel_b,
        },
        "inputs": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (NETWORK_INPUT, BICM_INPUT)
        },
        "output": {
            "path": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(OUTPUT),
            "bytes": OUTPUT.stat().st_size,
        },
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    build()
