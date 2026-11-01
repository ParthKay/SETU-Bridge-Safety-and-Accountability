# ruff: noqa: E501
"""Generate the technical PDF report for the zero-trust bridge engine.

Usage::

    python scripts/generate_report_pdf.py

The report is written to
``Zero_Trust_Bridge_Survival_Engine_Technical_Report.pdf`` at the project root.
All figures are derived from the persisted artifacts in ``artifacts/``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIG_DIR = ROOT / "docs" / "figures"
OUTPUT_PDF = ROOT / "Zero_Trust_Bridge_Survival_Engine_Technical_Report.pdf"

NAVY = HexColor("#102A43")
TEAL = HexColor("#1F6F78")
ORANGE = HexColor("#C05621")
LIGHT_BLUE = HexColor("#EAF2F8")
LIGHT_GRAY = HexColor("#F4F6F8")
BORDER = HexColor("#CBD5E0")
GRAY = HexColor("#4A5568")
GREEN = HexColor("#276749")
RED = HexColor("#9B2C2C")

HEX_NAVY = "#102A43"
HEX_TEAL = "#1F6F78"
HEX_ORANGE = "#C05621"
HEX_GRAY = "#718096"

# Register a Unicode TrueType family so equations, en dashes, and comparison
# symbols render consistently in the PDF.
_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for _font_name, _file_name in (
    ("ReportSans", "DejaVuSans.ttf"),
    ("ReportSans-Bold", "DejaVuSans-Bold.ttf"),
    ("ReportSans-Oblique", "DejaVuSans-Oblique.ttf"),
    ("ReportSans-BoldOblique", "DejaVuSans-BoldOblique.ttf"),
):
    pdfmetrics.registerFont(TTFont(_font_name, str(_FONT_DIR / _file_name)))


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_artifacts() -> dict[str, Any]:
    required = [
        ARTIFACTS / "end_to_end_summary.json",
        ARTIFACTS / "data" / "pipeline_audit.json",
        ARTIFACTS / "module2" / "survival_report.json",
        ARTIFACTS / "module3" / "audit_report.json",
        ARTIFACTS / "module4" / "triage_report.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing report artifacts. Run scripts/run_end_to_end.py first:\n"
            + "\n".join(missing)
        )
    data = {
        "summary": _read_json(ARTIFACTS / "end_to_end_summary.json"),
        "pipeline": _read_json(ARTIFACTS / "data" / "pipeline_audit.json"),
        "survival": _read_json(ARTIFACTS / "module2" / "survival_report.json"),
        "audit": _read_json(ARTIFACTS / "module3" / "audit_report.json"),
        "triage": _read_json(ARTIFACTS / "module4" / "triage_report.json"),
        "bridges": pd.read_parquet(ARTIFACTS / "data" / "bridges.parquet"),
        "climate": pd.read_parquet(ARTIFACTS / "data" / "climate.parquet"),
        "inspections": pd.read_parquet(ARTIFACTS / "data" / "inspections.parquet"),
        "truth": pd.read_parquet(ARTIFACTS / "data" / "EVAL_ONLY_ground_truth.parquet"),
        "audit_table": pd.read_parquet(ARTIFACTS / "module3" / "audit_findings.parquet"),
        "triage_table": pd.read_parquet(ARTIFACTS / "module4" / "triage_findings.parquet"),
        "xai": pd.read_parquet(ARTIFACTS / "module4" / "xai_contributions.parquet"),
    }
    return data


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _style_axes(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#CBD5E0")
    axis.spines["bottom"].set_color("#CBD5E0")
    axis.tick_params(colors="#4A5568", labelsize=8)
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.7)
    axis.set_axisbelow(True)


def make_architecture_figure(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: str,
        face: str,
        edge: str,
    ) -> None:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            linewidth=1.2,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h - 3.0,
            title,
            ha="center",
            va="top",
            fontsize=9.4,
            fontweight="bold",
            color=edge,
        )
        ax.text(
            x + w / 2,
            y + h - 8.0,
            body,
            ha="center",
            va="top",
            fontsize=7.5,
            color="#2D3748",
            linespacing=1.35,
        )

    def arrow(x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#718096", lw=1.4),
        )
        if label:
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2 + 1.4,
                label,
                fontsize=6.7,
                color="#718096",
                ha="center",
            )

    box(
        2,
        43,
        20,
        13,
        "NBI / IBMS + Climate",
        "Static inventory, inspections,\nrainfall, discharge, scour",
        "#F7FAFC",
        HEX_GRAY,
    )
    box(
        27,
        43,
        21,
        13,
        "Module 1",
        "Synthetic panel + Andersen-Gill\nintervals + leakage-safe design",
        "#E6F4F1",
        HEX_TEAL,
    )
    box(
        53,
        43,
        21,
        13,
        "Module 2",
        "Cox TV + RSF + GBSA + ensemble\nHarrell / Uno / Brier metrics",
        "#EAF2F8",
        HEX_NAVY,
    )
    box(
        78,
        43,
        20,
        13,
        "Artifacts",
        "Models, metrics, reports,\nreproducible serialisation",
        "#F7FAFC",
        HEX_GRAY,
    )
    box(
        53,
        18,
        21,
        13,
        "Module 3",
        "Conditional survival audit\nZ-residual + IsolationForest",
        "#FFF5EB",
        HEX_ORANGE,
    )
    box(
        78,
        18,
        20,
        13,
        "Module 4",
        "Disaster shock + RUL triage\nSHAP force plots",
        "#F0FFF4",
        "#276749",
    )
    box(
        2,
        18,
        20,
        13,
        "Operational output",
        "Immediate NDT priorities,\nRUL loss, audit escalation",
        "#F7FAFC",
        HEX_GRAY,
    )

    arrow(22.5, 49.5, 26.5, 49.5, "schema")
    arrow(48.5, 49.5, 52.5, 49.5, "features")
    arrow(74.5, 49.5, 77.5, 49.5, "persist")
    arrow(63.5, 42.5, 63.5, 31.5, "S(t | X)")
    arrow(74.5, 24.5, 77.5, 24.5, "findings")
    arrow(52.5, 24.5, 48.0, 24.5, "physics")
    arrow(21.5, 24.5, 27.0, 24.5, "triage")
    arrow(52.5, 49.0, 78.0, 21.0, "survival")
    ax.text(
        50,
        4.5,
        "Objective features only in every model matrix. Reported condition and ground truth are evaluation/audit evidence, never survival inputs.",
        ha="center",
        fontsize=7.4,
        color=HEX_GRAY,
        style="italic",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_dataset_figure(data: dict[str, Any], path: Path) -> Path:
    bridges = data["bridges"]
    truth = data["truth"]
    merged = bridges.merge(truth, on="BRIDGE_ID", how="left")
    climate = data["climate"]
    darbhanga = climate[climate["DISTRICT"].astype(str) == "Darbhanga"].sort_values("YEAR")

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8))
    ax = axes[0, 0]
    bins = np.arange(2.5, 10.5, 1.0)
    ax.hist(
        [merged.loc[merged["EVENT"] == 0, "REPORTED_LAST_MIN_COND"],
         merged.loc[merged["EVENT"] == 1, "REPORTED_LAST_MIN_COND"]],
        bins=bins,
        stacked=True,
        color=["#7FB3C8", "#C05621"],
        edgecolor="white",
        linewidth=0.6,
        label=["Censored", "Failure"],
    )
    ax.set_title("Reported minimum condition at last inspection", fontsize=9.5, color=HEX_NAVY)
    ax.set_xlabel("NBI condition rating (0-9)", fontsize=8)
    ax.set_ylabel("Bridges", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    _style_axes(ax)

    ax = axes[0, 1]
    ax.plot(darbhanga["YEAR"], darbhanga["MONSOON_ANOMALY"], color="#1F6F78", lw=1.1)
    floods = darbhanga[darbhanga["FLOOD_EXCEEDANCE"] == 1]
    ax.scatter(floods["YEAR"], floods["MONSOON_ANOMALY"], color="#C05621", s=14, zorder=3)
    ax.axhline(0.0, color="#A0AEC0", lw=0.7, ls="--")
    ax.set_title("Darbhanga monsoon anomaly and flood exceedances", fontsize=9.5, color=HEX_NAVY)
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Z-score", fontsize=8)
    _style_axes(ax)

    ax = axes[1, 0]
    material = bridges["STRUCTURE_KIND_MATERIAL"].astype(str).value_counts()
    ax.barh(material.index[::-1], material.values[::-1], color="#1F6F78", alpha=0.85)
    ax.set_title("Bridge material mix", fontsize=9.5, color=HEX_NAVY)
    ax.set_xlabel("Bridges", fontsize=8)
    _style_axes(ax)

    ax = axes[1, 1]
    design = bridges["STRUCTURE_TYPE_DESIGN"].astype(str).value_counts().head(8)
    ax.barh(design.index[::-1], design.values[::-1], color="#102A43", alpha=0.82)
    ax.set_title("Structural design mix", fontsize=9.5, color=HEX_NAVY)
    ax.set_xlabel("Bridges", fontsize=8)
    _style_axes(ax)

    fig.suptitle(
        "Synthetic NBI / Indian climate benchmark: cohort composition and exposure",
        fontsize=11,
        color=HEX_NAVY,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_model_figure(data: dict[str, Any], path: Path) -> Path:
    metrics = data["survival"]["models"]
    models = ["cox_time_varying", "random_survival_forest", "gradient_boosting", "ensemble"]
    labels = ["Cox TV", "RSF", "GBSA", "Ensemble"]
    colors = ["#102A43", "#1F6F78", "#C05621", "#6B46C1"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    for axis, metric, title in (
        (axes[0], "uno_ipcw_c", "Uno IPCW C-index"),
        (axes[1], "harrell_c", "Harrell C-index"),
    ):
        values_validation = [metrics[name]["validation"][metric] for name in models]
        values_test = [metrics[name]["test"][metric] for name in models]
        positions = np.arange(len(labels))
        width = 0.36
        axis.bar(positions - width / 2, values_validation, width, color=colors, alpha=0.55, label="Validation")
        axis.bar(positions + width / 2, values_test, width, color=colors, label="Test")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_ylim(0.65, 0.84)
        axis.set_title(title, fontsize=10, color=HEX_NAVY)
        axis.set_ylabel("C-index", fontsize=8)
        _style_axes(axis)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Survival discrimination: validation and held-out test",
        fontsize=11,
        color=HEX_NAVY,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_importance_figure(data: dict[str, Any], path: Path) -> Path:
    importance = data["survival"]["feature_importance"]["gradient_boosting"][:10]
    labels = [item["feature"] for item in importance][::-1]
    values = [item["importance"] for item in importance][::-1]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.barh(labels, values, color="#1F6F78", alpha=0.88)
    ax.set_title("Gradient-boosting split importance: top 10 objective features", fontsize=10.5, color=HEX_NAVY)
    ax.set_xlabel("Normalised split importance", fontsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_audit_figure(data: dict[str, Any], path: Path) -> Path:
    table = data["audit_table"].copy()
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    normal = table[~table["PAPER_INSPECTION_SUSPECT"].astype(bool)]
    suspect = table[table["PAPER_INSPECTION_SUSPECT"].astype(bool)]
    ax.scatter(
        normal["PHYSICS_EXPECTED_CONDITION"],
        normal["REPORTED_LAST_MIN_COND"],
        s=17,
        color="#7FB3C8",
        alpha=0.65,
        label="Routine audit record",
        edgecolor="none",
    )
    ax.scatter(
        suspect["PHYSICS_EXPECTED_CONDITION"],
        suspect["REPORTED_LAST_MIN_COND"],
        s=62,
        color="#C05621",
        label="Paper-inspection suspect",
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.axvline(6.12, color="#9B2C2C", lw=1.1, ls="--", label="Physics condition at S=0.68")
    ax.axhspan(6.5, 8.5, color="#C05621", alpha=0.08)
    ax.set_xlim(0, 9)
    ax.set_ylim(2.5, 9.5)
    ax.set_xlabel("Physics-implied condition = 9 × S(t0+5 | t0)", fontsize=8.5)
    ax.set_ylabel("Reported minimum condition", fontsize=8.5)
    ax.set_title(
        "Audit view: reported condition versus physics-implied condition",
        fontsize=10.5,
        color=HEX_NAVY,
    )
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_triage_figure(data: dict[str, Any], path: Path) -> Path:
    table = data["triage_table"].sort_values("TRIAGE_PRIORITY_SCORE", ascending=False).head(10)
    labels = table["BRIDGE_ID"].astype(str).to_numpy()[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    positions = np.arange(len(labels))
    axes[0].barh(
        positions - 0.18,
        table["BASELINE_SURVIVAL_AT_DECISION"].to_numpy()[::-1],
        0.36,
        color="#7FB3C8",
        label="Baseline",
    )
    axes[0].barh(
        positions + 0.18,
        table["POST_EVENT_SURVIVAL_AT_DECISION"].to_numpy()[::-1],
        0.36,
        color="#C05621",
        label="After shock",
    )
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].set_xlim(0, 1.0)
    axes[0].set_title("5-year conditional survival", fontsize=9.5, color=HEX_NAVY)
    axes[0].set_xlabel("Probability", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7)
    _style_axes(axes[0])

    axes[1].barh(
        positions - 0.18,
        table["RUL_BASELINE_YEARS"].to_numpy()[::-1],
        0.36,
        color="#7FB3C8",
        label="Baseline RUL",
    )
    axes[1].barh(
        positions + 0.18,
        table["RUL_POST_EVENT_YEARS"].to_numpy()[::-1],
        0.36,
        color="#C05621",
        label="Post-event RUL",
    )
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels([])
    axes[1].set_title("Remaining useful life (years)", fontsize=9.5, color=HEX_NAVY)
    axes[1].set_xlabel("Years until S = 0.45", fontsize=8)
    axes[1].legend(frameon=False, fontsize=7)
    _style_axes(axes[1])

    fig.suptitle(
        "Post-disaster triage: Darbhanga compound monsoon shock",
        fontsize=11,
        color=HEX_NAVY,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_figures(data: dict[str, Any]) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "architecture": make_architecture_figure(FIG_DIR / "architecture.png"),
        "dataset": make_dataset_figure(data, FIG_DIR / "dataset_overview.png"),
        "models": make_model_figure(data, FIG_DIR / "model_comparison.png"),
        "importance": make_importance_figure(data, FIG_DIR / "feature_importance.png"),
        "audit": make_audit_figure(data, FIG_DIR / "audit_scatter.png"),
        "triage": make_triage_figure(data, FIG_DIR / "triage_results.png"),
    }


# ---------------------------------------------------------------------------
# PDF styles and layout
# ---------------------------------------------------------------------------


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {}
    styles["CoverTitle"] = ParagraphStyle(
        "CoverTitle",
        parent=base["Title"],
        fontName="ReportSans-Bold",
        fontSize=28,
        leading=33,
        textColor=NAVY,
        spaceAfter=10,
        alignment=TA_LEFT,
    )
    styles["CoverSubtitle"] = ParagraphStyle(
        "CoverSubtitle",
        parent=base["Normal"],
        fontName="ReportSans",
        fontSize=13.5,
        leading=18,
        textColor=TEAL,
        spaceAfter=4,
    )
    styles["CoverMeta"] = ParagraphStyle(
        "CoverMeta",
        parent=base["Normal"],
        fontName="ReportSans",
        fontSize=9.4,
        leading=14,
        textColor=GRAY,
    )
    styles["H1"] = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontName="ReportSans-Bold",
        fontSize=17,
        leading=21,
        textColor=NAVY,
        spaceBefore=16,
        spaceAfter=7,
        keepWithNext=True,
    )
    styles["H2"] = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontName="ReportSans-Bold",
        fontSize=12.5,
        leading=16,
        textColor=TEAL,
        spaceBefore=11,
        spaceAfter=4,
        keepWithNext=True,
    )
    styles["H3"] = ParagraphStyle(
        "H3",
        parent=base["Heading3"],
        fontName="ReportSans-Bold",
        fontSize=10.4,
        leading=13.5,
        textColor=NAVY,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True,
    )
    styles["Body"] = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName="ReportSans",
        fontSize=9.9,
        leading=14.2,
        textColor=HexColor("#1A202C"),
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    styles["BodySmall"] = ParagraphStyle(
        "BodySmall",
        parent=styles["Body"],
        fontSize=8.6,
        leading=12.0,
        spaceAfter=4,
    )
    styles["Bullet"] = ParagraphStyle(
        "Bullet",
        parent=styles["Body"],
        leftIndent=13,
        bulletIndent=3,
        spaceAfter=3,
    )
    styles["Quote"] = ParagraphStyle(
        "Quote",
        parent=styles["Body"],
        fontName="ReportSans-Oblique",
        fontSize=9.5,
        leading=13.5,
        textColor=GRAY,
        leftIndent=12,
        rightIndent=12,
        spaceBefore=4,
        spaceAfter=7,
    )
    styles["Caption"] = ParagraphStyle(
        "Caption",
        parent=base["Normal"],
        fontName="ReportSans-Oblique",
        fontSize=8.2,
        leading=10.5,
        textColor=GRAY,
        alignment=TA_CENTER,
        spaceBefore=3,
        spaceAfter=8,
    )
    styles["TableHeader"] = ParagraphStyle(
        "TableHeader",
        parent=base["Normal"],
        fontName="ReportSans-Bold",
        fontSize=8.1,
        leading=10.0,
        textColor=colors.white,
        alignment=TA_LEFT,
        wordWrap="CJK",
    )
    styles["TableCell"] = ParagraphStyle(
        "TableCell",
        parent=base["Normal"],
        fontName="ReportSans",
        fontSize=7.9,
        leading=10.0,
        textColor=HexColor("#1A202C"),
        wordWrap="CJK",
    )
    styles["TOC0"] = ParagraphStyle(
        "TOC0",
        parent=base["Normal"],
        fontName="ReportSans-Bold",
        fontSize=9.8,
        leading=15,
        textColor=NAVY,
    )
    styles["TOC1"] = ParagraphStyle(
        "TOC1",
        parent=base["Normal"],
        fontName="ReportSans",
        fontSize=8.7,
        leading=12.5,
        leftIndent=14,
        textColor=GRAY,
    )
    return styles


class TechnicalReportTemplate(BaseDocTemplate):
    """Doc template with bookmarks, automatic TOC entries, headers, and footers."""

    def __init__(self, filename: str, **kwargs: Any) -> None:
        super().__init__(filename, **kwargs)
        self._toc_counter = 0

    def afterFlowable(self, flowable: Any) -> None:
        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        if style_name not in ("H1", "H2"):
            return
        level = 0 if style_name == "H1" else 1
        text = flowable.getPlainText()
        self.notify("TOCEntry", (level, text, self.page))
        self._toc_counter += 1
        key = f"section-{self._toc_counter}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)


def _draw_cover(canvas: Any, doc: Any) -> None:
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 34 * mm, width, 34 * mm, stroke=0, fill=1)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, width, 12 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.white)
    canvas.setFont("ReportSans-Bold", 9)
    canvas.drawString(18 * mm, height - 20 * mm, "ZERO-TRUST INFRASTRUCTURE ANALYTICS")
    canvas.setFont("ReportSans", 8)
    canvas.drawRightString(
        width - 18 * mm,
        height - 20 * mm,
        "Classical ML | Survival Analysis | Audit Verification",
    )
    canvas.setFillColor(colors.white)
    canvas.setFont("ReportSans", 8)
    canvas.drawString(
        18 * mm,
        5 * mm,
        "Technical design, data, model rationale, validation, and operational triage",
    )
    canvas.restoreState()


def _draw_later(canvas: Any, doc: Any) -> None:
    width, height = A4
    canvas.saveState()
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, height - 16 * mm, width - 18 * mm, height - 16 * mm)
    canvas.setFont("ReportSans", 7.6)
    canvas.setFillColor(GRAY)
    canvas.drawString(
        18 * mm,
        height - 13.5 * mm,
        "Zero-Trust Bridge Survival & Audit Verification Engine",
    )
    canvas.drawRightString(
        width - 18 * mm,
        height - 13.5 * mm,
        "Technical Report",
    )
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.drawString(18 * mm, 8.5 * mm, "Generated from persisted run artifacts.")
    canvas.drawRightString(width - 18 * mm, 8.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def P(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def bullets(items: Iterable[str], style: ParagraphStyle) -> list[Paragraph]:
    return [Paragraph(item, style, bulletText="\u2022") for item in items]


def make_table(
    rows: Sequence[Sequence[Any]],
    widths: Sequence[float],
    styles: dict[str, ParagraphStyle],
    *,
    header: bool = True,
    font_size: float | None = None,
) -> Table:
    rendered: list[list[Any]] = []
    for row_index, row in enumerate(rows):
        rendered_row: list[Any] = []
        for cell in row:
            if isinstance(cell, Paragraph):
                rendered_row.append(cell)
            else:
                style = styles["TableHeader"] if header and row_index == 0 else styles["TableCell"]
                if font_size is not None:
                    style = ParagraphStyle(
                        f"{style.name}-{font_size}",
                        parent=style,
                        fontSize=font_size,
                        leading=font_size + 2.2,
                    )
                rendered_row.append(Paragraph(str(cell), style))
        rendered.append(rendered_row)
    table = Table(rendered, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    command = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        command += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]
        for row_index in range(1, len(rendered)):
            if row_index % 2 == 0:
                command.append(("BACKGROUND", (0, row_index), (-1, row_index), LIGHT_GRAY))
    table.setStyle(TableStyle(command))
    return table


def figure_flowable(path: Path, caption: str, styles: dict[str, ParagraphStyle], width: float = 170 * mm) -> KeepTogether:
    with PILImage.open(path) as image:
        ratio = image.height / image.width
    image_flowable = Image(str(path), width=width, height=width * ratio)
    return KeepTogether(
        [Spacer(1, 3), image_flowable, P(caption, styles["Caption"])]
    )


# ---------------------------------------------------------------------------
# Report content
# ---------------------------------------------------------------------------


def build_story(data: dict[str, Any], styles: dict[str, ParagraphStyle], figures: dict[str, Path]) -> list[Any]:
    summary = data["summary"]
    pipeline = data["pipeline"]
    survival = data["survival"]
    audit = data["audit"]
    triage = data["triage"]
    bridges = data["bridges"]
    truth = data["truth"]
    inspections = data["inspections"]
    climate = data["climate"]

    test_metrics = survival["models"]["cox_time_varying"]["test"]
    validation_metrics = survival["models"]["cox_time_varying"]["validation"]
    model_selection = survival["model_selection"]
    audit_metrics = audit.get("ground_truth_evaluation") or {}
    triage_metrics = summary["module4"]["metrics"]
    event = triage["event"]
    config = summary.get("config", {})
    split = pipeline["split"]

    paper_rate = float(truth["PAPER_INSPECTION_FLAG"].mean())
    event_rate = float(bridges["EVENT"].mean())
    mean_duration = float(bridges["DURATION_YEARS"].mean())
    max_duration = int(bridges["DURATION_YEARS"].max())
    n_states = int(bridges["STATE"].nunique())
    n_districts = int(bridges["DISTRICT"].nunique())

    story: list[Any] = []

    # ------------------------------------------------------------------
    # Cover
    # ------------------------------------------------------------------
    story.append(Spacer(1, 42 * mm))
    story.append(P("ZERO-TRUST BRIDGE SURVIVAL & AUDIT VERIFICATION ENGINE", styles["CoverTitle"]))
    story.append(P("Climate-Adaptive Survival Analysis and Paper-Inspection Anomaly Detection", styles["CoverSubtitle"]))
    story.append(Spacer(1, 5 * mm))
    story.append(P("Technical Design, Data, Model Rationale, Validation, and Post-Disaster Triage Report", styles["CoverMeta"]))
    story.append(Spacer(1, 8 * mm))
    cover_table = make_table(
        [
            ["Document", "Technical report"],
            ["Project", "Zero-trust infrastructure reliability engine"],
            ["Version", "1.0"],
            ["Report date", date.today().isoformat()],
            ["Run seed", str(summary["seed"])],
            ["Training cohort", f"{summary['n_bridges']:,} synthetic bridges"],
            ["Selected survival model", survival["model_selection"]["selected_model"]],
            ["End-to-end status", summary["status"].upper()],
        ],
        [45 * mm, 95 * mm],
        styles,
        header=False,
        font_size=8.8,
    )
    story.append(cover_table)
    story.append(Spacer(1, 9 * mm))
    story.append(
        P(
            "<b>Abstract.</b> This report describes a complete classical machine-learning system for "
            "predicting climate-driven bridge degradation, auditing reported inspection claims, and "
            "prioritising physical non-destructive testing after a flood or cyclone. The system uses "
            "right-censored survival analysis, a time-varying Cox model, random survival forests, "
            "gradient-boosted survival trees, a robust audit residual, an IsolationForest, and "
            "SHAP-based local explanations. The current validated run uses a 1,200-bridge synthetic "
            "NBI/IBMS benchmark with Indian district-level monsoon, discharge, and scour shocks. The "
            "pipeline is deliberately leakage-safe: reported condition ratings and evaluation-only "
            "ground truth never enter the survival feature matrix.",
            styles["Body"],
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(
        P(
            "The report is written for engineers, data scientists, and reviewers who need to understand "
            "not only what the code does, but why each modelling and validation choice was made.",
            styles["Quote"],
        )
    )
    story.append(NextPageTemplate("Later"))
    story.append(PageBreak())

    # ------------------------------------------------------------------
    # Contents
    # ------------------------------------------------------------------
    story.append(P("Contents", ParagraphStyle("ContentsTitle", parent=styles["H1"], spaceBefore=0)))
    toc = TableOfContents()
    toc.levelStyles = [styles["TOC0"], styles["TOC1"]]
    toc.dotsMinLevel = 0
    story.append(toc)
    story.append(PageBreak())

    # ------------------------------------------------------------------
    # 1. Executive summary
    # ------------------------------------------------------------------
    story.append(P("1. Executive Summary", styles["H1"]))
    story.append(
        P(
            "The project addresses a narrow but consequential failure mode in bridge management: a "
            "structure can look acceptable on a paper inspection while its physical condition, "
            "hydraulic exposure, and loading history imply an elevated probability of failure. "
            "Conventional bridge management systems often reduce this problem to memoryless Markov "
            "transitions between condition states. That formulation cannot represent cumulative scour, "
            "monsoon shocks, or the divergence between a reported rating and objective deterioration. "
            "This system replaces that view with a right-censored, time-varying survival formulation "
            "and a separate audit layer that challenges the human claim.",
            styles["Body"],
        )
    )
    story.append(
        P(
            f"The validated run trains on {summary['n_bridges']:,} synthetic bridges drawn from an "
            f"NBI-style schema across {n_states} states and {n_districts} Indian districts. The cohort "
            f"contains {len(inspections):,} inspection records, {len(climate):,} district-year climate "
            f"records, a {event_rate:.1%} observed failure rate, and a {paper_rate:.1%} latent "
            f"paper-inspection rate. After bridge-level splitting, the training set contains "
            f"{split['train']['n_bridges']:,} bridges and {split['train']['n_intervals']:,} "
            f"Andersen-Gill intervals; validation and test contain {split['validation']['n_bridges']:,} "
            f"and {split['test']['n_bridges']:,} bridges respectively.",
            styles["Body"],
        )
    )
    story.append(
        P(
            f"The Cox time-varying model was selected using validation Uno IPCW concordance, not test "
            f"performance. It achieved a validation Uno C-index of {validation_metrics['uno_ipcw_c']:.4f} "
            f"and a held-out test Uno C-index of {test_metrics['uno_ipcw_c']:.4f}; Harrell's C-index was "
            f"{test_metrics['harrell_c']:.4f} and the Integrated Brier Score was "
            f"{test_metrics['integrated_brier_score']:.4f}. The audit layer produced "
            f"{audit['findings']['n_paper_inspection_suspect']} primary paper-inspection suspects and "
            f"{audit['findings']['n_audit_alert']} broader alerts on the held-out audit cohort, with an "
            f"Audit Z-score ROC-AUC of {audit_metrics['audit_z_score']['roc_auc']:.4f}.",
            styles["Body"],
        )
    )
    story.append(
        P(
            f"In the post-disaster demonstration, a compound monsoon event was applied to Darbhanga. "
            f"Of the {triage_metrics['n_scored']} test-split bridges in the district, "
            f"{triage_metrics['n_immediate_ndt']} received an immediate NDT recommendation, "
            f"{triage_metrics['priority_counts'].get('CRITICAL', 0)} were classified CRITICAL, and the "
            f"mean survival drop at five years was {triage_metrics['mean_survival_drop']:.3f}. The mean "
            f"RUL loss was {triage_metrics['mean_rul_loss_years']:.2f} years, with a maximum of "
            f"{triage_metrics['max_rul_loss_years']:.2f} years. SHAP attributions passed an additivity "
            f"check with a maximum error of {triage_metrics['shap']['max_additivity_error']:.2e}.",
            styles["Body"],
        )
    )
    story.append(
        P(
            "<b>Bottom line:</b> the project is operational end to end, reproducible from a fixed seed, "
            "and explicit about what is and is not validated. The current evidence is a controlled "
            "synthetic benchmark, not a claim of field-certified accuracy on the raw NBI/IBMS extract.",
            styles["Quote"],
        )
    )

    # ------------------------------------------------------------------
    # 2. Problem
    # ------------------------------------------------------------------
    story.append(P("2. Problem Context and Motivation", styles["H1"]))
    story.append(P("2.1 The hydraulic failure pathway", styles["H2"]))
    story.append(
        P(
            "A large share of Indian bridge failures is driven by hydraulic processes rather than by "
            "ordinary traffic wear alone. Monsoon rainfall raises discharge and flow velocity; high "
            "velocity increases bed and pier scour; repeated flood exceedances remove foundation "
            "support; and inadequate waterway area concentrates flow around substructure elements. "
            "The consequence is not a smooth linear loss of condition. A bridge may appear stable for "
            "years and then lose a large fraction of its substructure capacity in one or two extreme "
            "seasons. A model that ignores time-varying hydraulic exposure is therefore structurally "
            "mis-specified for this failure mode.",
            styles["Body"],
        )
    )
    story.append(
        P(
            "The synthetic generator encodes this asymmetry deliberately: substructure degradation has "
            "the highest hydraulic sensitivity, and a separate abrupt shock term is applied when a "
            "high-scour year coincides with a flood exceedance. This is why the survival engine includes "
            "both smooth age-related deterioration and point-in-time climate covariates.",
            styles["Body"],
        )
    )
    story.append(P("2.2 The paper-inspection problem", styles["H2"]))
    story.append(
        P(
            "Bridge condition ratings are usually produced by visual inspection. Visual inspection is "
            "useful, but it is subjective, intermittent, and vulnerable to optimistic bias. A bridge can "
            "be rubber-stamped as healthy while the inspector has not performed physical NDT, has not "
            "measured scour, or has not reconciled the claim with the district's rainfall and flood "
            "history. The audit problem is not to classify a bridge as safe or unsafe from the rating "
            "alone. It is to test whether the reported rating is statistically and physically consistent "
            "with objective evidence.",
            styles["Body"],
        )
    )
    story.append(P("2.3 Why memoryless Markov chains are insufficient", styles["H2"]))
    story.append(
        P(
            "A first-order Markov chain models the next condition state as a function of the current "
            "state only. It has no explicit representation of age, cumulative exposure, time-varying "
            "covariates, censoring, or the duration already spent in a state. It also assumes that a "
            "transition probability estimated from a historical average is adequate for a bridge that has "
            "just experienced an extreme flood. Those assumptions are precisely the ones that fail in a "
            "monsoon-driven portfolio. Survival analysis is a better fit because it models the hazard of "
            "failure over time, supports right-censoring, and allows the hazard to depend on a changing "
            "covariate vector X(t).",
            styles["Body"],
        )
    )

    # ------------------------------------------------------------------
    # 3. Objectives
    # ------------------------------------------------------------------
    story.append(P("3. Objectives and Success Criteria", styles["H1"]))
    story.append(
        P(
            "The system is designed around four linked engineering questions. Each question maps to one "
            "module and one artefact, so a reviewer can inspect the evidence at each boundary.",
            styles["Body"],
        )
    )
    story.append(
        Table(
            [
                [
                    P("Question", styles["TableHeader"]),
                    P("Module", styles["TableHeader"]),
                    P("Evidence produced", styles["TableHeader"]),
                ],
                [
                    P("How does failure risk evolve under time-varying monsoon and scour exposure?", styles["TableCell"]),
                    P("1 + 2", styles["TableCell"]),
                    P("Counting-process dataset, survival curves, hazard ratios, C-indices, IBS", styles["TableCell"]),
                ],
                [
                    P("Does a reported healthy condition contradict physics and climate history?", styles["TableCell"]),
                    P("3", styles["TableCell"]),
                    P("Physics-implied condition, Audit Z-score, IsolationForest flag, priority", styles["TableCell"]),
                ],
                [
                    P("After a disaster, which bridges lost the most remaining life and need NDT now?", styles["TableCell"]),
                    P("4", styles["TableCell"]),
                    P("Post-event survival, RUL, RUL loss, NDT recommendation", styles["TableCell"]),
                ],
                [
                    P("Why is a particular bridge flagged?", styles["TableCell"]),
                    P("4 + SHAP", styles["TableCell"]),
                    P("Local additive feature contributions and force plots", styles["TableCell"]),
                ],
            ],
            [72 * mm, 20 * mm, 76 * mm],
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            ),
        )
    )
    story.append(Spacer(1, 3))
    story.append(P("Success criteria used in this implementation:", styles["H3"]))
    story.extend(
        bullets(
            [
                "The survival target must be right-censored and constructed from the minimum of deck, superstructure, and substructure condition.",
                "No reported condition, latent truth, paper-inspection label, or event time may enter a survival model matrix.",
                "Train, validation, and test bridges must be disjoint before any imputation or encoding is fitted.",
                "Performance must be reported with survival-specific metrics, never classification accuracy or standard RMSE.",
                "The audit layer must produce a holdout finding table and a post-hoc evaluation that is clearly separated from scoring.",
                "The post-disaster layer must recompute a survival curve, not merely re-rank a static score.",
                "Every local explanation must be auditable for feature provenance and additive consistency.",
            ],
            styles["Bullet"],
        )
    )

    # ------------------------------------------------------------------
    # 4. Architecture
    # ------------------------------------------------------------------
    story.append(P("4. System Architecture", styles["H1"]))
    story.append(
        P(
            "The architecture is a staged evidence pipeline. Module 1 produces model-ready longitudinal "
            "data. Module 2 estimates survival and risk. Module 3 compares the physics view with the "
            "human inspection claim. Module 4 updates the physical state after a disaster and converts "
            "the new survival curve into an operational NDT priority. The modules communicate through "
            "typed dataclasses and persisted artefacts rather than through hidden global state.",
            styles["Body"],
        )
    )
    story.append(figure_flowable(figures["architecture"], "Figure 1. End-to-end module architecture and evidence flow.", styles, width=175 * mm))
    story.append(P("4.1 Module boundaries", styles["H2"]))
    story.append(
        make_table(
            [
                ["Module", "Primary inputs", "Primary outputs", "Key guardrail"],
                ["1. Data pipeline", "Synthetic NBI/IBMS registry, district climate panel, inspection history", "Static and counting-process splits, sksurv targets, audit report", "Bridge-level split before imputation; feature allowlist"],
                ["2. Survival engine", "Preprocessed static and time-varying matrices", "Cox TV, RSF, GBSA, ensemble, metrics, hazard ratios", "Survival metrics only; test set not used for selection"],
                ["3. Audit verifier", "Survival model, objective bridge metadata, latest climate snapshot, reported condition", "Audit findings, Z-score, IsolationForest flag, priority", "Reported condition allowed only in audit layer, not survival"],
                ["4. Triage and XAI", "Fitted survival models, post-disaster climate shock, audit findings", "Survival drop, RUL loss, NDT priority, SHAP force plots", "SHAP features checked against truth/audit prefixes"],
            ],
            [25 * mm, 53 * mm, 54 * mm, 36 * mm],
            styles,
            font_size=7.5,
        )
    )
    story.append(Spacer(1, 4))
    story.append(P("4.2 The leakage boundary", styles["H2"]))
    story.append(
        P(
            "The single most important architectural decision is the leakage boundary. The survival view "
            "sees only objective features that would be available before or at the prediction time: "
            "structural attributes, traffic loading, pre-entry climate history, and current climate "
            "observations. It never sees the reported condition rating, the latent true condition, the "
            "paper-inspection label, the event indicator, the duration, or the inspector bias. The audit "
            "view is the only place where the reported rating is an input, because challenging that "
            "rating is the audit's purpose. Ground truth is accepted only after scoring, for evaluation.",
            styles["Body"],
        )
    )

    # ------------------------------------------------------------------
    # 5. Data
    # ------------------------------------------------------------------
    story.append(P("5. Data Foundation", styles["H1"]))
    story.append(P("5.1 What the model is trained on", styles["H2"]))
    story.append(
        P(
            f"The current validated run trains on a synthetic longitudinal benchmark generated to match "
            f"the schema and failure physics of the US National Bridge Inventory (NBI) and an Indian "
            f"Bridge Management System (IBMS) style portfolio. It is not a raw NBI extract. The generator "
            f"exists so that the full pipeline can be tested before a multi-gigabyte raw inventory is "
            f"cleaned, and so that the paper-inspection ground truth is known and controllable for "
            f"evaluation. The training cohort contains {summary['n_bridges']:,} bridges, "
            f"{len(inspections):,} inspection records, {len(climate):,} district-year climate records, "
            f"and {split['train']['n_intervals']:,} training intervals.",
            styles["Body"],
        )
    )
    story.append(
        make_table(
            [
                ["Cohort attribute", "Value"],
                ["Bridges", f"{summary['n_bridges']:,}"],
                ["States / districts", f"{n_states} / {n_districts}"],
                ["Build years", f"{int(bridges['YEAR_BUILT'].min())} – {int(bridges['YEAR_BUILT'].max())}"],
                ["Study end year", "2024"],
                ["Climate panel", f"{len(climate):,} district-year rows, {climate['YEAR'].min()} – {climate['YEAR'].max()}"],
                ["Inspection records", f"{len(inspections):,}"],
                ["Observed failure rate", f"{event_rate:.2%}"],
                ["Latent paper-inspection rate", f"{paper_rate:.2%}"],
                ["Mean observed duration", f"{mean_duration:.2f} years"],
                ["Maximum observed duration", f"{max_duration} years"],
                ["Train / validation / test bridges", f"{split['train']['n_bridges']:,} / {split['validation']['n_bridges']:,} / {split['test']['n_bridges']:,}"],
                ["Train / validation / test intervals", f"{split['train']['n_intervals']:,} / {split['validation']['n_intervals']:,} / {split['test']['n_intervals']:,}"],
            ],
            [65 * mm, 100 * mm],
            styles,
        )
    )
    story.append(figure_flowable(figures["dataset"], "Figure 2. Cohort composition, reported condition distribution, climate exposure, and structural mix.", styles, width=172 * mm))

    story.append(P("5.2 Static structural schema", styles["H2"]))
    story.append(
        P(
            "Every bridge has an NBI-style static record. The model uses geography (state, district, "
            "latitude, longitude, flood proneness), age (year built), structural form (material, design, "
            "number of spans, maximum span, deck length, roadway width), hydraulic context (waterway "
            "adequacy, scour-critical flag), loading (ADT, truck percentage, design load, traffic growth), "
            "and ownership. Pre-entry climate aggregates are computed from the five years strictly before "
            "the bridge enters observation, so the static survival model cannot see future rainfall.",
            styles["Body"],
        )
    )
    story.append(
        make_table(
            [
                ["Group", "Fields used by the pipeline"],
                ["Identity and geography", "BRIDGE_ID, STATE, DISTRICT, LATITUDE, LONGITUDE, FLOOD_PRONENESS"],
                ["Structure", "YEAR_BUILT, STRUCTURE_KIND_MATERIAL, STRUCTURE_TYPE_DESIGN, MAX_SPAN_LEN, NUM_SPANS, DECK_LEN, ROADWAY_WIDTH"],
                ["Hydraulics", "WATERWAY_ADEQUACY, SCOUR_CRITICAL"],
                ["Loading and ownership", "ADT_TRAFFIC, TRUCK_ADT_PCT, DESIGN_LOAD, TRAFFIC_GROWTH_RATE, OWNER"],
                ["Pre-entry climate", "PRE_ENTRY_MONSOON_MEAN_5Y, PRE_ENTRY_MONSOON_MAX_5Y, PRE_ENTRY_SCOUR_MAX_5Y, PRE_ENTRY_FLOOD_EXCEEDANCES_5Y, PRE_ENTRY_RAINFALL_MEAN_5Y"],
            ],
            [38 * mm, 127 * mm],
            styles,
            font_size=7.7,
        )
    )
    story.append(P("5.3 Dynamic climate and loading schema", styles["H2"]))
    story.append(
        P(
            "The counting-process view adds year-specific covariates. Age and current traffic are carried "
            "forward from the static record. The climate panel supplies a standardized monsoon anomaly, "
            "seasonal rainfall, peak discharge, peak velocity, a peak scour index, a binary flood "
            "exceedance, and two cumulative counters: cumulative flood exceedances and cumulative scour "
            "load. Point-in-time joins use the calendar year at the end of each interval. Cumulative "
            "counters are constructed causally and are therefore available at the decision time.",
            styles["Body"],
        )
    )
    story.append(
        make_table(
            [
                ["Dynamic field", "Definition"],
                ["AGE_YEARS", "Age at the end of the interval"],
                ["ADT_CURRENT / TRUCK_ADT_PCT_CURRENT", "Traffic loading at the interval end"],
                ["MONSOON_ANOMALY", "District monsoon rainfall Z-score relative to the historical reference period"],
                ["SEASONAL_RAINFALL_MM", "Seasonal rainfall total"],
                ["PEAK_DISCHARGE_M3S / PEAK_VELOCITY_MPS", "Hydrological forcing proxies"],
                ["PEAK_SCOUR_INDEX", "Discharge × velocity / 1000, a relative scour forcing index"],
                ["FLOOD_EXCEEDANCE", "Binary indicator that district discharge exceeded its 90th percentile"],
                ["CUMULATIVE_FLOOD_EXCEEDANCE", "Causal count of flood exceedances to the interval end"],
                ["CUMULATIVE_SCOUR_LOAD", "Causal sum of peak scour index to the interval end"],
            ],
            [52 * mm, 113 * mm],
            styles,
            font_size=7.7,
        )
    )

    story.append(P("5.4 How the synthetic deterioration process works", styles["H2"]))
    story.append(
        P(
            "The generator is not a random-label simulator. It constructs a latent physical condition "
            "trajectory for deck, superstructure, and substructure. The annual decrement is driven by a "
            "base rate, material and design factors, traffic multiplier, waterway multiplier, span "
            "multiplier, a bridge-level frailty term, an age factor, and a climate multiplier. The "
            "climate multiplier is exponential in hydraulic exposure, with the substructure receiving the "
            "largest sensitivity. An additional abrupt shock is applied to the substructure when high "
            "scour coincides with flood exceedance. Reported ratings are generated from the latent "
            "condition plus an inspector bias; a configured fraction of bridges receives a large "
            "optimistic bias, which creates the paper-inspection signal.",
            styles["Body"],
        )
    )
    story.append(
        P(
            "In compact form, the latent condition decrement for component c and bridge i in year t is:",
            styles["Body"],
        )
    )
    story.append(
        P(
            "<b>ΔC<sub>i,c,t</sub> = base<sub>c</sub> × material<sub>i</sub> × design<sub>i</sub> × "
            "traffic<sub>i</sub> × waterway<sub>i</sub> × span<sub>i</sub> × frailty<sub>i</sub> × "
            "age(t)<sup>0.7</sup> × exp(s<sub>c</sub> × hydraulic_exposure<sub>i,t</sub>) × noise</b>",
            ParagraphStyle("Equation", parent=styles["Body"], alignment=TA_CENTER, fontName="ReportSans", textColor=NAVY),
        )
    )
    story.append(
        P(
            "where hydraulic exposure combines positive monsoon anomaly, scour above a threshold, "
            "cumulative flood count, and cyclone occurrence. The event is defined when the minimum of "
            "deck, superstructure, and substructure latent condition first reaches 4.0 or lower. "
            "Inspection timing is discrete: a bridge is observed at one- or two-year intervals, so the "
            "recorded event time is the first inspection at or after the latent crossing. If no crossing "
            "occurs before the study end, the record is right-censored.",
            styles["Body"],
        )
    )
    story.append(P("5.5 Climate shock simulation", styles["H2"]))
    story.append(
        P(
            "The climate panel is generated for 25 districts and 87 years (1938–2024). State-level "
            "monsoon anomalies follow an AR(1) process with district-level residuals, and known extreme "
            "years receive additional boost. Rainfall, cyclone probability, discharge, velocity, scour, "
            "and flood exceedance are then generated conditionally. This produces realistic clustering: "
            "flood years are not independent coin flips, and high-rainfall years also tend to have high "
            "discharge and scour. The result is a benchmark where time-varying hydraulic covariates carry "
            "genuine predictive information rather than noise.",
            styles["Body"],
        )
    )
    story.append(P("5.6 Train, validation, and test design", styles["H2"]))
    story.append(
        P(
            "Splitting is performed at the bridge level before any preprocessing. That prevents the same "
            "bridge's longitudinal records from appearing in both training and evaluation. The split is "
            "stratified on the event indicator, so each partition contains a comparable failure rate. "
            "Validation is used for model selection; test is touched only once for final reporting. The "
            "audit verifier is calibrated on the training split and evaluated on validation and test rows. "
            "The post-disaster demonstration scores the test split only, because that is the honest "
            "holdout view.",
            styles["Body"],
        )
    )
    story.append(
        make_table(
            [
                ["Split", "Bridges", "Intervals", "Events", "Event rate"],
                ["Train", f"{split['train']['n_bridges']:,}", f"{split['train']['n_intervals']:,}", f"{split['train']['n_events']:,}", f"{split['train']['event_rate']:.2%}"],
                ["Validation", f"{split['validation']['n_bridges']:,}", f"{split['validation']['n_intervals']:,}", f"{split['validation']['n_events']:,}", f"{split['validation']['event_rate']:.2%}"],
                ["Test", f"{split['test']['n_bridges']:,}", f"{split['test']['n_intervals']:,}", f"{split['test']['n_events']:,}", f"{split['test']['event_rate']:.2%}"],
            ],
            [28 * mm, 30 * mm, 35 * mm, 30 * mm, 32 * mm],
            styles,
        )
    )

    # ------------------------------------------------------------------
    # 6. Module 1
    # ------------------------------------------------------------------
    story.append(P("6. Module 1: Longitudinal and Dynamic Climate Pipeline", styles["H1"]))
    story.append(P("6.1 Count-process construction", styles["H2"]))
    story.append(
        P(
            "Each bridge record has a duration from entry to the first observed failure or censoring. "
            "The Andersen-Gill splitter converts that record into one interval per observed year: "
            "(id, t_start, t_stop, event, X_t). The event flag is one only on the final interval of a "
            "bridge that failed, and zero on every earlier interval and on all intervals of a censored "
            "bridge. This is the canonical counting-process representation used by "
            "lifelines.CoxTimeVaryingFitter. The dynamic covariates attached to an interval are the "
            "values at the interval end, not future values.",
            styles["Body"],
        )
    )
    story.append(
        P(
            "The splitter enforces t_start < t_stop, non-negative duration, no duplicate intervals, "
            "contiguous coverage, and an event count that matches the bridge-level target. The "
            "pipeline additionally verifies that the count of intervals per bridge equals its integer "
            "duration and that events appear only in the final interval.",
            styles["Body"],
        )
    )
    story.append(P("6.2 Preprocessing and imputation", styles["H2"]))
    story.append(
        P(
            "Missing values are handled with IterativeImputer, which models each incomplete feature as a "
            "function of the others. Median imputation is simpler but would destroy relationships between "
            "correlated structural and climate variables. The imputer adds missingness indicators so the "
            "model can distinguish an imputed value from a genuinely observed one. Categorical fields are "
            "one-hot encoded with an explicit missing category, and numeric fields are standardized. "
            "The imputer, encoder, and scaler are fitted only on the training split and then applied "
            "unchanged to validation and test.",
            styles["Body"],
        )
    )
    story.append(
        make_table(
            [
                ["Preprocessing property", "Value in validated run"],
                ["Static input features", f"{pipeline['feature_counts']['static_input']}"],
                ["Static encoded features", f"{pipeline['feature_counts']['static_encoded']}"],
                ["Time-varying input features", f"{pipeline['feature_counts']['time_varying_input']}"],
                ["Time-varying encoded features", f"{pipeline['feature_counts']['time_varying_encoded']}"],
                ["Imputer", "IterativeImputer, maximum 8 iterations, fitted on training rows"],
                ["Categorical handling", "OneHotEncoder with explicit missing category"],
                ["Numeric handling", "StandardScaler after imputation"],
                ["Maximum imputer fit rows", "50,000 (caps memory on large raw extracts)"],
            ],
            [62 * mm, 103 * mm],
            styles,
        )
    )
    story.append(P("6.3 Survival targets", styles["H2"]))
    story.append(
        P(
            "Two target representations are constructed from the same raw records. For scikit-survival, "
            "the target is a structured NumPy array built with sksurv.util.Surv.from_arrays(event=..., "
            "time=...), with dtype [('event', bool), ('time', float)]. For lifelines, the target is the "
            "counting-process frame described above. Having both representations is not redundant: the "
            "Cox model needs the interval form, while the tree models need one row per bridge and the "
            "censored event time.",
            styles["Body"],
        )
    )
    story.append(P("6.4 Verification checks", styles["H2"]))
    story.append(
        P(
            "The pipeline executes assertions rather than optional diagnostics. The validated run reports "
            "the following checks, all true:",
            styles["Body"],
        )
    )
    story.append(
        make_table(
            [["Check", "Status"]] + [[key.replace("_", " ").capitalize(), "PASS" if value else "FAIL"] for key, value in pipeline["checks"].items()],
            [130 * mm, 35 * mm],
            styles,
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        P(
            "The pre-entry climate check deserves emphasis. The five-year pre-entry aggregates are "
            "reconstructed from district records strictly before ENTRY_YEAR. A test verifies the "
            "reconstructed mean against the raw panel. This prevents a subtle but fatal leak where a "
            "static feature accidentally includes the first observed year or later.",
            styles["Body"],
        )
    )

    # ------------------------------------------------------------------
    # 7. Module 2
    # ------------------------------------------------------------------
    story.append(P("7. Module 2: Survival Engine and Model Selection", styles["H1"]))
    story.append(P("7.1 Cox time-varying model", styles["H2"]))
    story.append(
        P(
            "The primary model is a penalised Andersen-Gill Cox model fitted with "
            "lifelines.CoxTimeVaryingFitter. The hazard at time t is h(t | X(t)) = h0(t) exp(βᵀX(t)), "
            "where X(t) includes the static structural features and the year-specific climate and "
            "loading covariates. The model estimates a single global coefficient vector while allowing "
            "the covariate vector to change over time. It handles right-censoring and staggered entry "
            "naturally, and it produces interpretable hazard ratios.",
            styles["Body"],
        )
    )
    story.append(
        P(
            "The fit uses a small elastic-net penalty to stabilise the high-dimensional one-hot design. "
            "Near-constant features and highly collinear pairs (absolute correlation above 0.995) are "
            "removed before fitting. In the validated run, 99 input features entered the Cox view and 88 "
            "were retained. The fitted model converged with an effective penalizer of 0.01.",
            styles["Body"],
        )
    )
    story.append(P("7.2 Random survival forest", styles["H2"]))
    story.append(
        P(
            "The random survival forest (RSF) is an ensemble of survival trees that split nodes to "
            "maximise a log-rank statistic. It can represent non-linear interactions and threshold "
            "effects that a single global Cox coefficient cannot, such as a high truck share only being "
            "dangerous in combination with poor waterway adequacy and severe scour. The validated run "
            "uses 120 trees, log-rank splitting, a minimum leaf size of 3, and square-root feature "
            "sampling.",
            styles["Body"],
        )
    )
    story.append(P("7.3 Gradient-boosted survival analysis", styles["H2"]))
    story.append(
        P(
            "GradientBoostingSurvivalAnalysis fits a stage-wise additive ensemble using the Cox partial "
            "likelihood loss. Each stage fits a shallow regression tree to the negative gradient of the "
            "loss. This gives a smooth, regularised non-linear risk function. The validated run uses 100 "
            "stages, a learning rate of 0.05, depth 3, and 80% subsampling. The shallowness is deliberate: "
            "with a few thousand bridges, deep boosted trees would overfit interactions that are not "
            "supported by the data.",
            styles["Body"],
        )
    )
    story.append(P("7.4 Ensemble", styles["H2"]))
    story.append(
        P(
            "The ensemble combines the RSF and gradient-boosting risk scores by rank averaging. Rank "
            "averaging avoids the problem that the two models produce risks on different scales. It is "
            "appropriate for concordance-based evaluation, where only ordering matters. For pointwise "
            "SHAP attribution, however, a batch-dependent rank score is not stable. Module 4 therefore "
            "uses a fixed background-calibrated z-score average of the two component risks for SHAP. "
            "This is an explicit and documented distinction, not an accidental inconsistency.",
            styles["Body"],
        )
    )
    story.append(P("7.5 Selection protocol", styles["H2"]))
    story.append(
        P(
            "The four models are compared on the validation split using Uno's IPCW C-index, with "
            "Harrell's C-index as a fallback. The test split is never used to choose the model. This "
            "matters because the gradient-boosting model happens to have a slightly higher test Uno "
            "C-index than the Cox model in this run. Selecting it on test performance would be leakage. "
            "The Cox model is selected because it has the highest validation Uno C-index "
            f"({model_selection['validation_scores']['cox_time_varying']:.4f}) and because its "
            "time-varying hazard ratios directly answer the engineering question about monsoon and scour "
            "acceleration.",
            styles["Body"],
        )
    )
    story.append(figure_flowable(figures["models"], "Figure 3. Validation and test discrimination for all four survival models.", styles, width=170 * mm))
    story.append(
        make_table(
            [
                ["Model", "Val Harrell", "Val Uno", "Val IBS", "Test Harrell", "Test Uno", "Test IBS"],
                ["Cox time-varying", f"{survival['models']['cox_time_varying']['validation']['harrell_c']:.4f}", f"{survival['models']['cox_time_varying']['validation']['uno_ipcw_c']:.4f}", f"{survival['models']['cox_time_varying']['validation']['integrated_brier_score']:.4f}", f"{survival['models']['cox_time_varying']['test']['harrell_c']:.4f}", f"{survival['models']['cox_time_varying']['test']['uno_ipcw_c']:.4f}", f"{survival['models']['cox_time_varying']['test']['integrated_brier_score']:.4f}"],
                ["Random survival forest", f"{survival['models']['random_survival_forest']['validation']['harrell_c']:.4f}", f"{survival['models']['random_survival_forest']['validation']['uno_ipcw_c']:.4f}", f"{survival['models']['random_survival_forest']['validation']['integrated_brier_score']:.4f}", f"{survival['models']['random_survival_forest']['test']['harrell_c']:.4f}", f"{survival['models']['random_survival_forest']['test']['uno_ipcw_c']:.4f}", f"{survival['models']['random_survival_forest']['test']['integrated_brier_score']:.4f}"],
                ["Gradient boosting", f"{survival['models']['gradient_boosting']['validation']['harrell_c']:.4f}", f"{survival['models']['gradient_boosting']['validation']['uno_ipcw_c']:.4f}", f"{survival['models']['gradient_boosting']['validation']['integrated_brier_score']:.4f}", f"{survival['models']['gradient_boosting']['test']['harrell_c']:.4f}", f"{survival['models']['gradient_boosting']['test']['uno_ipcw_c']:.4f}", f"{survival['models']['gradient_boosting']['test']['integrated_brier_score']:.4f}"],
                ["Ensemble", f"{survival['models']['ensemble']['validation']['harrell_c']:.4f}", f"{survival['models']['ensemble']['validation']['uno_ipcw_c']:.4f}", f"{survival['models']['ensemble']['validation']['integrated_brier_score']:.4f}", f"{survival['models']['ensemble']['test']['harrell_c']:.4f}", f"{survival['models']['ensemble']['test']['uno_ipcw_c']:.4f}", f"{survival['models']['ensemble']['test']['integrated_brier_score']:.4f}"],
            ],
            [34 * mm, 22 * mm, 21 * mm, 20 * mm, 22 * mm, 21 * mm, 20 * mm],
            styles,
            font_size=7.3,
        )
    )
    story.append(Spacer(1, 4))
    story.append(P("7.6 Survival metrics and why they are used", styles["H2"]))
    story.append(
        P(
            "Harrell's concordance index measures the probability that a model ranks a randomly chosen "
            "pair of comparable subjects correctly. It is intuitive but can be biased when censoring is "
            "high because it does not reweight censored observations. Uno's IPCW C-index applies inverse "
            "probability of censoring weights, making it more appropriate for a portfolio where many "
            "bridges are censored at the study end. The Brier score measures the squared error of "
            "predicted survival probabilities at a fixed horizon; the Integrated Brier Score integrates "
            "that error across the requested horizons. These metrics answer different questions: "
            "discrimination versus calibration. Reporting all three prevents a model with good ordering "
            "but poor absolute probabilities from looking better than it is.",
            styles["Body"],
        )
    )
    story.append(P("7.7 What the hazard ratios say", styles["H2"]))
    cox = survival["cox_time_varying"]
    story.append(
        P(
            "The fitted Cox model is consistent with the intended failure physics. Timber has the highest "
            "material hazard ratio in the top coefficient table (2.64 relative to the reference), "
            "followed by steel (1.95); prestressed concrete (0.61) and culverts (0.48) are comparatively "
            "protective. Truss design has a hazard ratio of 1.61. These are associational estimates from "
            "the synthetic benchmark, not causal field estimates, but they confirm that the model is "
            "learning the encoded structural priors rather than a spurious label pattern.",
            styles["Body"],
        )
    )
    acceleration = cox["monsoon_acceleration"]
    one_sd = acceleration["hazard_ratio_per_1sd"]
    two_sd = acceleration["hazard_ratio_per_2sd"]
    story.append(
        make_table(
            [
                ["Climate shock quantity", "Estimated hazard ratio"],
                ["+1 SD monsoon anomaly", f"{one_sd['MONSOON_ANOMALY']:.4f}"],
                ["+2 SD monsoon anomaly", f"{two_sd['MONSOON_ANOMALY']:.4f}"],
                ["+1 SD cumulative flood exceedance", f"{one_sd['CUMULATIVE_FLOOD_EXCEEDANCE']:.4f}"],
                ["+2 SD cumulative flood exceedance", f"{two_sd['CUMULATIVE_FLOOD_EXCEEDANCE']:.4f}"],
                ["+1 SD cumulative scour load", f"{one_sd['CUMULATIVE_SCOUR_LOAD']:.4f}"],
                ["+2 SD cumulative scour load", f"{two_sd['CUMULATIVE_SCOUR_LOAD']:.4f}"],
                ["Combined +1 SD monsoon + flood + scour", f"{acceleration['combined_1sd_shock_hazard_ratio']:.4f}"],
                ["Combined +2 SD monsoon + flood + scour", f"{acceleration['combined_extreme_shock_hazard_ratio']:.4f}"],
            ],
            [105 * mm, 55 * mm],
            styles,
        )
    )
    story.append(Spacer(1, 3))
    story.append(
        P(
            "All climate covariates are standardised on the training split, so one model unit is one "
            "physical standard deviation. Reporting +1 SD and +2 SD removes the unit inconsistency of "
            "the earlier absolute-increment table, where '+40 cumulative scour load' was compared with a "
            "single flood exceedance. The composite shock exponentiates the summed scaled linear "
            "predictors, exp(sum_i beta_i * delta_i), so it is never smaller than its positive individual "
            "components. Cumulative exposure remains more dangerous than a single wet year, which is why "
            "the post-disaster Module 4 shock adds to the cumulative counters.",
            styles["Body"],
        )
    )
    story.append(figure_flowable(figures["importance"], "Figure 4. Gradient-boosting feature importance. Span, age, material, waterway adequacy, and pre-entry rainfall dominate the static risk view.", styles, width=165 * mm))

    # ------------------------------------------------------------------
    # 8. Module 3
    # ------------------------------------------------------------------
    story.append(P("8. Module 3: Paper-Inspection Audit Verifier", styles["H1"]))
    story.append(P("8.1 Conditional survival is the correct audit quantity", styles["H2"]))
    story.append(
        P(
            "An old bridge may have a low unconditional survival probability simply because it has already "
            "survived a long time. That is not evidence of a false inspection claim. The audit therefore "
            "uses conditional survival over the next h years: S(t0 + h | t0, X(t0)) = S(t0 + h) / S(t0). "
            "Here t0 is the last observation time and h is the audit horizon, set to five years. The "
            "physics-implied condition is the monotone proxy 9 × S(t0 + h | t0), which places survival on "
            "the same 0–9 scale as the reported NBI rating.",
            styles["Body"],
        )
    )
    story.append(
        P(
            "The primary flag is deliberately explicit: a bridge is a PAPER_INSPECTION_SUSPECT when the "
            "inspector reports a healthy condition of 7–8 while the physics-implied conditional survival "
            "is below 0.68. The reported condition is used only in the audit table; it never enters the "
            "survival model that produces the physics view.",
            styles["Body"],
        )
    )
    story.append(P("8.2 Two complementary anomaly detectors", styles["H2"]))
    story.append(
        P(
            "The audit uses a robust residual standardiser and an IsolationForest. The residual is "
            "reported condition minus physics-implied condition. The centre and scale are the median and "
            "1.4826 × median absolute deviation fitted on the training split, producing a one-sided "
            "Audit Z-score. The median/MAD approach is chosen because the residual distribution is "
            "heavy-tailed: a handful of paper inspections should not move the calibration. In the "
            f"validated run the calibration centre was {audit['detector']['residual_calibration']['residual_center']:.3f} "
            f"and the scale was {audit['detector']['residual_calibration']['residual_scale']:.3f} "
            f"over {audit['detector']['residual_calibration']['n_rows']:,} training rows.",
            styles["Body"],
        )
    )
    story.append(
        P(
            "The IsolationForest is fitted on objective climate and structural features plus the physics "
            "scores and the reported condition. It therefore flags internally inconsistent audit records, "
            "not merely unusual bridges. It is unsupervised because the real world does not provide a "
            "reliable label for falsified inspections; the synthetic paper flag is evaluation-only. The "
            "audit also defines a broader AUDIT_ALERT that combines the primary flag, the statistical "
            "Z-residual rule, and the IsolationForest rule.",
            styles["Body"],
        )
    )
    story.append(
        make_table(
            [
                ["Rule", "Definition", "Purpose"],
                ["PAPER_INSPECTION_SUSPECT", "Reported condition 7–8 AND conditional S(t0+5) < 0.68", "High-recall primary finding"],
                ["Statistical alert", "Reported condition ≥ 7 AND condition gap ≥ 0.5 AND Z ≥ 1.5", "Operational screening set"],
                ["Unsupervised alert", "Reported condition ≥ 7 AND IsolationForest flag AND gap ≥ 0.5", "Catch unusual multivariate audit records"],
                ["AUDIT_ALERT", "Any of the above", "Broader screening set"],
            ],
            [33 * mm, 82 * mm, 54 * mm],
            styles,
            font_size=7.5,
        )
    )
    story.append(P("8.3 Audit results", styles["H2"]))
    story.append(
        P(
            f"On the {audit['cohort']['n_audited']} validation and test bridges, "
            f"{audit['findings']['n_reported_healthy']} reported a healthy condition and "
            f"{audit['findings']['n_paper_inspection_suspect']} met the primary suspect rule. The broader "
            f"alert flagged {audit['findings']['n_audit_alert']} bridges, and the IsolationForest flagged "
            f"{audit['findings']['n_isolation_anomalies']}. Against the evaluation-only paper-inspection "
            f"label, the Audit Z-score achieved ROC-AUC {audit_metrics['audit_z_score']['roc_auc']:.4f} "
            f"and average precision {audit_metrics['audit_z_score']['average_precision']:.4f}. The "
            f"primary rule had precision {audit_metrics['PAPER_INSPECTION_SUSPECT']['precision']:.3f} "
            f"and recall {audit_metrics['PAPER_INSPECTION_SUSPECT']['recall']:.3f}; the broader alert "
            f"had precision {audit_metrics['AUDIT_ALERT']['precision']:.3f} and recall "
            f"{audit_metrics['AUDIT_ALERT']['recall']:.3f}.",
            styles["Body"],
        )
    )
    story.append(
        P(
            "The low recall of the primary rule is not a coding defect. It is the direct consequence of "
            "the stated operating point: a bridge must be both reported healthy (7–8) and below the "
            "critical survival threshold. A screening system that flags every old bridge is not useful to "
            "an inspection team. The correct interpretation is that the primary rule is a high-precision "
            "escalation list, while the broader alert and the continuous Z-score are used for ranking and "
            "review. If operational policy prefers recall, the threshold can be raised to 0.55 or the "
            "Z-threshold lowered; the report makes that trade-off explicit rather than hiding it.",
            styles["Body"],
        )
    )
    story.append(figure_flowable(figures["audit"], "Figure 5. Reported condition against physics-implied condition. The shaded healthy-claim band and the vertical S=0.68 line define the paper-inspection suspect region.", styles, width=160 * mm))

    # ------------------------------------------------------------------
    # 9. Module 4
    # ------------------------------------------------------------------
    story.append(P("9. Module 4: Post-Disaster Triage and Explainability", styles["H1"]))
    story.append(P("9.1 Event representation", styles["H2"]))
    story.append(
        P(
            "A disaster event is a typed object with a target scope (district, state, or explicit bridge "
            "IDs), an event type, a severity multiplier, and explicit changes to the climate vector. The "
            "event can increase the monsoon anomaly, scale rainfall, discharge, velocity, and scour, set "
            "a flood exceedance, and add to the cumulative flood and scour counters. The scope is applied "
            "before the shock, so unaffected bridges are returned unchanged. This is important for "
            "auditability: the triage output can be reproduced from the event record alone.",
            styles["Body"],
        )
    )
    story.append(
        P(
            f"The demonstration event is <b>{event['event_id']}</b>, a {event['event_type'].lower()} "
            f"event scoped to <b>{event['district']}</b> with severity {event['severity']:.2f}. It adds "
            f"{event['monsoon_anomaly_increment']:.1f} to the monsoon anomaly, multiplies rainfall by "
            f"{event['rainfall_multiplier']:.1f}, discharge by {event['discharge_multiplier']:.1f}, "
            f"velocity by {event['velocity_multiplier']:.1f}, and scour by {event['scour_multiplier']:.1f}, "
            f"and adds a flood exceedance plus {event['cumulative_scour_increment']:.1f} cumulative scour "
            f"load. These are scenario parameters, not estimates of a specific historical event.",
            styles["Body"],
        )
    )
    story.append(P("9.2 Survival-curve update", styles["H2"]))
    story.append(
        P(
            "For the Cox view, the updater modifies the last counting-process interval for each targeted "
            "bridge. The fitted baseline cumulative hazard and the updated covariate vector are then used "
            "to evaluate a conditional survival curve on a one-year grid from zero to 60 years: "
            "S(t0 + h | t0, X_shock) = exp(−exp(βᵀX_shock) × [H0(t0 + h) − H0(t0)]). For a tree or "
            "ensemble view, the updater modifies the pre-entry and landmark exposure aggregates and the "
            "existing step-function conditional-survival helper is evaluated on the same grid. The "
            "resulting curve is checked for finiteness, probability bounds, and monotonicity.",
            styles["Body"],
        )
    )
    story.append(P("9.3 Remaining useful life", styles["H2"]))
    story.append(
        P(
            "RUL is defined as the first time the conditional survival curve reaches the operational "
            "threshold τ = 0.45: RUL = inf{h ≥ 0 : S(t0 + h | t0) ≤ τ}. The implementation evaluates a "
            "discrete grid and linearly interpolates between the last point above τ and the first point "
            "below τ. If the curve never crosses τ within the 60-year evaluation window, the RUL is "
            "right-censored at 60 years and flagged as censored. This avoids inventing a precise crossing "
            "that the data cannot support.",
            styles["Body"],
        )
    )
    story.append(
        P(
            "The triage output reports baseline RUL, post-event RUL, RUL loss, survival at the decision "
            "horizon, survival drop, priority score, priority class, and an immediate-NDT boolean. The "
            "priority score combines current post-event risk (50%), the shock-induced survival increase "
            "(30%), and the relative RUL loss (20%). A bridge is CRITICAL if post-event survival is below "
            "0.45 or post-event RUL is at most five years; HIGH if it loses at least five RUL years, has a "
            "large survival drop, or is a paper-inspection suspect.",
            styles["Body"],
        )
    )
    story.append(figure_flowable(figures["triage"], "Figure 6. Top post-disaster triage findings: five-year conditional survival and remaining useful life before and after the Darbhanga shock.", styles, width=173 * mm))
    story.append(P("9.4 SHAP explainability", styles["H2"]))
    story.append(
        P(
            "SHAP decomposes a model prediction into a baseline value and additive feature "
            "contributions: f(x) = φ0 + Σ φj. The project uses shap.Explainer with the permutation "
            "algorithm on a background sample drawn from the training cohort. Permutation SHAP is "
            "model-agnostic, which matters because the survival models are wrapped objects rather than "
            "plain scikit-learn regressors. The explainer is applied to the survival risk prediction, "
            "not to the raw survival probability, so positive contributions mean increased risk.",
            styles["Body"],
        )
    )
    story.append(
        P(
            "The ensemble requires special handling. SurvivalEnsembleModel.predict_risk rank-averages "
            "the two component risks within whatever batch it receives. That is valid for concordance, "
            "but it makes pointwise SHAP values depend on the batch composition. Module 4 fixes this by "
            "calibrating a z-score for each component on the SHAP background sample and averaging the "
            "calibrated risks. The result is a fixed, batch-independent risk function. The report verifies "
            "additivity by reconstructing each prediction from its base value plus SHAP contributions.",
            styles["Body"],
        )
    )
    story.append(
        P(
            f"The validated run explained {triage_metrics['shap']['n_explained']} bridges and wrote "
            f"{triage_metrics['shap']['n_explained']} standalone HTML force plots. The maximum additivity "
            f"error was {triage_metrics['shap']['max_additivity_error']:.2e}, essentially machine "
            f"precision. The contribution table reports feature value, signed SHAP value, absolute "
            f"magnitude, direction, base value, and model risk prediction for the top ten features per "
            f"bridge. Audit and truth fields are rejected by the same prefix allowlist used by the "
            f"survival pipeline.",
            styles["Body"],
        )
    )
    story.append(P("9.5 Triage results", styles["H2"]))
    story.append(
        P(
            f"The event targeted {triage['cohort']['n_targeted']} bridges in the district, of which "
            f"{triage_metrics['n_scored']} were in the held-out test split. All "
            f"{triage_metrics['n_scored']} scored bridges received an immediate NDT recommendation: "
            f"{triage_metrics['priority_counts'].get('CRITICAL', 0)} CRITICAL and "
            f"{triage_metrics['priority_counts'].get('HIGH', 0)} HIGH. The mean five-year survival drop "
            f"was {triage_metrics['mean_survival_drop']:.3f} and the maximum was "
            f"{triage_metrics['max_survival_drop']:.3f}. The mean RUL loss was "
            f"{triage_metrics['mean_rul_loss_years']:.2f} years and the maximum was "
            f"{triage_metrics['max_rul_loss_years']:.2f} years.",
            styles["Body"],
        )
    )
    top = triage["findings"]["top_findings"][:5]
    story.append(
        make_table(
            [
                ["Bridge", "District", "Baseline S", "Post S", "RUL before", "RUL after", "RUL loss", "Priority"],
                *[
                    [
                        item["bridge_id"],
                        item["district"],
                        f"{item['baseline_survival']:.3f}",
                        f"{item['post_event_survival']:.3f}",
                        f"{item['rul_baseline_years']:.1f}",
                        f"{item['rul_post_event_years']:.1f}",
                        f"{item['rul_loss_years']:.1f}",
                        item["priority"],
                    ]
                    for item in top
                ],
            ],
            [24 * mm, 24 * mm, 22 * mm, 20 * mm, 23 * mm, 22 * mm, 22 * mm, 18 * mm],
            styles,
            font_size=7.3,
        )
    )
    story.append(Spacer(1, 4))

    # ------------------------------------------------------------------
    # 10. Why these models
    # ------------------------------------------------------------------
    story.append(P("10. Why These Models Were Chosen", styles["H1"]))
    story.append(
        P(
            "Model choice is not a matter of using the most complex available estimator. Each choice is "
            "tied to the structure of the problem and to the evidence available at training time.",
            styles["Body"],
        )
    )
    story.append(P("10.1 Why survival analysis instead of classification", styles["H2"]))
    story.append(
        P(
            "The target is a time to a critical condition state, not a binary label at an arbitrary "
            "snapshot. Many bridges are still healthy when the study ends; discarding them or treating "
            "them as negatives would bias the estimate. Survival analysis uses right-censored "
            "observations correctly: a bridge that has survived 20 years without failure contributes "
            "information about survival up to 20 years, not a fabricated failure at year 20. It also "
            "produces a full survival curve, which is required for conditional audit and post-disaster "
            "RUL. A classifier cannot produce a curve without imposing an arbitrary horizon and losing "
            "the censoring structure.",
            styles["Body"],
        )
    )
    story.append(P("10.2 Why Cox time-varying is the primary model", styles["H2"]))
    story.append(
        P(
            "The Andersen-Gill Cox model is the natural baseline for a longitudinal portfolio with "
            "time-varying covariates. It supports staggered entry, right-censoring, and repeated "
            "intervals, and it produces hazard ratios that engineers and reviewers can interpret. The "
            "time-varying form allows the hazard to respond to the current monsoon and scour state "
            "rather than forcing the entire effect into a static baseline. Its global linearity is a "
            "limitation, which is exactly why the non-linear tree models are also fitted and compared.",
            styles["Body"],
        )
    )
    story.append(P("10.3 Why random survival forest and gradient boosting are retained", styles["H2"]))
    story.append(
        P(
            "The tree models test whether non-linear interactions matter. In this benchmark they do add "
            "value: the gradient-boosting model has the highest held-out Harrell C-index, and the random "
            "forest captures span, roadway width, truck share, ADT, flood proneness, and pre-entry "
            "rainfall as important splits. They are not selected as the final audit model because the "
            "selection rule is fixed on validation and because the Cox model provides the interpretable "
            "climate hazard ratios needed for the engineering narrative. Keeping all four models in the "
            "report also makes the comparison honest: the selected model is not claimed to be "
            "universally superior.",
            styles["Body"],
        )
    )
    story.append(P("10.4 Why IsolationForest rather than a supervised fraud classifier", styles["H2"]))
    story.append(
        P(
            "A supervised classifier would need a reliable label of falsified inspection. In the field, "
            "that label is rarely available: paper inspections are discovered after failures, audits, or "
            "whistleblower reports, so the positive class is systematically under-observed. Training a "
            "classifier on such labels would reproduce the selection bias. IsolationForest is an "
            "unsupervised anomaly detector that asks whether an audit record is unusual in the joint "
            "space of objective exposure, physics score, and reported condition. It is complemented by "
            "the transparent physics residual, so the final flag is not a black-box anomaly score.",
            styles["Body"],
        )
    )
    story.append(P("10.5 Why SHAP permutation rather than a bespoke importance heuristic", styles["H2"]))
    story.append(
        P(
            "Global split importance explains the model on average, not why a specific bridge was "
            "flagged. SHAP provides local, additive attributions with a consistency guarantee under the "
            "permutation formulation. Permutation SHAP is model-agnostic, so the same explanation "
            "procedure works for the Cox risk function and the tree ensemble. The additivity check is "
            "important: if the base value plus contributions does not reconstruct the prediction, the "
            "explanation is not trustworthy. The validated run passes that check to machine precision.",
            styles["Body"],
        )
    )
    story.append(P("10.6 Why no deep learning", styles["H2"]))
    story.append(
        P(
            "The available cohort is tabular, right-censored, and relatively small by deep-learning "
            "standards. A neural survival model would add many parameters, require careful tuning and "
            "calibration, and sacrifice the interpretability of hazard ratios, without evidence that the "
            "non-linear interactions exceed what survival trees can capture. The stated project "
            "constraint is classical ML only. The chosen stack respects that constraint and is "
            "appropriate for the sample size and the engineering audit requirement.",
            styles["Body"],
        )
    )

    # ------------------------------------------------------------------
    # 11. Evaluation
    # ------------------------------------------------------------------
    story.append(P("11. Evaluation, Validation, and Leakage Controls", styles["H1"]))
    story.append(
        P(
            "The evaluation is intentionally conservative. Model selection uses validation only, the "
            "audit calibration uses training only, and the post-disaster demonstration scores the test "
            "split only. The following table summarises the headline evidence.",
            styles["Body"],
        )
    )
    story.append(
        make_table(
            [
                ["Dimension", "Evidence"],
                ["Survival discrimination", f"Selected Cox: test Harrell C {test_metrics['harrell_c']:.4f}; Uno IPCW C {test_metrics['uno_ipcw_c']:.4f}"],
                ["Survival calibration", f"Selected Cox: test Integrated Brier Score {test_metrics['integrated_brier_score']:.4f}"],
                ["Audit ranking", f"Audit Z-score ROC-AUC {audit_metrics['audit_z_score']['roc_auc']:.4f}; average precision {audit_metrics['audit_z_score']['average_precision']:.4f}"],
                ["Audit operating point", f"{audit['findings']['n_paper_inspection_suspect']} primary suspects; {audit['findings']['n_audit_alert']} broader alerts"],
                ["Post-disaster triage", f"{triage_metrics['n_immediate_ndt']} immediate NDT; mean RUL loss {triage_metrics['mean_rul_loss_years']:.2f} years"],
                ["Explainability", f"Additivity error {triage_metrics['shap']['max_additivity_error']:.2e}; {triage_metrics['shap']['n_explained']} force plots"],
                ["Pipeline integrity", "All Module 1 checks pass; all Module 4 checks pass"],
            ],
            [42 * mm, 123 * mm],
            styles,
        )
    )
    story.append(P("11.1 Leakage controls", styles["H2"]))
    story.extend(
        bullets(
            [
                "Bridge-level split before imputation, encoding, calibration, or model fitting.",
                "Feature allowlists reject EVENT, DURATION, REPORTED_*, TRUE_*, LATENT_*, INSPECTOR_*, and other truth-side prefixes from survival matrices.",
                "Pre-entry climate aggregates are reconstructed from years strictly before ENTRY_YEAR.",
                "Point-in-time climate joins use the interval end year, never future years.",
                "Reported condition is permitted only in the audit feature set, and only the latest minimum rating is allowed.",
                "Audit ground truth is merged after scoring; it never enters the residual calibration or IsolationForest.",
                "SHAP feature names are checked against the same truth/audit prefixes.",
                "The test set is not used for model selection or calibration.",
            ],
            styles["Bullet"],
        )
    )
    story.append(P("11.2 What the checks do not prove", styles["H2"]))
    story.append(
        P(
            "Passing internal consistency checks does not establish external validity. The synthetic "
            "generator is a model of reality, and the paper-inspection labels are simulated. The audit "
            "precision and recall therefore measure whether the pipeline recovers the generator's latent "
            "mechanism, not whether it would recover real corruption in an IBMS extract. The correct "
            "next validation step is a temporal holdout on real NBI/IBMS records with audit outcomes "
            "collected independently of the model.",
            styles["Body"],
        )
    )

    # ------------------------------------------------------------------
    # 12. Operational workflow
    # ------------------------------------------------------------------
    story.append(P("12. Operational Workflow", styles["H1"]))
    story.append(
        P(
            "The system is designed to run as a monthly or post-event batch job. A typical operating "
            "cycle is:",
            styles["Body"],
        )
    )
    story.append(
        P(
            "<b>1. Ingest.</b> Load the latest inventory, inspection, traffic, and district climate "
            "records. Validate required columns and missingness. <b>2. Score.</b> Apply the frozen "
            "preprocessing pipeline and survival model to every active bridge. Compute conditional "
            "survival over the audit horizon. <b>3. Audit.</b> Compare physics-implied condition with "
            "reported condition, compute the Audit Z-score, and run IsolationForest. <b>4. Triage.</b> "
            "If a flood or cyclone has occurred, apply the event shock to the affected district, "
            "recompute survival curves, and rank RUL loss. <b>5. Explain.</b> Generate SHAP force plots "
            "for the top-N bridges. <b>6. Dispatch.</b> Send the NDT list to the field team and preserve "
            "the model version, event record, and explanation artefacts for audit.",
            styles["Body"],
        )
    )
    story.append(
        P(
            "The operational output should be read as a prioritisation tool. A high-priority flag means "
            "the bridge deserves physical inspection now; it is not a certification that the bridge will "
            "fail, nor a legal finding of inspector misconduct. The force plot is a communication aid for "
            "the inspection team, not a substitute for engineering judgment.",
            styles["Quote"],
        )
    )

    # ------------------------------------------------------------------
    # 13. Reproducibility
    # ------------------------------------------------------------------
    story.append(P("13. Reproducibility and Execution", styles["H1"]))
    story.append(
        P(
            "The project is reproducible from a fixed seed. The complete end-to-end command is:",
            styles["Body"],
        )
    )
    story.append(
        P(
            "<font name='Courier'>python scripts/run_end_to_end.py --config config/default.json</font>",
            ParagraphStyle("Code", parent=styles["Body"], fontName="Courier", fontSize=8.5, leading=12, backColor=LIGHT_BLUE, borderPadding=5, leftIndent=8, rightIndent=8),
        )
    )
    story.append(
        P(
            "A fast smoke run is available with <font name='Courier'>--quick</font>. The end-to-end script "
            "generates the synthetic cohort, saves the raw and processed data, fits all four survival "
            "models, fits the audit verifier, reloads the persisted artefacts from disk, applies the "
            "post-disaster event, writes SHAP explanations, and produces a consolidated JSON summary. "
            "Reloading the engine and audit artefacts is deliberate: it tests the serialisation boundary, "
            "not just the in-memory path.",
            styles["Body"],
        )
    )
    story.append(
        make_table(
            [
                ["Command", "Purpose"],
                ["python -m pytest -q", "Run all 41 verification tests"],
                ["python -m ruff check src tests scripts", "Static style and import checks"],
                ["python scripts/generate_report_pdf.py", "Regenerate this PDF from current artefacts"],
                ["python -m bridge_engine.triage_and_xai --help", "Module 4 CLI"],
            ],
            [88 * mm, 77 * mm],
            styles,
            font_size=7.8,
        )
    )

    # ------------------------------------------------------------------
    # 14. Limitations
    # ------------------------------------------------------------------
    story.append(P("14. Limitations, Risks, and Honest Caveats", styles["H1"]))
    story.append(
        P(
            "The following limitations are material and should be stated in any deployment review.",
            styles["Body"],
        )
    )
    story.extend(
        bullets(
            [
                "<b>Synthetic evidence.</b> The current results validate the implementation on a synthetic benchmark, not on raw Indian bridge failures. The generator is calibrated to known failure physics, but it is still a simulation.",
                "<b>Paper-label construction.</b> The paper-inspection label is simulated from an inspector-bias process. Real falsification is under-observed and may have different structure.",
                "<b>Audit recall.</b> The primary rule is high precision and low recall by design. A deployment must choose an operating threshold consistent with inspection capacity and risk appetite.",
                "<b>Conditional survival extrapolation.</b> Tree-model survival curves are step functions with a parametric tail beyond the observed domain. The Cox model uses a log-linear tail beyond the largest baseline hazard time. Both are extrapolations.",
                "<b>Event scenario parameters.</b> The Darbhanga shock is a scenario, not a reconstruction of a specific historical flood. Its multipliers should be calibrated against district gauge and satellite data before operational use.",
                "<b>Unobserved confounders.</b> Maintenance history, foundation type, riverbed morphology, and inspection quality are not fully represented. Their omission can bias hazard estimates.",
                "<b>No causal claim.</b> Hazard ratios are associational. They describe the fitted model, not a controlled causal effect of one intervention.",
                "<b>Distribution shift.</b> A model trained on historical climate may not extrapolate safely to unprecedented rainfall or to a new structural typology without monitoring and recalibration.",
            ],
            styles["Bullet"],
        )
    )

    # ------------------------------------------------------------------
    # 15. Roadmap
    # ------------------------------------------------------------------
    story.append(P("15. Recommended Next Steps", styles["H1"]))
    story.extend(
        bullets(
            [
                "Replace the synthetic adapter with a cleaned raw NBI/IBMS loader while preserving the same schema and leakage checks.",
                "Add a temporal holdout: train on pre-2015 records and test on later failures to measure forward-shift performance.",
                "Calibrate district event multipliers against IMD rainfall, CWC discharge, and satellite flood observations.",
                "Collect independent NDT outcomes and use them to validate or recalibrate the audit operating point.",
                "Add maintenance interventions as time-varying covariates; the current model treats maintenance as unobserved.",
                "Add spatial random effects or district frailty terms to account for unobserved local geology and construction practice.",
                "Add conformal or Bayesian uncertainty bands around RUL and audit scores for operational decision-making.",
                "Integrate a field mobile workflow so that NDT results feed back into the next scoring cycle.",
                "Monitor calibration drift and re-fit on a fixed cadence rather than continuously, to preserve auditability.",
            ],
            styles["Bullet"],
        )
    )

    # ------------------------------------------------------------------
    # Appendix A
    # ------------------------------------------------------------------
    story.append(P("Appendix A. Selected Field Dictionary", styles["H1"]))
    story.append(
        make_table(
            [
                ["Field", "Interpretation"],
                ["YEAR_BUILT", "Construction year; age is derived at the interval end"],
                ["STRUCTURE_KIND_MATERIAL", "Concrete, prestressed concrete, steel, composite, masonry, timber"],
                ["STRUCTURE_TYPE_DESIGN", "Slab, beam/girder, box girder, truss, arch, suspension, culvert"],
                ["ADT_TRAFFIC / TRUCK_ADT_PCT", "Average daily traffic and truck share"],
                ["MAX_SPAN_LEN", "Longest span in metres"],
                ["WATERWAY_ADEQUACY", "NBI-style waterway adequacy rating, higher is better"],
                ["SCOUR_CRITICAL", "Indicator of scour-critical foundation exposure"],
                ["MONSOON_ANOMALY", "District monsoon rainfall Z-score"],
                ["PEAK_DISCHARGE_M3S", "Peak seasonal discharge proxy"],
                ["PEAK_VELOCITY_MPS", "Peak flow velocity proxy"],
                ["PEAK_SCOUR_INDEX", "Discharge × velocity / 1000"],
                ["FLOOD_EXCEEDANCE", "Discharge above the district's reference 90th percentile"],
                ["CUMULATIVE_FLOOD_EXCEEDANCE", "Causal flood count to the interval end"],
                ["CUMULATIVE_SCOUR_LOAD", "Causal scour index sum to the interval end"],
                ["REPORTED_LAST_MIN_COND", "Latest reported minimum condition, audit side only"],
                ["EXPECTED_SURVIVAL", "Conditional survival over the audit horizon"],
                ["RUL", "First time the conditional survival curve crosses the threshold"],
                ["PAPER_INSPECTION_SUSPECT", "Reported healthy condition with physics survival below 0.68"],
            ],
            [48 * mm, 112 * mm],
            styles,
            font_size=7.6,
        )
    )

    # ------------------------------------------------------------------
    # Appendix B
    # ------------------------------------------------------------------
    story.append(P("Appendix B. Validated Hyperparameters", styles["H1"]))
    story.append(
        make_table(
            [
                ["Component", "Setting"],
                ["Synthetic cohort", f"{summary['n_bridges']:,} bridges, seed {summary['seed']}, study end 2024"],
                ["Split", f"test {config.get('pipeline', {}).get('test_size', 0.20):.0%}, validation {config.get('pipeline', {}).get('validation_size', 0.15):.0%}"],
                ["Cox penalty", f"penalizer {survival['config']['cox_penalizer']}, l1_ratio {survival['config']['cox_l1_ratio']}"],
                ["Cox collinearity", f"correlation threshold {survival['config']['cox_correlation_threshold']}"],
                ["RSF", f"{survival['config']['rsf_n_estimators']} trees, min leaf {survival['config']['rsf_min_samples_leaf']}, max features {survival['config']['rsf_max_features']}"],
                ["GBSA", f"{survival['config']['gbsa_n_estimators']} stages, learning rate {survival['config']['gbsa_learning_rate']}, depth {survival['config']['gbsa_max_depth']}, subsample {survival['config']['gbsa_subsample']}"],
                ["Evaluation horizons", ", ".join(str(int(value)) for value in survival['config']['horizons']) + " years"],
                ["Audit horizon / threshold", f"{audit['physics_model']['audit_horizon_years']:.0f} years / S < {audit['physics_model']['critical_survival_threshold']}"],
                ["Audit calibration", f"median/MAD on {audit['detector']['residual_calibration']['n_rows']:,} training rows"],
                ["IsolationForest", f"{audit['detector']['isolation_forest']['n_estimators']} trees, contamination {audit['detector']['isolation_forest']['contamination']}"],
                ["Triage RUL grid", "0–60 years, 1-year step; threshold 0.45"],
                ["SHAP", f"permutation explainer, background {triage['config']['shap_background_size']}, max evaluations {triage['config']['shap_max_evals']}"],
            ],
            [55 * mm, 105 * mm],
            styles,
            font_size=7.7,
        )
    )

    # ------------------------------------------------------------------
    # Appendix C
    # ------------------------------------------------------------------
    story.append(P("Appendix C. Artefact Inventory", styles["H1"]))
    story.append(
        make_table(
            [
                ["Path", "Contents"],
                ["artifacts/data/bridges.parquet", "Bridge inventory and last-inspection fields"],
                ["artifacts/data/climate.parquet", "District-year climate and hydraulic panel"],
                ["artifacts/data/inspections.parquet", "Longitudinal inspection records"],
                ["artifacts/data/static_*.parquet", "Model-ready static splits"],
                ["artifacts/data/counting_*.parquet", "Andersen-Gill interval splits"],
                ["artifacts/data/EVAL_ONLY_ground_truth.parquet", "Evaluation-only latent truth and paper flag"],
                ["artifacts/module2/survival_engine.joblib", "Fitted Cox, RSF, GBSA, ensemble"],
                ["artifacts/module2/survival_report.json", "Metrics, hazard ratios, feature importance"],
                ["artifacts/module3/audit_findings.parquet", "Audit scores, flags, and priorities"],
                ["artifacts/module3/audit_models.joblib", "IsolationForest, feature pipeline, calibration"],
                ["artifacts/module4/triage_findings.parquet", "Post-event survival, RUL, and priority"],
                ["artifacts/module4/xai_contributions.parquet", "Local SHAP contributions"],
                ["artifacts/module4/force_plots/*.html", "Standalone local force plots"],
                ["artifacts/end_to_end_summary.json", "Consolidated run summary"],
            ],
            [82 * mm, 78 * mm],
            styles,
            font_size=7.7,
        )
    )

    # ------------------------------------------------------------------
    # Appendix D
    # ------------------------------------------------------------------
    story.append(P("Appendix D. Conceptual References and Glossary", styles["H1"]))
    story.append(
        P(
            "The methods used here are established classical techniques. The report does not depend on "
            "proprietary algorithms. For deeper reading, consult the original literature on the Cox "
            "proportional hazards model, Andersen-Gill counting processes, random survival forests, "
            "gradient boosting, IsolationForest, and SHAP. The implementation choices in this project are "
            "documented in the module docstrings and in the persisted JSON reports.",
            styles["Body"],
        )
    )
    story.append(
        make_table(
            [
                ["Term", "Meaning in this project"],
                ["Right-censoring", "The bridge has not failed by the end of observation; we know it survived at least that long."],
                ["Counting process", "One row per bridge-year with start, stop, event, and time-varying covariates."],
                ["Hazard", "Instantaneous rate of failure given survival to time t and covariates X(t)."],
                ["Conditional survival", "Probability of surviving the next h years given survival to t0."],
                ["RUL", "Remaining useful life; years until conditional survival falls to the operational threshold."],
                ["Concordance", "Probability that a model ranks a comparable pair in the correct order."],
                ["IPCW", "Inverse probability of censoring weighting, used to reduce censoring bias."],
                ["IBS", "Integrated Brier Score across a set of time horizons."],
                ["Audit Z-score", "Robust standardized reported-minus-physics condition gap."],
                ["Paper inspection", "A reported healthy condition that is inconsistent with objective physics and climate evidence."],
                ["SHAP", "Shapley additive explanations; local additive feature attribution."],
            ],
            [38 * mm, 122 * mm],
            styles,
            font_size=7.7,
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        P(
            "This report was generated from the persisted artifacts of the validated end-to-end run. "
            "The seed, cohort size, metrics, and model names in the text are read from those artifacts, "
            "so the document can be regenerated after any future run.",
            styles["Quote"],
        )
    )

    return story


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_pdf() -> Path:
    data = load_artifacts()
    figures = make_figures(data)
    styles = build_styles()
    story = build_story(data, styles, figures)

    doc = TechnicalReportTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=22 * mm,
        bottomMargin=18 * mm,
        title="Zero-Trust Bridge Survival & Audit Verification Engine",
        author="Zero-Trust Bridge Engine",
        subject="Technical design, data, model rationale, validation, and triage report",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="normal",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc.addPageTemplates(
        [
            PageTemplate(id="First", frames=[frame], onPage=_draw_cover),
            PageTemplate(id="Later", frames=[frame], onPage=_draw_later),
        ]
    )
    doc.multiBuild(story)
    return OUTPUT_PDF


if __name__ == "__main__":
    output = build_pdf()
    print(f"Wrote {output}")

