"""Research figure style — a single committed look for dissertation figures.

Design: modern LaTeX. Charter serif to match a Bitstream-Charter document body,
STIX mathtext, no chart junk, cool sequential ramp for matrices, one fixed
categorical palette reused across every figure so a colour means the same
thing in every plot of the thesis.

Usage
-----
    from research_style import apply_style, plot_confusion_matrix, save_figure

    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    plot_confusion_matrix(cm, EKMAN_LABELS, ax=ax, normalize="true")
    save_figure(fig, "results/run_01/confusion_matrix")

Every public function draws only what it is given. Nothing here invents,
smooths, or resamples data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, to_hex, to_rgb
from matplotlib.figure import Figure

__all__ = [
    "apply_style",
    "INK",
    "MUTED",
    "BACKGROUND",
    "PALETTE",
    "EKMAN_LABELS",
    "EKMAN_COLORS",
    "CLASS_ALIASES",
    "EXTRA_COLORS",
    "SEQUENTIAL",
    "DIVERGING",
    "canonical_class",
    "color_for",
    "plot_confusion_matrix",
    "plot_grouped_bars",
    "plot_box_by_class",
    "plot_training_curves",
    "plot_metric_comparison",
    "plot_architecture",
    "plot_mel_spectrogram",
    "annotate_significance",
    "save_figure",
    "latex_table",
    "plot_table",
    "plot_donut",
    "plot_model_bars",
    "Chips",
    "Check",
    "Muted",
]

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

BACKGROUND = "#ffffff"
INK = "#14202b"
MUTED = "#7d8a99"
GRID = "#c8d0d8"

#: Categorical palette. Ordered so adjacent entries stay distinguishable.
PALETTE: tuple[str, ...] = (
    "#2b7fb8",  # blue
    "#f0a92b",  # amber
    "#d1453b",  # red
    "#2f9e63",  # green
    "#7a4fa3",  # violet
    "#e0559b",  # magenta
    "#7d8a99",  # slate
    "#8c6239",  # brown
)

#: Ekman scheme plus neutral, in the label-id order used across this project.
EKMAN_LABELS: tuple[str, ...] = (
    "neutral",
    "happiness",
    "sadness",
    "fear",
    "anger",
    "disgust",
    "surprise",
)

#: Fixed colour per emotion. Never reorder — figures across chapters rely on it.
EKMAN_COLORS: dict[str, str] = {
    "neutral": "#7d8a99",
    "happiness": "#f0a92b",
    "sadness": "#2b7fb8",
    "fear": "#7a4fa3",
    "anger": "#d1453b",
    "disgust": "#2f9e63",
    "surprise": "#e0559b",
}

#: Sequential ramp for counts, rates, and confusion matrices.
SEQUENTIAL: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
    "rf_sequential",
    ["#fbfcfd", "#cfe3ee", "#7fb3d0", "#3a6f9a", "#152f4d"],
    N=256,
)

#: Diverging ramp. Use ONLY for signed quantities (differences, residuals).
DIVERGING: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
    "rf_diverging",
    ["#1f4d7d", "#7fa8c4", "#f7f3ea", "#e0a08e", "#8c2020"],
    N=256,
)


#: Corpus-specific spellings of the Ekman classes. Corpora disagree on wording
#: (RAVDESS says ``happy``, nEMO says ``happiness``), and a class must not change
#: colour because of which corpus a figure happens to come from.
CLASS_ALIASES: dict[str, str] = {
    "happy": "happiness",
    "joy": "happiness",
    "sad": "sadness",
    "angry": "anger",
    "anger ": "anger",
    "fearful": "fear",
    "scared": "fear",
    "disgusted": "disgust",
    "surprised": "surprise",
    "pleasant surprise": "surprise",  # TESS
}

#: Classes that exist in some corpora but are not Ekman categories. They still need
#: a stable colour. ``calm`` appears only in RAVDESS and maps to ``neutral`` in any
#: cross-corpus analysis, so its colour sits deliberately close to the neutral grey.
EXTRA_COLORS: dict[str, str] = {
    "calm": "#5b8f8a",
    "other": "#b9c2cb",
    "unparsable": "#b9c2cb",
}


def canonical_class(label: str) -> str:
    """Canonical class name for a corpus-specific spelling."""
    key = label.strip().lower()
    return CLASS_ALIASES.get(key, key)


def color_for(label: str, index: int = 0) -> str:
    """Colour for a class label: the fixed emotion colour, else the palette.

    Corpus-specific spellings are resolved first, so ``happy`` and ``happiness``
    always render in the same colour.
    """
    canonical = canonical_class(label)
    if canonical in EKMAN_COLORS:
        return EKMAN_COLORS[canonical]
    if canonical in EXTRA_COLORS:
        return EXTRA_COLORS[canonical]
    return PALETTE[index % len(PALETTE)]


#: One opacity for every filled mark that carries a class colour (boxes, stacked
#: bars, cards), so a class reads as the same tint in every figure.
FILL_ALPHA = 0.71


def tint(color: str, alpha: float = FILL_ALPHA) -> str:
    """``color`` at ``alpha`` flattened onto the white background, as opaque hex.

    Opaque rather than a real alpha channel: stacked marks with white separators
    and contrast checks on labels then see the colour the reader sees.
    """
    base = np.array(to_rgb(color))
    ground = np.array(to_rgb(BACKGROUND))
    return to_hex(alpha * base + (1 - alpha) * ground)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def apply_style(base_font_size: float = 10.0) -> None:
    """Install the style globally. Call once, before creating any figure."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Charter", "XCharter", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": base_font_size,
            "axes.titlesize": base_font_size + 1,
            "axes.labelsize": base_font_size,
            "xtick.labelsize": base_font_size - 1,
            "ytick.labelsize": base_font_size - 1,
            "legend.fontsize": base_font_size - 1,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.titlepad": 9.0,
            "axes.prop_cycle": mpl.cycler(color=list(PALETTE)),
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,  # embed as TrueType, not Type 3 — journals require it
            "ps.fonttype": 42,
        }
    )


def _contrast_ink(rgba: Sequence[float]) -> str:
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "#ffffff" if luminance < 0.55 else INK


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: Sequence[str],
    ax: Axes | None = None,
    *,
    normalize: Literal["true", "pred", "all", None] = None,
    title: str | None = None,
    xlabel: str = "Predicted label",
    ylabel: str = "True label",
    row_labels: Sequence[str] | None = None,
    mask_diagonal: bool = False,
    diverging: bool = False,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    bold_above: float | None = None,
    percent: bool = False,
    show_zeros: bool = False,
) -> Axes:
    """Render a confusion matrix.

    Parameters
    ----------
    matrix
        Square array of counts, ``matrix[true, pred]``.
    normalize
        ``"true"`` divides by row sums (per-class recall on the diagonal),
        ``"pred"`` by column sums, ``"all"`` by the grand total, ``None``
        keeps raw counts.
    mask_diagonal
        Hide the diagonal so off-diagonal structure sets the colour scale.
        Use for error-focused or annotation-shift matrices.
    diverging
        Use the diverging ramp centred at zero. Only valid when the matrix
        holds signed values (e.g. a difference between two models).
    row_labels
        Row labels, when the rows are not the same categories as the columns —
        a rectangular contingency matrix. Needed whenever the true and predicted
        label spaces differ: an extra "other/unparsable" column for a generative
        model, or a cross-corpus matrix over partially overlapping classes.
        Defaults to ``labels`` (the square case).
    bold_above
        Cell values at or above this threshold are drawn bold. Defaults to the
        75th percentile of the displayed values.
    percent
        With ``normalize``, print cells as whole percentages and leave zero
        cells blank — easier to read from a projector than ``0.xx``.
    show_zeros
        Print ``0`` in empty cells instead of leaving them blank.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {matrix.shape}")
    rows = list(row_labels) if row_labels is not None else list(labels)
    if len(labels) != matrix.shape[1]:
        raise ValueError(f"{len(labels)} column labels for {matrix.shape[1]} columns")
    if len(rows) != matrix.shape[0]:
        raise ValueError(f"{len(rows)} row labels for {matrix.shape[0]} rows")
    if mask_diagonal and matrix.shape[0] != matrix.shape[1]:
        raise ValueError("mask_diagonal needs a square matrix")

    display = matrix.copy()
    if normalize == "true":
        totals = display.sum(axis=1, keepdims=True)
        display = np.divide(display, totals, out=np.zeros_like(display), where=totals != 0)
    elif normalize == "pred":
        totals = display.sum(axis=0, keepdims=True)
        display = np.divide(display, totals, out=np.zeros_like(display), where=totals != 0)
    elif normalize == "all":
        total = display.sum()
        display = display / total if total else display

    scaled = display.copy()
    if mask_diagonal:
        np.fill_diagonal(scaled, np.nan)

    finite = scaled[np.isfinite(scaled)]
    if finite.size == 0:
        raise ValueError("nothing to plot: every cell is masked or non-finite")

    cmap = (DIVERGING if diverging else SEQUENTIAL).copy()
    cmap.set_bad(BACKGROUND)

    if diverging:
        extent = float(np.nanmax(np.abs(finite))) or 1.0
        norm: Normalize = TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)
    else:
        top = 1.0 if (percent and normalize is not None) else float(np.nanmax(finite)) or 1.0
        norm = Normalize(vmin=0.0, vmax=top)  # percent mode: fixed 0–100% so matrices compare across slides

    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 4.8))

    aspect = "equal" if matrix.shape[0] == matrix.shape[1] else "auto"
    image = ax.imshow(scaled, cmap=cmap, norm=norm, aspect=aspect, interpolation="nearest")

    is_rate = normalize is not None
    threshold = bold_above if bold_above is not None else float(np.percentile(finite, 75))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if mask_diagonal and i == j:
                ax.text(j, i, "·", ha="center", va="center", color=MUTED, fontsize=11)
                continue
            value = display[i, j]
            if value == 0 and not show_zeros and (not is_rate or percent):
                continue
            text = f"{100 * value:.0f}" if (is_rate and percent) else (f"{value:.2f}" if is_rate else f"{value:.0f}")
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=mpl.rcParams["font.size"] - (0.5 if percent else 1.5),
                color=_contrast_ink(cmap(norm(scaled[i, j] if np.isfinite(scaled[i, j]) else 0))),
                fontweight="bold" if value >= threshold else "normal",
            )

    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Hairline separators drawn in the page colour — cleaner than grid lines.
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color=BACKGROUND, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if colorbar:
        bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=0, labelsize=mpl.rcParams["font.size"] - 2)
        if colorbar_label is None:
            colorbar_label = {
                "true": "share of true class",
                "pred": "share of predicted class",
                "all": "share of all examples",
                None: "examples",
            }[normalize]
        if percent and is_rate:
            bar.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        bar.set_label(colorbar_label, fontsize=mpl.rcParams["font.size"] - 2)

    return ax


# ---------------------------------------------------------------------------
# Bars and curves
# ---------------------------------------------------------------------------


def plot_grouped_bars(
    categories: Sequence[str],
    series: dict[str, Sequence[float]],
    ax: Axes | None = None,
    *,
    ylabel: str = "",
    title: str | None = None,
    per_category_color: bool = False,
    errors: dict[str, Sequence[float]] | None = None,
    value_labels: bool = False,
    value_fmt: str = "%.3g",
    colors: Sequence[str] | None = None,
) -> Axes:
    """Grouped bar chart.

    ``per_category_color=True`` colours bars by category (using the fixed
    emotion colours where the name matches) and distinguishes series by alpha;
    otherwise each series gets its own palette colour.

    ``colors`` overrides the per-series colours, one per series. Use it when the
    series are variants of one thing (splits, seeds, ablation arms) and should
    read as tints of a single hue rather than as unrelated categories. Ignored
    when ``per_category_color`` is set, because then the legend cannot show a
    series colour at all.

    ``value_fmt`` sets how ``value_labels`` are written. The default drops
    trailing zeros, which is wrong whenever the figure accompanies a table: a
    reader has to see the same string in both places, so pass the table's own
    precision (``"%.1f"``, ``"%.2f"``) rather than letting 92.0 print as 92.
    """
    if colors is not None and len(colors) != len(series):
        raise ValueError(f"{len(colors)} colours for {len(series)} series")
    if not series:
        raise ValueError("no series given")
    n_categories = len(categories)
    for name, values in series.items():
        if len(values) != n_categories:
            raise ValueError(f"series {name!r} has {len(values)} values, expected {n_categories}")

    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.0))

    positions = np.arange(n_categories)
    n_series = len(series)
    width = min(0.8 / n_series, 0.38)
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * width

    for series_index, (name, values) in enumerate(series.items()):
        if per_category_color:
            category_colors = [color_for(c, i) for i, c in enumerate(categories)]
            alpha = 0.42 + 0.58 * (series_index / max(n_series - 1, 1))
        else:
            color = (
                colors[series_index] if colors is not None
                else PALETTE[series_index % len(PALETTE)]
            )
            alpha = 1.0
        bars = ax.bar(
            positions + offsets[series_index],
            values,
            width,
            color=category_colors if per_category_color else color,
            alpha=alpha,
            edgecolor="none",
            label=name,
            yerr=None if errors is None else errors.get(name),
            capsize=2.5 if errors else 0,
            error_kw={"elinewidth": 0.8, "ecolor": MUTED},
        )
        if value_labels:
            ax.bar_label(bars, fmt=value_fmt, fontsize=mpl.rcParams["font.size"] - 2.5, padding=2)

    ax.set_xticks(positions, categories, rotation=40 if max(map(len, categories)) > 6 else 0,
                  ha="right" if max(map(len, categories)) > 6 else "center")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    if per_category_color:
        handles = [
            mpl.patches.Patch(
                facecolor=MUTED,
                alpha=0.42 + 0.58 * (i / max(n_series - 1, 1)),
                label=name,
            )
            for i, name in enumerate(series)
        ]
        ax.legend(handles=handles, loc="best")
    elif n_series > 1:
        ax.legend(loc="best")
    return ax


def plot_box_by_class(
    groups: dict[str, Sequence[float]],
    ax: Axes | None = None,
    *,
    ylabel: str = "",
    title: str | None = None,
    horizontal: bool = False,
    show_means: bool = True,
    log_scale: bool = False,
) -> Axes:
    """Box plot per class, coloured with the fixed class palette.

    Whiskers span 1.5 IQR; points beyond are drawn individually rather than
    hidden, because outlier texts are exactly what length analysis is looking
    for.
    """
    if not groups:
        raise ValueError("no groups given")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.0))

    names = list(groups)
    data = [np.asarray(groups[name], dtype=float) for name in names]
    positions = np.arange(len(names))

    artists = ax.boxplot(
        data,
        positions=positions,
        widths=0.62,
        vert=not horizontal,
        patch_artist=True,
        showmeans=show_means,
        meanprops={
            "marker": "D",
            "markersize": 3.2,
            "markerfacecolor": BACKGROUND,
            "markeredgecolor": INK,
            "markeredgewidth": 0.8,
        },
        medianprops={"color": INK, "linewidth": 1.2},
        whiskerprops={"color": MUTED, "linewidth": 0.8},
        capprops={"color": MUTED, "linewidth": 0.8},
        flierprops={
            "marker": "o",
            "markersize": 2.0,
            "markerfacecolor": MUTED,
            "markeredgecolor": "none",
            "alpha": 0.45,
        },
    )
    for index, patch in enumerate(artists["boxes"]):
        patch.set_facecolor(tint(color_for(names[index], index)))
        patch.set_edgecolor(MUTED)
        patch.set_linewidth(0.7)

    rotate = max(map(len, names)) > 6
    if horizontal:
        ax.set_yticks(positions, names)
        ax.set_xlabel(ylabel)
        ax.xaxis.grid(True)
        if log_scale:
            ax.set_xscale("log")
    else:
        ax.set_xticks(positions, names, rotation=40 if rotate else 0,
                      ha="right" if rotate else "center")
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True)
        if log_scale:
            ax.set_yscale("log")

    ax.set_axisbelow(True)
    if title:
        ax.set_title(title)
    return ax


def plot_training_curves(
    history: dict[str, Sequence[float]],
    ax: Axes | None = None,
    *,
    title: str | None = None,
    xlabel: str = "Epoch",
    ylabel: str = "",
) -> Axes:
    """Training curves. Validation series are drawn dashed, training solid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.8))
    for index, (name, values) in enumerate(history.items()):
        is_validation = name.startswith("val") or "valid" in name
        ax.plot(
            np.arange(1, len(values) + 1),
            values,
            label=name,
            color=PALETTE[(index // 2 if is_validation else index) % len(PALETTE)]
            if len(history) > 2
            else PALETTE[0 if not is_validation else 2],
            linestyle="--" if is_validation else "-",
            linewidth=1.4,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    return ax


def plot_metric_comparison(
    models: Sequence[str],
    values: Sequence[float],
    ax: Axes | None = None,
    *,
    errors: Sequence[float] | None = None,
    metric: str = "macro F1",
    highlight: str | None = None,
    title: str | None = None,
) -> Axes:
    """Horizontal model comparison, sorted ascending. Highlight one model."""
    order = np.argsort(values)
    ordered_models = [models[i] for i in order]
    ordered_values = [values[i] for i in order]
    ordered_errors = None if errors is None else [errors[i] for i in order]

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 0.45 * len(models) + 1.4))

    colors = [PALETTE[2] if m == highlight else PALETTE[0] for m in ordered_models]
    bars = ax.barh(
        np.arange(len(ordered_models)),
        ordered_values,
        color=colors,
        edgecolor="none",
        height=0.62,
        xerr=ordered_errors,
        capsize=2.5,
        error_kw={"elinewidth": 0.8, "ecolor": MUTED},
    )
    ax.bar_label(bars, fmt="%.3f", fontsize=mpl.rcParams["font.size"] - 2, padding=3)
    ax.set_yticks(np.arange(len(ordered_models)), ordered_models)
    ax.set_xlabel(metric)
    ax.set_xlim(0, min(1.0, max(ordered_values) * 1.18))
    if title:
        ax.set_title(title)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    return ax


def plot_model_bars(
    models: Sequence[str],
    values: Sequence[float],
    ax: Axes | None = None,
    *,
    reference: Sequence[float] | None = None,
    reference_label: str = "reference",
    values_label: str = "ours",
    highlight: str | None = None,
    metric: str = "Accuracy [%]",
    xmax: float = 100.0,
    decimals: int = 1,
    reference_decimals: int | None = None,
    decimal_comma: bool = True,
) -> Axes:
    """Horizontal bars on a 0–``xmax`` scale, in the order given (top to bottom).

    With ``reference`` each model gets a pale track of the reference length
    (e.g. an accuracy claimed in the literature) behind its own bar, so the
    gap reads directly. With ``highlight`` one model is drawn dark and the
    rest light. The axis always starts at zero.
    """
    ax = ax or plt.gca()
    size = mpl.rcParams["font.size"]
    def fmt(v: float, places: int = decimals) -> str:
        return f"{v:.{places}f}".replace(".", "," if decimal_comma else ".") + "%"

    ref_places = decimals if reference_decimals is None else reference_decimals
    y = np.arange(len(models))[::-1]
    ours = "#3a6f9a"
    height = 0.56
    if reference is not None:
        ax.barh(y, reference, height=height, color="#dde3e9", edgecolor="none", zorder=1, label=reference_label)
        for yi, ref in zip(y, reference):
            ax.text(ref + xmax * 0.012, yi, fmt(ref, ref_places), va="center", ha="left", fontsize=size - 1,
                    color="#55616e", weight="bold")
    colours = [ours if (highlight is None or m == highlight) else "#9fc3da" for m in models]
    ax.barh(y, values, height=height, color=colours, edgecolor="none", zorder=2, label=values_label)
    for yi, v, m in zip(y, values, models):
        inside = reference is not None
        ax.text(v - xmax * 0.012 if inside else v + xmax * 0.012, yi, fmt(v), va="center",
                ha="right" if inside else "left", fontsize=size, weight="bold",
                color="#ffffff" if inside else INK, zorder=3)
    ax.set_yticks(y, models)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax * 1.1)  # room for the value labels past the longest bar
    ax.set_xticks(np.linspace(0, xmax, 6))
    ax.set_xlabel(metric)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    if reference is not None:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, handlelength=1.2,
                  fontsize=size - 1)
    return ax


def plot_architecture(
    layers: Sequence[dict],
    ax: Axes | None = None,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    input_label: str | None = None,
    output_label: str | None = None,
) -> Axes:
    """Block diagram of a neural network, one box per layer, top to bottom.

    Each entry in ``layers`` is a dict with:

    ``kind``        layer type, e.g. ``"LSTM"`` (required)
    ``detail``      configuration string, e.g. ``"hidden 100, 2 layers"``
    ``shape``       output shape label, e.g. ``"batch × 30 × 300"``
    ``parameters``  parameter count
    ``frozen``      True to mark the block as not trained

    Drawn rather than exported from a graph library so it matches the rest of
    the figures and stays legible at journal column width; a computation-graph
    dump is unreadable in print.
    """
    if not layers:
        raise ValueError("no layers given")

    if ax is None:
        _, ax = plt.subplots(figsize=(6.8, 0.86 * len(layers) + 1.9))

    box_height, gap = 0.62, 0.30
    step = box_height + gap
    total = len(layers) * step

    def band(y: float, text: str, color: str) -> None:
        ax.text(0.5, y, text, ha="center", va="center", fontsize=8.5,
                color=color, style="italic")

    top = total
    if input_label:
        band(top + 0.22, input_label, MUTED)

    for position, layer in enumerate(layers):
        y = total - (position + 1) * step + gap
        frozen = bool(layer.get("frozen"))
        kind = str(layer.get("kind", "?"))

        # Parameterised layers carry the model's capacity, so they are filled;
        # activations and dropout are outlined, which keeps the eye on where
        # the weights are.
        parameters = int(layer.get("parameters", 0) or 0)
        if parameters == 0:
            face, edge, ink = BACKGROUND, MUTED, MUTED
        elif frozen:
            face, edge, ink = "#e9edf1", MUTED, INK
        else:
            face, edge, ink = "#dcecf7", PALETTE[0], INK

        ax.add_patch(
            mpl.patches.FancyBboxPatch(
                (0.06, y), 0.88, box_height,
                boxstyle="round,pad=0.006,rounding_size=0.02",
                linewidth=0.9, facecolor=face, edgecolor=edge,
            )
        )

        label = kind + (" · frozen" if frozen else "")
        ax.text(0.11, y + box_height * 0.62, label,
                ha="left", va="center", fontsize=9.5, color=ink, fontweight="bold")
        detail = str(layer.get("detail") or "")
        if detail:
            ax.text(0.11, y + box_height * 0.26, detail,
                    ha="left", va="center", fontsize=8, color=MUTED)

        shape = str(layer.get("shape") or "")
        if shape:
            ax.text(0.89, y + box_height * 0.62, shape,
                    ha="right", va="center", fontsize=8.5, color=ink)
        if parameters:
            ax.text(0.89, y + box_height * 0.26, f"{parameters:,} params",
                    ha="right", va="center", fontsize=7.5, color=MUTED)

        if position < len(layers) - 1:
            ax.annotate(
                "", xy=(0.5, y - gap + 0.04), xytext=(0.5, y),
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 0.9},
            )

    if output_label:
        band(-0.30, output_label, MUTED)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55 if output_label else -0.15, total + (0.5 if input_label else 0.15))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, pad=14)
    if subtitle:
        ax.text(0.5, total + (0.5 if input_label else 0.15), subtitle,
                ha="center", va="bottom", fontsize=8.5, color=MUTED)
    return ax


def plot_radar(
    categories: Sequence[str],
    series: dict[str, Sequence[float]],
    ax: Axes | None = None,
    *,
    title: str | None = None,
    colors: Sequence[str] | None = None,
    rmax: float | None = None,
    rlabel: str = "",
    fill_alpha: float = 0.10,
    per_category_color: bool = False,
    legend_position: str = "right",
) -> Axes:
    """Radar chart comparing profiles across a small set of categories.

    Each category gets an axis radiating from a common centre and each series
    becomes a closed line, which makes the *shape* of a profile comparable at a
    glance. Read it as a shape comparison only: the area enclosed depends on
    the order of the axes, which is arbitrary, so a radar chart never replaces
    the table it accompanies.

    Prefer :func:`plot_grouped_bars` when the question is about magnitudes,
    when there are fewer than four categories (three axes draw a triangle and
    say no more than a table row), or when two conditions have to be set side
    by side — a radar chart cannot show a contrast without doubling the lines.

    ``per_category_color`` colours each series by its own name where that name
    is a known class; otherwise series take successive palette colours.

    ``legend_position`` is ``"right"``, ``"bottom"`` or ``"none"``. A legend on
    the right shifts the disc off the centre of the figure, which is visible
    once the figure sits in a column of text; ``"bottom"`` keeps the disc
    centred and is the better choice for a plate that stands on its own.
    """
    if legend_position not in {"right", "bottom", "none"}:
        raise ValueError(f"legend_position must be right, bottom or none, got {legend_position!r}")
    if not series:
        raise ValueError("no series given")
    n_categories = len(categories)
    if n_categories < 3:
        raise ValueError(f"a radar chart needs at least 3 axes, got {n_categories}")
    if colors is not None and len(colors) != len(series):
        raise ValueError(f"{len(colors)} colours for {len(series)} series")
    for name, values in series.items():
        if len(values) != n_categories:
            raise ValueError(f"series {name!r} has {len(values)} values, expected {n_categories}")

    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 4.6), subplot_kw={"projection": "polar"})
    elif ax.name != "polar":
        raise ValueError("plot_radar needs a polar axes (subplot_kw={'projection': 'polar'})")

    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])

    for index, (name, values) in enumerate(series.items()):
        if colors is not None:
            color = colors[index]
        elif per_category_color:
            color = color_for(name, index)
        else:
            color = PALETTE[index % len(PALETTE)]
        closed_values = np.concatenate([np.asarray(values, dtype=float), np.asarray(values[:1], dtype=float)])
        ax.plot(closed_angles, closed_values, color=color, linewidth=1.2, label=name)
        ax.fill(closed_angles, closed_values, color=color, alpha=fill_alpha)

    # Clockwise from the top reads as a list rather than as a compass, which is
    # what the categories actually are.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles, [c.capitalize() for c in categories])
    ax.tick_params(axis="x", pad=6, colors=INK)
    ax.tick_params(axis="y", labelsize=mpl.rcParams["font.size"] - 2, colors=MUTED)

    if rmax is not None:
        ax.set_ylim(0, rmax)
    # The radial ticks sit on their own spoke, between two category axes, so
    # they never run along a data line. The unit goes just outside the last
    # ring rather than next to it, which would collide with the top tick.
    ax.set_rlabel_position(180 / n_categories)
    if rlabel:
        ax.text(np.deg2rad(180 / n_categories), ax.get_ylim()[1] * 1.26, rlabel,
                ha="center", va="center", fontsize=mpl.rcParams["font.size"] - 2,
                color=MUTED, clip_on=False)

    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.spines["polar"].set_color(GRID)
    ax.spines["polar"].set_linewidth(0.6)
    ax.set_facecolor(BACKGROUND)

    if title:
        ax.set_title(title, pad=18)
    if len(series) > 1 and legend_position != "none":
        if legend_position == "bottom":
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06),
                      ncol=min(len(series), 3), frameon=False,
                      columnspacing=1.6, handletextpad=0.6)
        else:
            ax.legend(loc="upper left", bbox_to_anchor=(1.06, 1.05), frameon=False)
    return ax


def plot_mel_spectrogram(
    spectrogram: np.ndarray,
    ax: Axes | None = None,
    *,
    duration_seconds: float,
    mel_frequencies: Sequence[float] | None = None,
    title: str | None = None,
    title_color: str | None = None,
    xlabel: str | None = "Time [s]",
    ylabel: str | None = "Mel frequency [Hz]",
    cmap: str | mpl.colors.Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    n_yticks: int = 4,
) -> Axes:
    """Render one mel spectrogram already converted to decibels.

    Parameters
    ----------
    spectrogram
        Array of shape ``(n_mels, n_frames)`` in dB. Nothing is rescaled,
        smoothed, or resampled here — pass exactly what the feature pipeline
        produced.
    duration_seconds
        Length of the analysis window the columns span. Used for the time axis
        only; it must match the window the features were computed over.
    mel_frequencies
        Centre frequency of every mel band, length ``n_mels`` (from
        ``librosa.mel_frequencies``). Without it the vertical axis is labelled
        by band index instead of frequency, since mel bands are not linear in Hz
        and a linear Hz axis would misplace every tick.
    cmap
        Defaults to the thesis sequential ramp, which is what keeps a figure
        looking like the rest of the document. Override only to reproduce the
        convention of another field — ``"magma"`` is what ``librosa.display``
        draws spectrograms with, and a reader of the speech literature expects
        it. Whichever is chosen, use it for every spectrogram in the document.
    vmin, vmax
        Colour limits. Pass the SAME pair to every panel of a multi-panel
        figure — panels on independent scales cannot be compared, which is
        usually the whole point of showing several.
    title_color
        Colour for the panel title, e.g. ``color_for("anger")`` when the panels
        are emotion classes.
    """
    spectrogram = np.asarray(spectrogram, dtype=float)
    if spectrogram.ndim != 2:
        raise ValueError(f"expected a 2-D (n_mels, n_frames) array, got {spectrogram.shape}")
    n_mels = spectrogram.shape[0]
    if mel_frequencies is not None and len(mel_frequencies) != n_mels:
        raise ValueError(f"{len(mel_frequencies)} mel frequencies for {n_mels} bands")

    ax = ax or plt.gca()
    image = ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=SEQUENTIAL if cmap is None else cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(0.0, float(duration_seconds), -0.5, n_mels - 0.5),
    )

    if mel_frequencies is None:
        ax.set_ylabel(ylabel if ylabel is not None else "Mel band")
    else:
        positions = np.linspace(0, n_mels - 1, n_yticks).round().astype(int)
        ax.set_yticks(positions)
        ax.set_yticklabels([f"{mel_frequencies[i] / 1000:.1f}k" if mel_frequencies[i] >= 1000
                            else f"{mel_frequencies[i]:.0f}" for i in positions])
        if ylabel is not None:
            ax.set_ylabel(ylabel)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(MUTED)
    ax.grid(False)
    ax.tick_params(length=2.5, width=0.6, color=MUTED)

    if title:
        ax.set_title(title, color=title_color or INK, pad=4)

    if colorbar:
        bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        bar.outline.set_linewidth(0.6)
        bar.outline.set_edgecolor(MUTED)
        bar.ax.tick_params(length=2.5, width=0.6, color=MUTED)
        if colorbar_label:
            bar.set_label(colorbar_label)
    return ax


def annotate_significance(
    ax: Axes,
    x_from: float,
    x_to: float,
    y: float,
    p_value: float,
    *,
    height: float = 0.02,
) -> None:
    """Draw a significance bracket labelled with the exact p-value.

    The exact value is printed rather than stars — reviewers ask for it, and
    a star hides whether p was 0.049 or 0.0001.
    """
    ax.plot(
        [x_from, x_from, x_to, x_to],
        [y, y + height, y + height, y],
        linewidth=0.8,
        color=MUTED,
    )
    text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    ax.text(
        (x_from + x_to) / 2,
        y + height,
        text,
        ha="center",
        va="bottom",
        fontsize=mpl.rcParams["font.size"] - 2,
        color=INK,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Donut (share of a whole)
# ---------------------------------------------------------------------------


def plot_donut(
    labels: Sequence[str],
    values: Sequence[float],
    ax: Axes | None = None,
    *,
    center_value: str | None = None,
    center_label: str | None = None,
    start_angle: float = 90.0,
    font_size: float | None = None,
    weight: str = "normal",
    decimal_comma: bool = True,
    min_label_gap: float = 0.36,
) -> Axes:
    """Donut chart of shares, every slice labelled directly as ``label  n (p%)``.

    Zero-valued slices are not drawn (they have no angle) — list them in a note.
    Slices go counter-clockwise from ``start_angle`` in the order given; colours
    come from :func:`color_for`, so a class keeps its colour. Labels on the same
    side are pushed apart by at least ``min_label_gap`` (axes units).
    """
    ax = ax or plt.gca()
    size = font_size or mpl.rcParams["font.size"]
    pairs = [(label, float(value)) for label, value in zip(labels, values) if value > 0]
    total = sum(value for _, value in pairs)
    colours = [color_for(label, i) for i, (label, _) in enumerate(pairs)]
    wedges, _ = ax.pie(
        [value for _, value in pairs], colors=colours, startangle=start_angle, counterclock=True,
        wedgeprops={"width": 0.36, "edgecolor": BACKGROUND, "linewidth": 2.0},
    )
    ax.set_aspect("equal")

    placed = []
    for (label, value), wedge in zip(pairs, wedges):
        mid = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        share = 100 * value / total
        pct = f"{share:.1f}".replace(".", "," if decimal_comma else ".")
        count = f"{value:g}"
        placed.append([np.cos(mid), np.sin(mid), 1.22 * np.sin(mid), label, f"{count} ({pct}%)"])

    for side in (1, -1):
        group = sorted([p for p in placed if np.sign(p[0] or 1) == side], key=lambda p: -p[2])
        for above, below in zip(group, group[1:]):
            if above[2] - below[2] < min_label_gap:
                shift = (min_label_gap - (above[2] - below[2])) / 2
                above[2] += shift
                below[2] -= shift

    for x, y, ty, label, detail in placed:
        side = 1 if x >= 0 else -1
        tx = side * 1.32
        ax.plot([0.82 * x, 1.04 * x, tx - side * 0.06], [0.82 * y, 1.04 * y, ty], color=MUTED, lw=0.9,
                solid_capstyle="round", zorder=1)
        ha = "left" if side > 0 else "right"
        ax.text(tx, ty, label, fontsize=size, weight="bold", ha=ha, va="bottom", color=INK)
        ax.text(tx, ty, detail, fontsize=size - 1, weight=weight, ha=ha, va="top", color=INK)

    if center_value:
        ax.text(0, 0.04, center_value, fontsize=size * 2.3, weight="bold", ha="center", va="bottom", color=INK)
    if center_label:
        ax.text(0, 0.0, center_label, fontsize=size, weight=weight, ha="center", va="top", color=INK)
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.3, 1.2)
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# Table as an image (slides)
# ---------------------------------------------------------------------------


class Chips:
    """Table cell: a wrapped row of class labels, each in its fixed class colour."""

    def __init__(self, labels: Sequence[str]):
        self.labels = [str(label) for label in labels]


class Check:
    """Table cell: a yes / no mark."""

    def __init__(self, value: bool):
        self.value = bool(value)


class Muted:
    """Table cell: secondary text such as ``no data`` — muted italic."""

    def __init__(self, text: str):
        self.text = str(text)


_YES = "#2f9e63"
_NO = "#c24a3f"
_BAND = "#f3f6f9"
_GROUP_BAND = "#e8f1f7"
_HEAD_BAND = "#3a6f9a"  # step 4 of the sequential ramp; white text keeps ~5.6:1 contrast


def _text_size(text: str, size: float, weight: str = "normal", style: str = "normal") -> tuple[float, float]:
    """Width and height of ``text`` in points, measured with the Agg renderer."""
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import FontProperties

    renderer = RendererAgg(10, 10, 72)  # 72 dpi: one pixel is one point
    prop = FontProperties(family=mpl.rcParams["font.serif"], size=size, weight=weight, style=style)
    width, height, _ = renderer.get_text_width_height_descent(text, prop, ismath=False)
    return width, height


def _wrap_words(text: str, size: float, max_width: float, weight: str = "normal", style: str = "normal") -> list[str]:
    if "\n" in text:  # explicit line breaks: wrap each paragraph on its own
        return [line for part in text.split("\n") for line in _wrap_words(part, size, max_width, weight, style)]
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}".strip()
        if current and _text_size(candidate.replace("\u00a0", " "), size, weight, style)[0] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    # U+00A0 only glues words together for wrapping; Charter draws it too wide.
    return [line.replace("\u00a0", " ") for line in lines] or [""]


def plot_table(
    header: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    col_widths: Sequence[float],
    align: Sequence[str] | None = None,
    title: str | None = None,
    title_prefix: str | None = None,
    note: str | None = None,
    highlight_rows: Sequence[int] = (),
    group_breaks: Sequence[int] = (),
    font_size: float = 10.0,
    zebra: bool = True,
    row_spans: Sequence[tuple[int, int, int]] = (),
    body_weight: str = "normal",
) -> Figure:
    """Booktabs-style table rendered as a figure, for slides and posters.

    ``col_widths`` are in points. Cells may be plain strings, :class:`Chips`
    (class labels coloured with :func:`color_for`), :class:`Check` or
    :class:`Muted`. ``highlight_rows`` get a tinted band; ``group_breaks`` are
    row indices that get a thin rule *above* them. ``row_spans`` holds
    ``(column, first_row, n_rows)`` merges: the cell in ``first_row`` is drawn
    centred over the whole span and the others in the span are ignored.
    ``body_weight="bold"`` thickens body text for projectors; the header then
    turns into a dark band with white text so it stays distinct.
    Nothing is reordered or
    abbreviated — every cell is drawn exactly as given.
    """
    if any(len(row) != len(header) for row in rows):
        raise ValueError("every row must have the same width as the header")
    align = list(align) if align is not None else ["left"] * len(header)
    pad_x, pad_y = 7.0, 7.0
    line_h = font_size * 1.32
    chip_size = font_size - 1.2
    chip_h = chip_size * 1.75
    chip_gap_x, chip_gap_y, chip_pad = 3.5, 3.5, 5.0
    head_size = font_size - 0.5
    head_weight = "bold"  # Charter Black (900) lacks Polish diacritics
    dark_head = body_weight == "bold"
    lefts = np.concatenate([[0.0], np.cumsum(col_widths)])
    total_w = float(lefts[-1])

    def layout(cell: object, width: float) -> tuple[float, object]:
        inner = width - 2 * pad_x
        if isinstance(cell, Chips):
            placed, x, y = [], 0.0, 0.0
            for label in cell.labels:
                w = _text_size(label, chip_size, body_weight)[0] + 2 * chip_pad
                if x and x + w > inner:
                    x, y = 0.0, y + chip_h + chip_gap_y
                placed.append((label, x, y, w))
                x += w + chip_gap_x
            return y + chip_h, placed
        if isinstance(cell, Check):
            return line_h, None
        text, style = (cell.text, "italic") if isinstance(cell, Muted) else (str(cell), "normal")
        lines = _wrap_words(text, font_size, inner, weight=body_weight, style=style)
        return line_h * len(lines), lines

    header_lines = [_wrap_words(h, head_size, w - 2 * pad_x, weight=head_weight) for h, w in zip(header, col_widths)]
    header_h = max(len(lines) for lines in header_lines) * head_size * 1.3 + 2 * pad_y
    laid = [[layout(cell, w) for cell, w in zip(row, col_widths)] for row in rows]
    spans = {(c, r0): n for c, r0, n in row_spans}
    covered = {(c, r0 + k) for c, r0, n in row_spans for k in range(n)}
    row_h = [
        max((h for c, (h, _) in enumerate(row) if (c, r) not in covered), default=line_h) + 2 * pad_y
        for r, row in enumerate(laid)
    ]
    for (c, r0), n in spans.items():  # grow the last spanned row if the merged cell does not fit
        need = laid[r0][c][0] + 2 * pad_y - sum(row_h[r0:r0 + n])
        if need > 0:
            row_h[r0 + n - 1] += need

    title_h = 0.0
    title_lines: list[str] = []
    if title:
        title_lines = _wrap_words(title, font_size + 1.5, total_w)
        title_h = len(title_lines) * (font_size + 1.5) * 1.35 + 10
    note_h = font_size * 2.2 if note else 0.0
    total_h = title_h + header_h + sum(row_h) + note_h + 4

    fig = plt.figure(figsize=(total_w / 72, total_h / 72))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    ax.axis("off")

    y = 0.0
    if title:
        for i, line in enumerate(title_lines):
            prefix = title_prefix if (i == 0 and title_prefix) else None
            x0 = 0.0
            if prefix:
                ax.text(0, y, prefix, fontsize=font_size + 1.5, weight="bold", va="top", color=INK)
                x0 = _text_size(prefix + " ", font_size + 1.5, weight="bold")[0]
            ax.text(x0, y, line, fontsize=font_size + 1.5, va="top", color=INK)
            y += (font_size + 1.5) * 1.35
        y += 10

    def hrule(yy: float, lw: float, color: str = INK) -> None:
        ax.plot([0, total_w], [yy, yy], color=color, lw=lw, solid_capstyle="butt")

    if dark_head:
        ax.add_patch(mpl.patches.Rectangle((0, y), total_w, header_h, color=_HEAD_BAND, lw=0, zorder=0))
    else:
        hrule(y, 1.4)
    for lines, left, width, a in zip(header_lines, lefts, col_widths, align):
        x = {"left": left + pad_x, "center": left + width / 2, "right": left + width - pad_x}[a]
        for k, line in enumerate(lines):
            ax.text(x, y + pad_y + k * head_size * 1.3, line, fontsize=head_size, weight=head_weight,
                    ha=a, va="top", color="#ffffff" if dark_head else INK)
    y += header_h
    if not dark_head:
        hrule(y, 0.7)

    for r, (row, cells, h) in enumerate(zip(rows, laid, row_h)):
        if r in highlight_rows:
            ax.add_patch(mpl.patches.Rectangle((0, y), total_w, h, color=_GROUP_BAND, lw=0, zorder=0))
        elif zebra and r % 2 == 1:
            ax.add_patch(mpl.patches.Rectangle((0, y), total_w, h, color=_BAND, lw=0, zorder=0))
        if r in group_breaks:
            hrule(y, 0.5, MUTED)
        for c, (cell, (content_h, content), left, width, a) in enumerate(zip(row, cells, lefts, col_widths, align)):
            if (c, r) in covered and (c, r) not in spans:
                continue
            span_h = sum(row_h[r:r + spans[(c, r)]]) if (c, r) in spans else h
            top = y + (span_h - content_h) / 2
            if isinstance(cell, Chips):
                for label, cx, cy, cw in content:
                    colour = color_for(label, 6)
                    if canonical_class(label) not in EKMAN_COLORS and canonical_class(label) not in EXTRA_COLORS:
                        colour = EXTRA_COLORS["other"]
                    box = mpl.patches.FancyBboxPatch(
                        (left + pad_x + cx, top + cy), cw, chip_h,
                        boxstyle=f"round,pad=0,rounding_size={chip_h / 2}",
                        facecolor=mpl.colors.to_rgba(colour, 0.16), edgecolor=colour, lw=1.0 if body_weight == "bold" else 0.7, zorder=2,
                    )
                    ax.add_patch(box)
                    ax.text(left + pad_x + cx + cw / 2, top + cy + chip_h / 2 - chip_size * 0.06, label, fontsize=chip_size,
                            ha="center", va="center", color=INK, weight=body_weight, zorder=3)
                continue
            x = {"left": left + pad_x, "center": left + width / 2, "right": left + width - pad_x}[a]
            if isinstance(cell, Check):
                ax.text(x, top + line_h / 2, "✓" if cell.value else "✗", fontsize=font_size + 3,
                        family="DejaVu Sans", weight="bold", ha=a, va="center",
                        color=_YES if cell.value else _NO)
                continue
            muted = isinstance(cell, Muted)
            for k, line in enumerate(content):
                ax.text(x, top + k * line_h, line, fontsize=font_size, ha=a, va="top",
                        color=(("#55616e" if dark_head else MUTED) if muted else INK),
                        style="italic" if muted else "normal",
                        weight=body_weight)
        y += h
    hrule(y, 1.4)

    if note:
        ax.text(0, y + font_size * 0.8, note, fontsize=font_size - 1.5, va="top", color=MUTED)
    return fig


def save_figure(figure: Figure, path: str | Path, *, formats: Iterable[str] = ("pdf", "png")) -> list[Path]:
    """Save to every requested format. ``path`` is given WITHOUT an extension.

    PDF is the one that goes into the thesis; PNG is for quick review.
    """
    stem = Path(path)
    if stem.suffix:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for extension in formats:
        target = stem.with_suffix(f".{extension}")
        figure.savefig(target)
        written.append(target)
    return written


def latex_table(
    rows: Sequence[Sequence[object]],
    header: Sequence[str],
    *,
    caption: str,
    label: str,
    column_format: str | None = None,
    bold_max_columns: Sequence[int] = (),
    notes: str | None = None,
) -> str:
    """Render a booktabs table matching the figure style.

    ``bold_max_columns`` bolds the largest numeric value in those column
    indices — the usual "best model per metric" convention.
    """
    if any(len(row) != len(header) for row in rows):
        raise ValueError("every row must have the same width as the header")

    body = [[str(cell) for cell in row] for row in rows]
    for column in bold_max_columns:
        numeric: list[tuple[int, float]] = []
        for row_index, row in enumerate(rows):
            try:
                numeric.append((row_index, float(row[column])))
            except (TypeError, ValueError):
                continue
        if numeric:
            best = max(numeric, key=lambda pair: pair[1])[0]
            body[best][column] = rf"\textbf{{{body[best][column]}}}"

    if column_format is None:
        column_format = "l" + "r" * (len(header) - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        rf"  \begin{{tabular}}{{{column_format}}}",
        r"    \toprule",
        "    " + " & ".join(header) + r" \\",
        r"    \midrule",
    ]
    lines += ["    " + " & ".join(row) + r" \\" for row in body]
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    if notes:
        lines.append(rf"  \par\smallskip\footnotesize {notes}")
    lines.append(r"\end{table}")
    return "\n".join(lines)
