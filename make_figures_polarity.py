"""Figures for the polarity study, in the dissertation's visual system.

Replaces ``charter.py``, which carried its own palette and therefore produced
figures that could not sit next to those of the other chapters. Everything here
goes through ``research_style``: one font, one palette, one set of conventions
across the whole thesis.

Two figures are produced from the same numbers, because they answer different
questions. The radar chart shows the polarity profile of offensive text and is
the direct counterpart of the one published in the article. The grouped bars
set offensive against non-offensive text in each corpus, which is the contrast
the conclusions actually rest on and which a radar chart cannot show without
doubling its lines.

Run from this directory:

    python make_figures_polarity.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from research_style import apply_style, plot_grouped_bars, plot_radar, save_figure

SUMMARY_PATH = Path("results/all_data_results.xlsx")
OUTPUT_DIR = Path("figures")

#: Datasets in the order used by every table of the chapter, with the English
#: one last so that the language boundary falls in one place.
DATASETS = ("MHC", "BAN-PL", "PolEval 2019", "HateSpeechPl", "Hate Speech Offensive")

#: Axis order of the published figure, taken from ``CONSTANTS.POLARITY_CATEGORIES``.
#: Kept deliberately: a reader comparing the thesis with the article has to see
#: the same shape, and the order of a radar chart's axes changes that shape.
POLARITY = ("neutral", "negative", "positive")


def shares(summary: pd.DataFrame, dataset: str, scope: str) -> dict[str, float] | None:
    """Percentage per polarity class, or ``None`` when the group is empty.

    The denominator is the number of statements that received a prediction,
    not the size of the dataset: a handful of statements are empty after
    cleaning or exceed the model's input limit, and dividing by the dataset
    size would leave the classes summing to less than a hundred.
    """
    row = summary[summary["dataset"] == dataset].iloc[0]
    total = row[f"{scope}_sentiment"]
    if not total or pd.isna(total) or total == 0:
        return None
    return {name: 100 * row[f"{scope}_sentiment_{name}"] / total for name in POLARITY}


def main() -> int:
    apply_style()
    summary = pd.read_excel(SUMMARY_PATH)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Figure 1: polarity profile of offensive statements ----------------
    # The legend carries the language code, because the chapter reads the two
    # languages differently and the dataset names alone do not say which is
    # which. Written as a leading ``[PL]`` / ``[EN]`` exactly as the published
    # figure wrote it, so the two can be compared entry by entry.
    hate_series = {}
    for dataset in DATASETS:
        values = shares(summary, dataset, "hate")
        if values is None:
            continue
        language = summary.loc[summary["dataset"] == dataset, "language"].iloc[0]
        hate_series[f"[{language.upper()}] {dataset}"] = [values[name] for name in POLARITY]

    fig, ax = plt.subplots(figsize=(5.0, 5.2), subplot_kw={"projection": "polar"})
    plot_radar(
        POLARITY, hate_series, ax=ax, rmax=100, rlabel="[%]",
        legend_position="bottom",
    )
    save_figure(fig, OUTPUT_DIR / "polarity_hate_radar")
    plt.close(fig)

    # --- Figure 2: offensive against non-offensive -------------------------
    # HateSpeechPl is left out: it contains no non-offensive statements, so it
    # has no contrast to show and a zero bar would read as a measurement.
    contrast_datasets = [d for d in DATASETS if shares(summary, d, "no_hate") is not None]
    series = {
        "hate": [shares(summary, d, "hate")["negative"] for d in contrast_datasets],
        "non-hate": [shares(summary, d, "no_hate")["negative"] for d in contrast_datasets],
    }

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    plot_grouped_bars(
        contrast_datasets, series, ax=ax,
        ylabel="Negative polarity [%]",
        title="Negative polarity by content type",
        colors=("#264d71", "#b2d1e3"),
        value_labels=True,
        value_fmt="%.1f",
    )
    ax.set_ylim(0, 100)
    save_figure(fig, OUTPUT_DIR / "polarity_hate_vs_nonhate")
    plt.close(fig)

    print(f"Datasets on the radar: {len(hate_series)}")
    for dataset, values in hate_series.items():
        print(f"  {dataset:<24} " + "  ".join(
            f"{name} {value:.1f}" for name, value in zip(POLARITY, values)))
    print(f"\nContrast figure covers {len(contrast_datasets)} datasets "
          f"(HateSpeechPl excluded — no non-offensive statements)")
    print(f"Wrote figures to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
