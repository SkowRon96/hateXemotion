"""Radar charts for the emotion study, in the dissertation's visual system.

One chart per Polish model, offensive statements only, in the axis order of the
published figures. The English corpus appears on every chart as a point of
reference, scored by the English model — so each chart carries two models and
its caption has to say so.

The three charts are meant to be read side by side. That is the point of the
study — three Polish models disagree about which emotion accompanies offensive
language, and the disagreement is easier to see as three shapes than as three
blocks of a table.

Run from this directory:

    python make_figures_emotions.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from research_style import apply_style, plot_radar, save_figure

SUMMARY_PATH = Path("results/all_data_results.xlsx")
OUTPUT_DIR = Path("figures")

#: Axis order of the published figures, taken from ``CONSTANTS.EMOTION_CATEGORIES``.
#: It is not the label-id order used elsewhere in the thesis (there anger comes
#: before disgust); kept as published because the order of a radar chart's axes
#: changes the shape a reader compares.
EMOTIONS = ("neutral", "happiness", "sadness", "fear", "disgust", "surprise", "anger")

DATASETS = ("MHC", "BAN-PL", "PolEval 2019", "HateSpeechPl")

#: The English corpus, scored by the English model, is added to every chart as
#: a point of reference — exactly as the published figures did. The line does
#: NOT come from the Polish model the chart is about, so each caption has to
#: name both models, the way the published titles did.
ENGLISH_ROW = "Hate Speech Offensive"
ENGLISH_MODEL = "Emotion English DistilRoBERTa-base"

#: Each Polish model, the summary rows it wrote, and the file to write.
#: The audited model has one row per seed and is averaged; the other two are
#: deterministic and have a single row each.
MODELS = (
    ("chatbot", "Chatbot", ("{dataset}",)),
    ("roberta_polish6", "Emotion_RoBERTa_polish6", ("{dataset} v2",)),
    ("polish_roberta", "Polish RoBERTa (audited)",
     ("{dataset} v3 s7", "{dataset} v3 s17", "{dataset} v3 s27")),
)


def shares(summary: pd.DataFrame, row_name: str, scope: str = "hate") -> list[float] | None:
    """Percentage per emotion for one summary row, or ``None`` if the row is empty.

    The denominator is the number of statements the model actually labelled,
    not the size of the dataset: a few statements exceed the model's input
    limit and the classes would otherwise not sum to a hundred.
    """
    matches = summary[summary["dataset"] == row_name]
    if matches.empty:
        raise KeyError(f"no summary row named {row_name!r}")
    row = matches.iloc[0]
    total = row[f"{scope}_emotion"]
    if not total or pd.isna(total) or total == 0:
        return None
    return [100 * row[f"{scope}_emotion_{name}"] / total for name in EMOTIONS]


def main() -> int:
    apply_style()
    summary = pd.read_excel(SUMMARY_PATH)
    OUTPUT_DIR.mkdir(exist_ok=True)

    for slug, title, patterns in MODELS:
        series = {}
        for dataset in DATASETS:
            # Averaging over seeds happens here rather than in the summary file,
            # so that the per-seed rows stay available for the spread reported
            # in the text.
            runs = [shares(summary, p.format(dataset=dataset)) for p in patterns]
            if any(run is None for run in runs):
                continue
            series[f"[PL] {dataset}"] = list(np.mean(runs, axis=0))

        series[f"[EN] {ENGLISH_ROW}"] = shares(summary, ENGLISH_ROW)

        fig, ax = plt.subplots(figsize=(5.4, 5.4), subplot_kw={"projection": "polar"})
        plot_radar(EMOTIONS, series, ax=ax, rmax=100, rlabel="[%]",
                   legend_position="bottom")
        save_figure(fig, OUTPUT_DIR / f"emotions_hate_radar_{slug}")
        plt.close(fig)

        print(f"{title}  ({len(patterns)} run(s) per dataset) + {ENGLISH_MODEL}")
        for name, values in series.items():
            top = max(range(len(EMOTIONS)), key=lambda i: values[i])
            print(f"  {name:<28} dominant {EMOTIONS[top]} {values[top]:.1f}")

    print(f"\nWrote {len(MODELS)} radar charts to {OUTPUT_DIR}/")
    print("Each chart mixes two models — the caption must name both.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
