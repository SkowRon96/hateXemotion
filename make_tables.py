"""LaTeX table bodies for both studies of the chapter.

Emits the row block of each results table, ready to paste between the header
and ``\\bottomrule``. Generating them rather than transcribing them by hand is
what keeps the figures and the tables reporting the same numbers, and it is the
only way the seed averages stay in step with the per-seed files.

Number conventions: the unit lives in the column header, values are bare, one decimal place for composition
shares, ``---`` where a group does not exist, and ``\\phantom{0}`` to align
columns that mix one- and two-digit integer parts.

Run from this directory:

    python make_tables.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SUMMARY_PATH = Path("results/all_data_results.xlsx")
OUTPUT_DIR = Path("tables")

POLARITY = ("positive", "negative", "neutral")
EMOTIONS = ("neutral", "happiness", "sadness", "fear", "disgust", "surprise", "anger")
SCOPES = (("Mixed", "all"), ("Hate", "hate"), ("Non-hate", "no_hate"))

DATASETS_PL = ("MHC", "BAN-PL", "PolEval 2019", "HateSpeechPl")
DATASETS_ALL = DATASETS_PL + ("Hate Speech Offensive",)

#: Emotion models, as they appear in the chapter, with the summary rows each
#: one wrote. The audited model has one row per seed and is averaged.
EMOTION_MODELS = (
    ("Chatbot", DATASETS_PL, ("{dataset}",)),
    ("Emotion\\_RoBERTa\\_polish6", DATASETS_PL, ("{dataset} v2",)),
    ("Polish RoBERTa", DATASETS_PL,
     ("{dataset} v3 s7", "{dataset} v3 s17", "{dataset} v3 s27")),
    ("Emotion English DistilRoBERTa-base", ("Hate Speech Offensive",), ("{dataset}",)),
)


def cell(value: float | None) -> str:
    """One value, padded so a column of mixed widths lines up on its axis."""
    if value is None:
        return "---"
    text = f"{value:.1f}"
    return "\\phantom{0}" + text if value < 10 else text


def shares(summary: pd.DataFrame, row_name: str, scope: str, kind: str,
           classes: tuple[str, ...]) -> list[float] | None:
    matches = summary[summary["dataset"] == row_name]
    if matches.empty:
        raise KeyError(f"no summary row named {row_name!r}")
    row = matches.iloc[0]
    total = row[f"{scope}_{kind}"]
    if not total or pd.isna(total) or total == 0:
        return None
    return [100 * row[f"{scope}_{kind}_{name}"] / total for name in classes]


def polarity_rows(summary: pd.DataFrame) -> list[str]:
    lines = []
    for dataset in DATASETS_ALL:
        for index, (label, scope) in enumerate(SCOPES):
            values = shares(summary, dataset, scope, "sentiment", POLARITY)
            first = dataset if index == 0 else ""
            cells = " & ".join(cell(v) for v in (values or [None] * len(POLARITY)))
            lines.append(f"{first} & {label} & {cells}\\tabularnewline")
    return lines


def emotion_rows(summary: pd.DataFrame) -> list[str]:
    lines = []
    for model, datasets, patterns in EMOTION_MODELS:
        for dataset_index, dataset in enumerate(datasets):
            for scope_index, (label, scope) in enumerate(SCOPES):
                runs = [shares(summary, p.format(dataset=dataset), scope, "emotion", EMOTIONS)
                        for p in patterns]
                values = None if any(r is None for r in runs) else list(np.mean(runs, axis=0))
                model_cell = model if (dataset_index == 0 and scope_index == 0) else ""
                dataset_cell = dataset if scope_index == 0 else ""
                cells = " & ".join(cell(v) for v in (values or [None] * len(EMOTIONS)))
                lines.append(f"{model_cell} & {dataset_cell} & {label} & {cells}\\tabularnewline")
    return lines


def seed_spread(summary: pd.DataFrame) -> list[str]:
    """Per-seed spread of the audited model, for the sentence that reports it."""
    lines = []
    for dataset in DATASETS_PL:
        for label, scope in SCOPES:
            runs = [shares(summary, f"{dataset} v3 s{seed}", scope, "emotion", EMOTIONS)
                    for seed in (7, 17, 27)]
            if any(r is None for r in runs):
                continue
            array = np.array(runs)
            worst = int(np.argmax(array.std(axis=0, ddof=1)))
            lines.append(
                f"{dataset:<14} {label:<9} "
                f"widest spread: {EMOTIONS[worst]} "
                f"{array.mean(axis=0)[worst]:.1f} +/- {array.std(axis=0, ddof=1)[worst]:.1f}"
            )
    return lines


def main() -> int:
    summary = pd.read_excel(SUMMARY_PATH)
    OUTPUT_DIR.mkdir(exist_ok=True)

    for name, rows in (("polarity_rows", polarity_rows(summary)),
                       ("emotion_rows", emotion_rows(summary))):
        path = OUTPUT_DIR / f"{name}.tex"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"{path}  ({len(rows)} rows)")

    print("\nSeed spread of the audited model:")
    for line in seed_spread(summary):
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
