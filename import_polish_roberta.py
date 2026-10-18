"""Merge the transformer's predictions into this project's results.

Reads the ``predictions_hate.parquet`` files produced by
``Polish_TER/analysis/32_hate_text_inference.py``, writes one
``results/<dataset>_v3_s<seed>_sentiment_results.xlsx`` per dataset and seed in
the same shape as the two passes already there, and refreshes the matching row
in ``results/all_data_results.xlsx`` through the existing summary code — so the
chapter's tables keep coming from one place.

**One row per seed, never an average.** Out of domain this model's dominant
class is seed-dependent — on TwitterEmo two seeds answer disgust and the third
answers neutral — so collapsing the seeds here would hide exactly the spread
the chapter has to report. Averaging, if it happens, happens in the table,
where it can be shown with its dispersion.

**Emotion only.** This pass adds an emotion model, not a sentiment model, so
the sentiment columns stay empty and the summary row reports no sentiment. The
sentiment table of the chapter is built from the earlier passes and is
unaffected; copying their sentiment in here would put the same predictions in
two rows and invite double counting.

Run from this directory, after copying ``results/08_hate_texts/`` back into the
Polish_TER project:

    python import_polish_roberta.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from hateXemotion import CONSTANTS
from hate_corpus import (
    DISPLAY_NAMES,
    EKMAN_NAMES,
    SOURCES,
    VARIANT_SUFFIX,
    build_corpus,
)

#: Where the stage lands when the results come back from the workstation. The
#: two projects sit side by side, and the predictions stay in the stage that
#: produced them rather than being copied in here, so that the manifest and the
#: run configuration remain next to the numbers they explain.
RUNS_DIR = Path("../../Polish_TER/results/08_hate_texts/runs")

PROBABILITY_COLUMNS = [f"p_{index}" for index in range(len(EKMAN_NAMES))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR,
                        help="directory holding one subdirectory per run")
    parser.add_argument("--run", action="append",
                        help="limit to one run_id; repeatable (default: all)")
    parser.add_argument("--source", action="append", choices=sorted(SOURCES),
                        help="limit to one dataset; repeatable (default: all)")
    return parser.parse_args()


def discover_runs(runs_dir: Path, wanted: list[str] | None) -> list[Path]:
    """Run directories holding predictions, newest naming scheme first."""
    if not runs_dir.is_dir():
        raise FileNotFoundError(
            f"{runs_dir} not found — copy results/08_hate_texts/ back from the "
            f"workstation into the Polish_TER project"
        )
    found = [
        directory for directory in sorted(runs_dir.iterdir())
        if (directory / "predictions_hate.parquet").exists()
        and (wanted is None or directory.name in wanted)
    ]
    if not found:
        raise FileNotFoundError(
            f"no predictions_hate.parquet under {runs_dir}"
            + (f" for {', '.join(wanted)}" if wanted else "")
        )
    return found


def read_run(directory: Path) -> tuple[int, pd.DataFrame]:
    """The seed and the predictions of one run.

    The seed is read from the run's own configuration rather than parsed out of
    its name: the name is a convention, the configuration is the record of what
    was actually trained.
    """
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    seed = int(config["train"]["seed"])

    predictions = pd.read_parquet(directory / "predictions_hate.parquet")
    expected = {"example_id", "source", "hate_category", "y_pred", *PROBABILITY_COLUMNS}
    missing = sorted(expected - set(predictions.columns))
    if missing:
        raise ValueError(
            f"{directory.name}: predictions are missing column(s): {', '.join(missing)}"
        )

    # The predicted class is stored as an index; every share reported later
    # depends on this tuple being in the order the model was trained with.
    predictions["emotion"] = [EKMAN_NAMES[int(value)] for value in predictions["y_pred"]]
    predictions["emotion_score"] = predictions[PROBABILITY_COLUMNS].max(axis=1)
    return seed, predictions.set_index("example_id")


def main() -> int:
    args = parse_args()
    runs = discover_runs(args.runs_dir, args.run)
    sources = args.source or list(SOURCES)
    corpus, _ = build_corpus(sources)
    print(f"\nRuns: {len(runs)}   datasets: {len(sources)}\n")

    for directory in runs:
        seed, predictions = read_run(directory)
        for source in sources:
            rows = corpus[corpus["source"] == source]
            known = rows["example_id"].isin(predictions.index)
            if not known.all():
                raise AssertionError(
                    f"{directory.name}/{source}: {int((~known).sum())} of "
                    f"{len(rows)} texts have no prediction — the corpus and the "
                    f"predictions come from different exports"
                )
            matched = predictions.loc[rows["example_id"]]

            # The column order is the one every other results file in this
            # project uses; the summary code reads these names, not positions.
            frame = pd.DataFrame(
                {
                    CONSTANTS.PANDAS_COLUMNS.TEXT: rows["text"].to_numpy(),
                    CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY: rows["hate_category"].to_numpy(),
                    CONSTANTS.PANDAS_COLUMNS.SENTIMENT: None,
                    CONSTANTS.PANDAS_COLUMNS.SENTIMENT_SCORE: None,
                    CONSTANTS.PANDAS_COLUMNS.EMOTION: matched["emotion"].to_numpy(),
                    CONSTANTS.PANDAS_COLUMNS.EMOTION_SCORE: matched["emotion_score"].to_numpy(),
                }
            )

            variant_name = f"{VARIANT_SUFFIX.lstrip('_')}_s{seed}"
            target = Path(f"results/{source}_{variant_name}_sentiment_results.xlsx")
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.to_excel(target, index=False)

            hate_rows = frame[
                frame[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY]
                == CONSTANTS.HATESPEECH_CATEGORIES.HATE
            ]
            shares = frame[CONSTANTS.PANDAS_COLUMNS.EMOTION].value_counts(normalize=True)
            hate_shares = hate_rows[CONSTANTS.PANDAS_COLUMNS.EMOTION].value_counts(normalize=True)
            print(
                f"{source + ' s' + str(seed):<28} {len(frame):>6} rows   "
                f"all: {shares.index[0]} {shares.iloc[0]:.4f}   "
                + (f"hate: {hate_shares.index[0]} {hate_shares.iloc[0]:.4f}"
                   if len(hate_rows) else "hate: ---")
            )

            # Reuse the pipeline's own summary pass rather than recomputing the
            # counts here: it is what produced every other row of the summary
            # file. The analyzer writes its own name into the summary, so it is
            # given the presentation name and pointed at the result file
            # explicitly — otherwise the row would have to be renamed by hand
            # afterwards, as the rows above it were.
            variant, path = SOURCES[source]
            display = f"{DISPLAY_NAMES[source]} v3 s{seed}"
            with variant(display, path, "pl", None, None) as analyzer:
                analyzer.result_excel_path = str(target)
                analyzer.run(True)

    print(
        f"\n{len(runs) * len(sources)} row(s) refreshed in "
        f"results/all_data_results.xlsx "
        f"(emotion only — sentiment columns are empty by design)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
