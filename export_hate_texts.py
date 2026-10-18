"""Write the Polish hate-speech corpus for scoring on the machine with the GPU.

The transformer added to this chapter was fine-tuned in the `Polish_TER`
project and its weights live on the workstation that trained them. Rather than
move ~1.4 GB of weights per seed, this script moves the texts: one parquet, a
few megabytes, which that project's ``analysis/32_hate_text_inference.py``
reads and returns as predictions.

Run from this directory, because the dataset paths are relative to it:

    python export_hate_texts.py

Then copy ``exchange/hate_texts.parquet`` to the workstation as
``Polish_TER/datasets/processed/hate_texts.parquet`` and run that project's
``analysis/32_hate_text_inference.py``.
"""

import argparse
from pathlib import Path

from hate_corpus import build_corpus

OUTPUT_PATH = Path("exchange/hate_texts.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus, report = build_corpus()

    print()
    for line in report:
        print(
            f"{line['source']:<20} {line['rows_usable']:>6} usable "
            f"({line['hate']} hate, {line['neither']} non-hate)"
            + (f", {line['dropped_empty']} dropped empty" if line["dropped_empty"] else "")
            + (f", {line['repeated_texts']} repeated" if line["repeated_texts"] else "")
        )
    print(f"{'total':<20} {len(corpus):>6}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(args.output, index=False)
    print(f"\nWrote {args.output}")
    print(
        "\nNext: copy it to Polish_TER/datasets/processed/hate_texts.parquet "
        "on the workstation and run\n"
        "      .venv\\Scripts\\python.exe analysis\\32_hate_text_inference.py "
        "--run tf-polish-roberta-large-v2-s7__0c59b0319a"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
