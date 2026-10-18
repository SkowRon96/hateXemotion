"""The Polish hate-speech corpus as one frame, shared by export and import.

``export_hate_texts.py`` sends this frame to the machine that holds the
fine-tuned transformer; ``import_polish_roberta.py`` rebuilds it to merge the
predictions back. Both must agree on every row and every identifier, so the
construction lives here once rather than twice.

Only the Polish datasets are included. The model being added
(``sdadas/polish-roberta-large-v2``, fine-tuned in the `Polish_TER` project) is
Polish-only, and running it on the English dataset would produce numbers that
look comparable to the other rows of the results table while measuring nothing.

Identity, not row order, is what carries a prediction back to its text. The
export and the import are separated by a file transfer between two machines,
and a reordering anywhere in between would otherwise reattach every emotion to
the wrong sentence — silently, and in a way no later check could detect.
"""

import hashlib
import re
import unicodedata

import pandas as pd

from hateXemotion import CONSTANTS
from dataset_variants import (
    BanPLhateXEmotion,
    HateCheckPolishhateXEmotion,
    HateSpeechPlhateXEmotion,
    PolEval2019hateXEmotion,
)

#: Emotion names by class index, in the order of ``ter.labels.Emotion`` in the
#: Polish_TER project. The model emits indices; this tuple is the only place
#: that turns them into names, and its order is part of the contract with the
#: trained weights — reordering it silently relabels every prediction.
EKMAN_NAMES = (
    CONSTANTS.EMOTION_CATEGORIES.NEUTRAL,
    CONSTANTS.EMOTION_CATEGORIES.HAPPINESS,
    CONSTANTS.EMOTION_CATEGORIES.SADNESS,
    CONSTANTS.EMOTION_CATEGORIES.FEAR,
    CONSTANTS.EMOTION_CATEGORIES.ANGER,
    CONSTANTS.EMOTION_CATEGORIES.DISGUST,
    CONSTANTS.EMOTION_CATEGORIES.SURPRISE,
)

#: One entry per Polish dataset: the class that knows how to read it, and the
#: path it is read from. ``PolEval2019hateXEmotion`` takes its two files from
#: inside its own ``preprocess_dataset``, hence the empty path.
SOURCES = {
    "ban_pl": (BanPLhateXEmotion, "datasets/pl/ban_pl/BAN-PL 2.csv"),
    "hatecheck_polish": (HateCheckPolishhateXEmotion, "datasets/pl/hatecheck_polish/hatecheck-polish.csv"),
    "hate_speech_pl": (HateSpeechPlhateXEmotion, "datasets/pl/hate_speech_pl/train-00000-of-00001.csv"),
    "poleval2019": (PolEval2019hateXEmotion, ""),
}

#: Suffix distinguishing this pass from the two already in ``results/``:
#: no suffix is the Chatbot model, ``_v2`` is the fine-tuned RoBERTa.
VARIANT_SUFFIX = "_v3"

#: How each dataset is named in ``results/all_data_results.xlsx``, which uses
#: presentation names rather than the identifiers the result files are named
#: with. The two are genuinely different things — one addresses a file, the
#: other appears in the chapter — and keeping the mapping here is what lets a
#: new pass land in the summary already consistent with the rows above it.
DISPLAY_NAMES = {
    "ban_pl": "BAN-PL",
    "hatecheck_polish": "MHC",
    "hate_speech_pl": "HateSpeechPl",
    "poleval2019": "PolEval 2019",
}

_WHITESPACE = re.compile(r"\s+")


def normalise_text(raw: str) -> str:
    """Normalise for *identity* only — NFC, trimmed, whitespace collapsed.

    Deliberately does not lowercase or strip punctuation: those would make two
    genuinely different texts share an identifier, and the texts themselves are
    passed to the model untouched.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", str(raw))).strip()


def make_example_id(source: str, text: str) -> str:
    """Stable identifier for one text, derived from content rather than position."""
    digest = hashlib.sha1(f"{source}\x00{normalise_text(text)}".encode("utf-8"))
    return digest.hexdigest()[:16]


def load_source(source: str) -> pd.DataFrame:
    """One dataset, preprocessed exactly as the analysis pipeline preprocesses it.

    The analyzer is constructed without models because ``preprocess_dataset``
    does not use them; this is reading and cleaning, not inference.
    """
    if source not in SOURCES:
        raise KeyError(f"unknown source {source!r}; expected one of {', '.join(SOURCES)}")
    variant, path = SOURCES[source]
    analyzer = variant(source, path, "pl", None, None, update_only=False)
    analyzer.preprocess_dataset()
    return analyzer.dataset


def build_corpus(sources: list[str] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Every Polish hate-speech text, with a unique identifier per row.

    Returns the corpus and one report line per dataset. Rows whose text is
    empty after cleaning are dropped rather than scored: a model given an empty
    string still returns a class, and that class would enter the chapter's
    percentages as if it described something.
    """
    frames, report = [], []
    for source in sources or list(SOURCES):
        data = load_source(source)
        text_column = CONSTANTS.PANDAS_COLUMNS.TEXT
        hate_column = CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY

        frame = pd.DataFrame(
            {
                "source": source,
                "text": data[text_column].astype(str).map(normalise_text),
                "hate_category": data[hate_column].astype(str),
            }
        )
        rows_read = len(frame)
        frame = frame[frame["text"] != ""].reset_index(drop=True)
        dropped = rows_read - len(frame)

        frame.insert(
            0, "example_id",
            [make_example_id(source, text) for text in frame["text"]],
        )
        # A corpus can repeat a text verbatim — HateCheck is template-generated
        # and the forum datasets contain reposts. Content-hashed ids would then
        # collide, and the merge back joins on the id, so repeats get an
        # occurrence suffix while still grouping identical texts under a common
        # prefix.
        duplicated = frame["example_id"].duplicated(keep=False)
        if duplicated.any():
            occurrence = frame.groupby("example_id").cumcount()
            frame.loc[duplicated, "example_id"] = (
                frame.loc[duplicated, "example_id"]
                + "#"
                + occurrence[duplicated].astype(str)
            )

        frames.append(frame)
        report.append(
            {
                "source": source,
                "rows_read": rows_read,
                "rows_usable": len(frame),
                "dropped_empty": dropped,
                "repeated_texts": int(duplicated.sum()) if duplicated.any() else 0,
                "hate": int((frame["hate_category"] == CONSTANTS.HATESPEECH_CATEGORIES.HATE).sum()),
                "neither": int((frame["hate_category"] == CONSTANTS.HATESPEECH_CATEGORIES.NEITHER).sum()),
            }
        )

    corpus = pd.concat(frames, ignore_index=True)
    repeated = corpus["example_id"].duplicated()
    if repeated.any():
        raise AssertionError(
            f"{int(repeated.sum())} identifier(s) repeat across datasets — "
            f"the source name should have made them unique"
        )
    return corpus, report
