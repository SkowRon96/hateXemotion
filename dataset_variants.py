"""One ``HateXEmotion`` subclass per dataset, holding its preprocessing.

Extracted from ``main.py`` unchanged. ``main.py`` runs its analyses at import
time, so anything that needs only the preprocessing — the corpus exporter, the
prediction importer — could not import from it without also loading every
model and rerunning every dataset. The classes live here and ``main.py``
imports them, so there is still exactly one definition of how a dataset is
read, cleaned and mapped onto hate/neither.
"""

import pandas as pd

from hateXemotion import HateXEmotion, CONSTANTS


class BanPLhateXEmotion(HateXEmotion):

    def preprocess_dataset(self):
        print(f"Preprocessing dataset at {self.datatset_path} for {self.language}...")
        data = pd.read_csv(self.datatset_path)
        data = data[["Text", "Class"]]
        data["sentiment"] = None
        data["sentiment_score"] = None
        data["emotion"] = None
        data["emotion_score"] = None
        data.columns = [value for key, value in CONSTANTS.PANDAS_COLUMNS.model_dump().items()]
        data[CONSTANTS.PANDAS_COLUMNS.TEXT] = data[CONSTANTS.PANDAS_COLUMNS.TEXT].apply(lambda x: self.clean_text(x))
        data[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] = data[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY].apply(lambda x: CONSTANTS.HATESPEECH_CATEGORIES.HATE if x == 1 else CONSTANTS.HATESPEECH_CATEGORIES.NEITHER)
        self.dataset = data
        print(f"Loaded {len(self.dataset)} records from {self.datatset_path}")


class HateCheckPolishhateXEmotion(HateXEmotion):

    def preprocess_dataset(self):
        print(f"Preprocessing dataset at {self.datatset_path} for {self.language}...")
        data = pd.read_csv(self.datatset_path)
        data = data[["test_case", "label_gold"]]
        data["sentiment"] = None
        data["sentiment_score"] = None
        data["emotion"] = None
        data["emotion_score"] = None
        data.columns = [value for key, value in CONSTANTS.PANDAS_COLUMNS.model_dump().items()]
        data[CONSTANTS.PANDAS_COLUMNS.TEXT] = data[CONSTANTS.PANDAS_COLUMNS.TEXT].apply(lambda x: self.clean_text(x))
        data[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] = data[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY].apply(lambda x: CONSTANTS.HATESPEECH_CATEGORIES.HATE if x == 'hateful' else CONSTANTS.HATESPEECH_CATEGORIES.NEITHER)
        self.dataset = data
        print(f"Loaded {len(self.dataset)} records from {self.datatset_path}")


class HateSpeechPlhateXEmotion(HateXEmotion):

    def preprocess_dataset(self):
        print(f"Preprocessing dataset at {self.datatset_path} for {self.language}...")
        data = pd.read_csv(self.datatset_path)
        data = data[["text"]]
        data["hate_category"] = None
        data["sentiment"] = None
        data["sentiment_score"] = None
        data["emotion"] = None
        data["emotion_score"] = None
        data.columns = [value for key, value in CONSTANTS.PANDAS_COLUMNS.model_dump().items()]
        data[CONSTANTS.PANDAS_COLUMNS.TEXT] = data[CONSTANTS.PANDAS_COLUMNS.TEXT].apply(lambda x: self.clean_text(x))
        data[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] = data[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY].apply(lambda x: CONSTANTS.HATESPEECH_CATEGORIES.HATE)
        self.dataset = data
        print(f"Loaded {len(self.dataset)} records from {self.datatset_path}")


class PolEval2019hateXEmotion(HateXEmotion):

    def preprocess_dataset(self):
        text_path = "datasets/pl/poleval2019/training_set_clean_only_text.txt"
        tags_path = "datasets/pl/poleval2019/training_set_clean_only_tags.txt"
        print(f"Preprocessing dataset at {text_path} and {tags_path} for {self.language}...")
        with open(text_path, "r", encoding="utf-8") as f_text, open(tags_path, "r", encoding="utf-8") as f_tags:
            texts = [line.replace('@anonymized_account ', '').strip() for line in f_text]
            labels = [line.strip() for line in f_tags]
            if len(texts) != len(labels):
                raise ValueError("Text and label files have different number of rows.")

        data = pd.DataFrame({"text": texts, "hate_category": labels})
        data["sentiment"] = None
        data["sentiment_score"] = None
        data["emotion"] = None
        data["emotion_score"] = None
        data.columns = [value for key, value in CONSTANTS.PANDAS_COLUMNS.model_dump().items()]
        data[CONSTANTS.PANDAS_COLUMNS.TEXT] = data[CONSTANTS.PANDAS_COLUMNS.TEXT].apply(lambda x: self.clean_text(x))
        data[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] = data[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY].apply(lambda x: CONSTANTS.HATESPEECH_CATEGORIES.HATE if x == '1' else CONSTANTS.HATESPEECH_CATEGORIES.NEITHER)
        self.dataset = data
        print(f"Loaded {len(self.dataset)} records from {self.datatset_path}")


class HateSpeechOffensivehateXEmotion(HateXEmotion):

    def preprocess_dataset(self):
        print(f"Preprocessing dataset at {self.datatset_path} for {self.language}...")
        data = pd.read_csv(self.datatset_path)
        data = data[["tweet", "class"]]
        data["sentiment"] = None
        data["sentiment_score"] = None
        data["emotion"] = None
        data["emotion_score"] = None
        data.columns = [value for key, value in CONSTANTS.PANDAS_COLUMNS.model_dump().items()]
        data[CONSTANTS.PANDAS_COLUMNS.TEXT] = data[CONSTANTS.PANDAS_COLUMNS.TEXT].apply(lambda x: self.clean_text(x))
        data[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] = data[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY].apply(lambda x: CONSTANTS.HATESPEECH_CATEGORIES.NEITHER if x == 2 else CONSTANTS.HATESPEECH_CATEGORIES.HATE)
        self.dataset = data
        print(f"Loaded {len(self.dataset)} records from {self.datatset_path}")
