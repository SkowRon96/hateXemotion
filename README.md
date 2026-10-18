# hateXemotion tool

This repository tests whether sentiment and emotion predictions can be used as a simple first filter for hate speech detection. Sentiment and emotion models are tested on five hate speech datasets, including four Polish and one English dataset. The analysis compares predicted emotions and polarity between hateful and non-hateful texts.

The experiments are implemented in **hateXemotion**, a tool developed to support the complete analysis process. It provides a common framework for preparing datasets, loading models, generating predictions, comparing results, and creating visualizations. The main stages are:

1. **Data preprocessing**: datasets are converted to a common format containing the statement text, a binary hate speech label, and model predictions. The texts are also cleaned of HTML tags and formatting characters.
2. **Model loading**: emotion and sentiment models can be loaded together when they support the same language. Their output labels are mapped to a common set of categories.
3. **Emotion and polarity prediction**: the models are applied to the datasets to predict the emotion and polarity of each statement. Each prediction is stored together with its confidence value.
4. **Summary**: predictions are collected in a table to allow comparison between models and datasets.
5. **Visualization**: radar charts are generated to show the relationship between detected emotions and hate speech.

Adding a new dataset requires only mapping its categories to the binary hate speech label. Adding a new model requires mapping its output labels to the common categories. The remaining stages are shared across all configurations, which keeps the results comparable between models and datasets.

## Modules

The files below are listed in the order of the stages above.

| Stage              | File                                                       | Purpose                                                                                                                                                                       |
| ------------------ | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data preprocessing | `dataset_variants.py`                                    | One class per dataset: reads it, cleans the texts and maps its labels onto the common `hate` / `neither`.                                                                 |
| Data preprocessing | `hate_corpus.py`                                         | Builds the combined Polish corpus with a stable identifier per text.                                                                                                          |
| Model loading      | `models/`                                                | Sentiment and emotion model wrappers (`en/`, `pl/`) with label mappings to the common categories.                                                                         |
| Prediction         | `hateXemotion.py`                                        | `HateXEmotion` base class: text cleaning, sentiment and emotion inference, result caching (`results/<dataset>_sentiment_results.xlsx`); shared constants (`CONSTANTS`). |
| Prediction         | `main.py`                                                | Runs the selected dataset × model combinations.                                                                                                                              |
| Summary            | `hateXemotion.py`                                        | Per-dataset summary of predictions for all, hateful and non-hateful texts (`results/all_data_results.xlsx`).                                                                |
| Summary            | `make_tables.py`                                         | LaTeX table rows → `tables/`.                                                                                                                                              |
| Visualization      | `make_figures_polarity.py`, `make_figures_emotions.py` | Radar and bar charts → `figures/`.                                                                                                                                         |
| Visualization      | `research_style.py`                                      | Shared figure style (fonts, palette, plotting helpers).                                                                                                                       |

## Models

The study uses publicly available models from Hugging Face and the author's own models.

### Public models

| Task      | Language | Model                                                                                                                      |
| --------- | -------- | -------------------------------------------------------------------------------------------------------------------------- |
| Sentiment | PL       | [eevvgg/bert-polish-sentiment-politics](https://huggingface.co/eevvgg/bert-polish-sentiment-politics)                       |
| Sentiment | EN       | [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) |
| Emotion   | PL       | [visegradmedia-emotion/Emotion_RoBERTa_polish6](https://huggingface.co/visegradmedia-emotion/Emotion_RoBERTa_polish6)       |
| Emotion   | EN       | [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)       |

### Own models

These models are not included in the repository. They are available upon contact with the author.

| Task    | Language | Model                                                                                                                                            |
| ------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Emotion | PL       | Chatbot: LSTM emotion recognition model (`models/pl/emotions_original_fastapi.py`)                                                             |
| Emotion | PL       | Polish RoBERTa: emotion recognition model fine-tuned from [sdadas/polish-roberta-large-v2](https://huggingface.co/sdadas/polish-roberta-large-v2) |

## Data

Datasets are not redistributed. Download them and place the files exactly at these paths, relative to the repository root:

| Dataset                | Source                                                                                            | Path                                                                          | Label → `hate` / `neither`                         |
| ---------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------- |
| BAN-PL                 | [ZILiAT-NASK/BAN-PL](https://github.com/ZILiAT-NASK/BAN-PL)                                        | `datasets/pl/ban_pl/BAN-PL 2.csv`                                           | `Class`: 1 → hate, 0 → neither                      |
| MHC (HateCheck Polish) | [Multilingual HateCheck](https://github.com/paul-rottger/multilingual-hatecheck)                   | `datasets/pl/hatecheck_polish/hatecheck-polish.csv`                         | `label_gold`: hateful → hate, non-hateful → neither |
| PolEval 2019 Task 6-1  | [PolEval 2019](http://2019.poleval.pl/index.php/tasks/task6)                                       | `datasets/pl/poleval2019/training_set_clean_only_text.txt`, `…_tags.txt` | 1 → hate, 0 → neither                                 |
| HateSpeechPl           | [hate_speech_pl](https://huggingface.co/datasets/hate_speech_pl)                                  | `datasets/pl/hate_speech_pl/train-00000-of-00001.csv`                       | all texts → hate                                       |
| Hate Speech Offensive  | [tdavidson/hate_speech_offensive](https://huggingface.co/datasets/tdavidson/hate_speech_offensive) | `datasets/en/hate_speech_offensive/train-00000-of-00001.csv`                | `class`: 0, 1 → hate, 2 → neither                   |

`results/`, `figures/`, `tables/` and `exchange/` are created by the scripts.

## Usage

Python 3.11. Install dependencies, or open the folder in the provided dev container (`.devcontainer/`):

```bash
pip install -r .docker/requirements.txt
```

Run everything from the repository root (paths are relative):

```bash
python main.py                    # inference + summary for the combinations enabled in main.py
python make_figures_polarity.py   # figures/
python make_figures_emotions.py
python make_tables.py             # tables/
```

In `main.py`, `analyzer.run()` scores the texts (resumable: rows already in `results/` are skipped). `analyzer.run(True)` only recomputes that dataset's row in `results/all_data_results.xlsx`.

## Citation

The code is released under the [MIT License](LICENSE). If you use it, please cite the paper below. The datasets and models remain subject to their own licenses and terms of use.

```bibtex
@inproceedings{skowronski2025sentiment,
  title={From sentiment to safety: conditional hate speech detection based on emotion analysis in social robotics},
  author={Skowro{\'n}ski, Kamil and Ga{\l}uszka, Adam and Probierz, Eryka},
  booktitle={Modelling and simulation 2025. The 2025 European Simulation and Modelling Conference. ESM'2025, October 22-24, 2025, Ghent, Belgium},
  editor={Bhonsale, S. S. and Polanska, M. E. and van Impe, J. F. M.},
  pages={57--62},
  year={2025},
  publisher={EUROSIS-ETI},
  address={Ostend}
}
```
