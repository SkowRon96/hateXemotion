from hateXemotion import CONSTANTS
from dataset_variants import (
    BanPLhateXEmotion,
    HateCheckPolishhateXEmotion,
    HateSpeechPlhateXEmotion,
    PolEval2019hateXEmotion,
    HateSpeechOffensivehateXEmotion,
)
from models.en.emotions import emotion_en_analyzer
from models.pl.emotions_original_fastapi import emotion_chatgpt_pl_analyzer
from models.pl.emotions import emotion_pl_analyzer, emotion_pl_analyzer_mapper
from models.en.sentiment import sentiment_en_analyzer
from models.pl.sentiment import sentiment_pl_analyzer

# with BanPLhateXEmotion("ban_pl", "datasets/pl/ban_pl/BAN-PL 2.csv", "pl", sentiment_pl_analyzer, emotion_chatgpt_pl_analyzer) as analyzer:
#     analyzer.run(True)

# with HateCheckPolishhateXEmotion("hatecheck_polish", "datasets/pl/hatecheck_polish/hatecheck-polish.csv", "pl", sentiment_pl_analyzer, emotion_chatgpt_pl_analyzer) as analyzer:
#     analyzer.run(True)

# with HateSpeechPlhateXEmotion("hate_speech_pl", "datasets/pl/hate_speech_pl/train-00000-of-00001.csv", "pl", sentiment_pl_analyzer, emotion_chatgpt_pl_analyzer) as analyzer:
#     analyzer.run(True)

# with PolEval2019hateXEmotion("poleval2019", "", "pl", sentiment_pl_analyzer, emotion_chatgpt_pl_analyzer) as analyzer:
#     analyzer.run(True)

# with HateSpeechOffensivehateXEmotion("hate_speech_offensive", "datasets/en/hate_speech_offensive/train-00000-of-00001.csv", "en", sentiment_en_analyzer, emotion_en_analyzer) as analyzer:
#     analyzer.run(True)




# v2

with BanPLhateXEmotion("ban_pl_v2", "datasets/pl/ban_pl/BAN-PL 2.csv", "pl", sentiment_pl_analyzer, emotion_pl_analyzer, emotion_pl_analyzer_mapper) as analyzer:
    analyzer.run(True)

with HateCheckPolishhateXEmotion("hatecheck_polish_v2", "datasets/pl/hatecheck_polish/hatecheck-polish.csv", "pl", sentiment_pl_analyzer, emotion_pl_analyzer, emotion_pl_analyzer_mapper) as analyzer:
    analyzer.run(True)

with HateSpeechPlhateXEmotion("hate_speech_pl_v2", "datasets/pl/hate_speech_pl/train-00000-of-00001.csv", "pl", sentiment_pl_analyzer, emotion_pl_analyzer, emotion_pl_analyzer_mapper) as analyzer:
    analyzer.run(True)

with PolEval2019hateXEmotion("poleval2019_v2", "", "pl", sentiment_pl_analyzer, emotion_pl_analyzer, emotion_pl_analyzer_mapper) as analyzer:
    analyzer.run(True)
