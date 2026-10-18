
from transformers import pipeline
from bs4 import BeautifulSoup
from pydantic import BaseModel
import pandas as pd

class SummaryColumns(BaseModel):
    DATASET: str = 'dataset'
    LANGUAGE: str = 'language'
    ALL: str = 'all'
    ALL_EMOTION: str = 'all_emotion'
    ALL_EMOTION_NEUTRAL: str = 'all_emotion_neutral'
    ALL_EMOTION_HAPPINESS: str = 'all_emotion_happiness'
    ALL_EMOTION_SADNESS: str = 'all_emotion_sadness'
    ALL_EMOTION_FEAR: str = 'all_emotion_fear'
    ALL_EMOTION_ANGER: str = 'all_emotion_anger'
    ALL_EMOTION_DISGUST: str = 'all_emotion_disgust'
    ALL_EMOTION_SURPRISE: str = 'all_emotion_surprise'
    ALL_SENTIMENT: str = 'all_sentiment'
    ALL_SENTIMENT_POSITIVE: str = 'all_sentiment_positive'
    ALL_SENTIMENT_NEGATIVE: str = 'all_sentiment_negative'
    ALL_SENTIMENT_NEUTRAL: str = 'all_sentiment_neutral'
    HATE: str = 'hate'
    HATE_EMOTION: str = 'hate_emotion'
    HATE_EMOTION_NEUTRAL: str = 'hate_emotion_neutral'
    HATE_EMOTION_HAPPINESS: str = 'hate_emotion_happiness'
    HATE_EMOTION_SADNESS: str = 'hate_emotion_sadness'
    HATE_EMOTION_FEAR: str = 'hate_emotion_fear'
    HATE_EMOTION_ANGER: str = 'hate_emotion_anger'
    HATE_EMOTION_DISGUST: str = 'hate_emotion_disgust'
    HATE_EMOTION_SURPRISE: str = 'hate_emotion_surprise'
    HATE_SENTIMENT: str = 'hate_sentiment'
    HATE_SENTIMENT_POSITIVE: str = 'hate_sentiment_positive'
    HATE_SENTIMENT_NEGATIVE: str = 'hate_sentiment_negative'
    HATE_SENTIMENT_NEUTRAL: str = 'hate_sentiment_neutral'
    NO_HATE: str = 'no_hate'
    NO_HATE_EMOTION: str = 'no_hate_emotion'
    NO_HATE_EMOTION_NEUTRAL: str = 'no_hate_emotion_neutral'
    NO_HATE_EMOTION_HAPPINESS: str = 'no_hate_emotion_happiness'
    NO_HATE_EMOTION_SADNESS: str = 'no_hate_emotion_sadness'
    NO_HATE_EMOTION_FEAR: str = 'no_hate_emotion_fear'
    NO_HATE_EMOTION_ANGER: str = 'no_hate_emotion_anger'
    NO_HATE_EMOTION_DISGUST: str = 'no_hate_emotion_disgust'
    NO_HATE_EMOTION_SURPRISE: str = 'no_hate_emotion_surprise'
    NO_HATE_SENTIMENT: str = 'no_hate_sentiment'
    NO_HATE_SENTIMENT_POSITIVE: str = 'no_hate_sentiment_positive'
    NO_HATE_SENTIMENT_NEGATIVE: str = 'no_hate_sentiment_negative'
    NO_HATE_SENTIMENT_NEUTRAL: str = 'no_hate_sentiment_neutral'

class PandasColumns(BaseModel):
    TEXT: str = 'text'
    HATE_CATEGORY: str = 'hate_category'
    SENTIMENT: str = 'sentiment'
    SENTIMENT_SCORE: str = 'sentiment_score'
    EMOTION: str = 'emotion'
    EMOTION_SCORE: str = 'emotion_score'

class EmotionCategories(BaseModel):
    NEUTRAL: str = 'neutral'
    HAPPINESS: str = 'happiness'
    SADNESS: str = 'sadness'
    FEAR: str = 'fear'
    DISGUST: str = 'disgust'
    SURPRISE: str = 'surprise'
    ANGER: str = 'anger'

    
    
    

class PolarityCategories(BaseModel):
    NEUTRAL: str = 'neutral'
    NEGATIVE: str = 'negative'
    POSITIVE: str = 'positive'
    
class HateSpeechCategories(BaseModel):
    HATE: str = 'hate'
    NEITHER: str = 'neither'

class CONSTANTS(BaseModel):
    PANDAS_COLUMNS: PandasColumns = PandasColumns()
    SUMMARY_COLUMNS: SummaryColumns = SummaryColumns()
    EMOTION_CATEGORIES: EmotionCategories = EmotionCategories()
    POLARITY_CATEGORIES: PolarityCategories = PolarityCategories()
    HATESPEECH_CATEGORIES: HateSpeechCategories = HateSpeechCategories()

CONSTANTS = CONSTANTS()

class HateXEmotion():
    def __init__(self, dataset_name: str, datatset_path: str, language:str, sentiment_model: pipeline, emotion_model: pipeline, emotion_mapping: dict = dict(), update_only: bool = True):
        self.datatset_path = datatset_path
        self.dataset_name = dataset_name
        self.language = language
        self.sentiment_model = sentiment_model
        self.emotion_model = emotion_model
        self.emotion_mapping = emotion_mapping
        self.result_excel_path = f"results/{self.dataset_name}_sentiment_results.xlsx"
        self.summary_excel_path = f"results/all_data_results.xlsx"
        self.update_only = update_only

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        if exc_type is not None:
            print(f"An error occurred: {exc_value}")
        else:
            self.save_results()
            print(f"Processing {self.datatset_path} for {self.language} completed successfully.")
    
    def load_results(self) -> bool:
        try:
            self.dataset = pd.read_excel(self.result_excel_path)
            print(f"Loaded existing results from {self.result_excel_path}")
            return True
        except FileNotFoundError:
            print(f"No existing results found at {self.result_excel_path}, starting fresh.")
            return False
        
    def save_results(self):
        # Implement saving logic specific to BanPLhateXemotion
        self.dataset.to_excel(self.result_excel_path, index=False)
        if hasattr(self, 'summary'):
            self.summary.to_excel(self.summary_excel_path, index=False)
        print(f"Results saved to {self.result_excel_path} and summary saved to {self.summary_excel_path}")

    def run(self, only_analyze: bool = False):
        print(f"Running analysis for {self.language} using {self.sentiment_model.__class__.__name__} and {self.emotion_model.__class__.__name__}...")

        if self.update_only:
            if not self.load_results():
                self.preprocess_dataset()
        else:
            self.preprocess_dataset()

        if not only_analyze:
            for idx, row in self.dataset.iterrows():
                text = row[CONSTANTS.PANDAS_COLUMNS.TEXT]
                try:
                    if pd.isna(self.dataset.at[idx, CONSTANTS.PANDAS_COLUMNS.SENTIMENT]) and pd.isna(self.dataset.at[idx, CONSTANTS.PANDAS_COLUMNS.SENTIMENT_SCORE]):
                        sentiment_result = self.sentiment_model(text)[0]
                        self.dataset.at[idx, CONSTANTS.PANDAS_COLUMNS.SENTIMENT] = sentiment_result.get('label','').lower()
                        self.dataset.at[idx, CONSTANTS.PANDAS_COLUMNS.SENTIMENT_SCORE] = sentiment_result.get('score')
                except Exception as e: 
                    print(f"Error processing sentiment for row {idx}: {e}")
                try:
                    if pd.isna(self.dataset.at[idx, CONSTANTS.PANDAS_COLUMNS.EMOTION]) and pd.isna(self.dataset.at[idx, CONSTANTS.PANDAS_COLUMNS.EMOTION_SCORE]):
                        emotion_result = self.emotion_model(text)[0]
                        self.dataset.at[idx, CONSTANTS.PANDAS_COLUMNS.EMOTION] = emotion_result.get('label','').lower().replace('joy', 'happiness').replace('neutral', 'neutral') if not self.emotion_mapping else self.emotion_mapping.get(emotion_result.get('label',''))
                        self.dataset.at[idx, CONSTANTS.PANDAS_COLUMNS.EMOTION_SCORE] = emotion_result.get('score')
                except Exception as e:
                    print(f"Error processing emotion for row {idx}: {e}")
                
                print(f"Processed row {idx + 1}/{len(self.dataset)}. Sentiment: {self.dataset.at[idx, CONSTANTS.PANDAS_COLUMNS.SENTIMENT]}, Emotion: {self.dataset.at[idx, CONSTANTS.PANDAS_COLUMNS.EMOTION]}")    
                if idx % 500 == 0:  self.save_results()
        else:
            self.summary = pd.read_excel(self.summary_excel_path)
            row_index = -1
            for idx, row in self.summary.iterrows():
                if self.dataset_name in self.summary.at[idx, CONSTANTS.SUMMARY_COLUMNS.DATASET]: 
                    row_index = idx
                    break
            if row_index < 0: 
                row_index = len(self.summary)
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.DATASET] = self.dataset_name
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.LANGUAGE] = self.language
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.ALL] = len(self.dataset)
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.ALL_EMOTION] = len(self.dataset[~self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION].isna()])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.ALL_EMOTION_NEUTRAL] = len(self.dataset[self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.NEUTRAL])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.ALL_EMOTION_HAPPINESS] = len(self.dataset[self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.HAPPINESS])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.ALL_EMOTION_SADNESS] = len(self.dataset[self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.SADNESS])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.ALL_EMOTION_FEAR] = len(self.dataset[self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.FEAR])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.ALL_EMOTION_ANGER] = len(self.dataset[self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.ANGER])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.ALL_EMOTION_DISGUST] = len(self.dataset[self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.DISGUST])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.ALL_EMOTION_SURPRISE] = len(self.dataset[self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.SURPRISE])  
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.ALL_SENTIMENT] = len(self.dataset[~self.dataset[CONSTANTS.PANDAS_COLUMNS.SENTIMENT].isna()])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.ALL_SENTIMENT_POSITIVE] = len(self.dataset[self.dataset[CONSTANTS.PANDAS_COLUMNS.SENTIMENT] == CONSTANTS.POLARITY_CATEGORIES.POSITIVE])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.ALL_SENTIMENT_NEGATIVE] = len(self.dataset[self.dataset[CONSTANTS.PANDAS_COLUMNS.SENTIMENT] == CONSTANTS.POLARITY_CATEGORIES.NEGATIVE])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.ALL_SENTIMENT_NEUTRAL] = len(self.dataset[self.dataset[CONSTANTS.PANDAS_COLUMNS.SENTIMENT] == CONSTANTS.POLARITY_CATEGORIES.NEUTRAL])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.HATE] = len(self.dataset[self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.HATE])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.HATE_EMOTION] =  len(self.dataset[~self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION].isna() & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.HATE)])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.HATE_EMOTION_NEUTRAL] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.NEUTRAL) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.HATE)])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.HATE_EMOTION_HAPPINESS] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.HAPPINESS) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.HATE)])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.HATE_EMOTION_SADNESS] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.SADNESS) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.HATE)])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.HATE_EMOTION_FEAR] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.FEAR) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.HATE)])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.HATE_EMOTION_ANGER] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.ANGER) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.HATE)])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.HATE_EMOTION_DISGUST] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.DISGUST) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.HATE)])  
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.HATE_EMOTION_SURPRISE] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.SURPRISE) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.HATE)])    
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.HATE_SENTIMENT] = len(self.dataset[~self.dataset[CONSTANTS.PANDAS_COLUMNS.SENTIMENT].isna() & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.HATE)])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.HATE_SENTIMENT_POSITIVE] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.SENTIMENT] == CONSTANTS.POLARITY_CATEGORIES.POSITIVE) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.HATE)])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.HATE_SENTIMENT_NEGATIVE] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.SENTIMENT] == CONSTANTS.POLARITY_CATEGORIES.NEGATIVE) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.HATE)])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.HATE_SENTIMENT_NEUTRAL] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.SENTIMENT] == CONSTANTS.POLARITY_CATEGORIES.NEUTRAL) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.HATE)])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.NO_HATE] = len(self.dataset[self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.NEITHER])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.NO_HATE_EMOTION] = len(self.dataset[~self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION].isna() & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.NEITHER)])  
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.NO_HATE_EMOTION_NEUTRAL] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.NEUTRAL) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.NEITHER)])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.NO_HATE_EMOTION_HAPPINESS] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.HAPPINESS) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.NEITHER)])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.NO_HATE_EMOTION_SADNESS] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.SADNESS) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.NEITHER)])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.NO_HATE_EMOTION_FEAR] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.FEAR) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.NEITHER)])  
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.NO_HATE_EMOTION_ANGER] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.ANGER) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.NEITHER)])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.NO_HATE_EMOTION_DISGUST] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.DISGUST) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.NEITHER)])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.NO_HATE_EMOTION_SURPRISE] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.EMOTION] == CONSTANTS.EMOTION_CATEGORIES.SURPRISE) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.NEITHER)])  
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.NO_HATE_SENTIMENT] = len(self.dataset[~self.dataset[CONSTANTS.PANDAS_COLUMNS.SENTIMENT].isna() & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.NEITHER)])  
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.NO_HATE_SENTIMENT_POSITIVE] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.SENTIMENT] == CONSTANTS.POLARITY_CATEGORIES.POSITIVE) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.NEITHER)])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.NO_HATE_SENTIMENT_NEGATIVE] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.SENTIMENT] == CONSTANTS.POLARITY_CATEGORIES.NEGATIVE) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.NEITHER)])
            self.summary.loc[row_index, CONSTANTS.SUMMARY_COLUMNS.NO_HATE_SENTIMENT_NEUTRAL] = len(self.dataset[(self.dataset[CONSTANTS.PANDAS_COLUMNS.SENTIMENT] == CONSTANTS.POLARITY_CATEGORIES.NEUTRAL) & (self.dataset[CONSTANTS.PANDAS_COLUMNS.HATE_CATEGORY] == CONSTANTS.HATESPEECH_CATEGORIES.NEITHER)])

    def clean_text(self, text: str) -> str:
        # Implement text cleaning logic if necessary
        text = text.replace('\t', '').replace('\n', '').replace('\r', '').replace('  ', ' ').strip()
        try: text = BeautifulSoup(text, "lxml").text
        except: pass
        return text
    
    def preprocess_dataset(self):
        raise NotImplementedError("This method should be implemented in a subclass.")