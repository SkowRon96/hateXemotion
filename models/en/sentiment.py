# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
from transformers import pipeline

model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

sentiment_en_analyzer = pipeline("text-classification", model=model_path)

print("Sentiment analysis model for english texts loaded successfully.")

# Example usage:
# sentiment_en_analyzer("I was extremely disappointed with the service; nothing went as expected.")