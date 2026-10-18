# https://huggingface.co/eevvgg/bert-polish-sentiment-politics
from transformers import pipeline

model_path = "eevvgg/bert-polish-sentiment-politics"

sentiment_pl_analyzer = pipeline("text-classification", model=model_path)

print("Sentiment analysis model for polish texts loaded successfully.")

# Example usage:
# sentiment_pl_analyzer("Nie mogłem uwierzyć własnym oczom — wygrałem główną nagrodę!")
