# https://huggingface.co/visegradmedia-emotion/Emotion_RoBERTa_polish6
#anger, fear, disgust, sadness, joy, and none of them
from transformers import pipeline

model_path = "visegradmedia-emotion/Emotion_RoBERTa_polish6"

emotion_pl_analyzer = pipeline("text-classification", model=model_path)

emotion_pl_analyzer_mapper = {
    "LABEL_0": 'anger',
    "LABEL_1": 'fear',
    "LABEL_2": 'disgust',
    "LABEL_3": 'sadness',
    "LABEL_4": 'happiness',
    "LABEL_5": 'neutral',
}

print("Emotion recognition model for polish texts loaded successfully.")

# Example usage:
# emotion_pl_analyzer("Mam już dość tego, że ciągle mnie ignorujesz!")
