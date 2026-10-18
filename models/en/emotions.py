# https://huggingface.co/j-hartmann/emotion-english-distilroberta-base?library=transformers

from transformers import pipeline

model_path = "j-hartmann/emotion-english-distilroberta-base"

emotion_en_analyzer = pipeline("text-classification", model=model_path)

print("Emotion recognition model for english texts loaded successfully.")

# Example usage:
# emotion_en_analyzer("I heard footsteps behind me in the dark, and my heart started racing.")