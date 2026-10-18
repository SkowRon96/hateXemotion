# Original emotion recogntion model for Polish language based on ChatGPT dataset

import json
import requests

url = "http://157.158.14.30:8003/predict"

def emotion_chatgpt_pl_analyzer(text: str) -> dict:
    response = requests.request("POST", url, headers={'Content-Type': 'text/plain'}, data=text)
    return [json.loads(response.text)]

print("Original emotion recogntion model for Polish language based on ChatGPT dataset loaded successfully.")

# Example usage:
# emotion_chatgpt_pl_analyzer("Mam już dość tego, że ciągle mnie ignorujesz!")