import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from app.config.settings import GEMINI_API_KEY, client

model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = None

def load_bert_model():
    global model, tokenizer, df

    print("⏳ Loading BERT model...")

    try:
        load_directory = "Halo536/bert-intent-model"
        tokenizer = BertTokenizer.from_pretrained(load_directory)
        model = BertForSequenceClassification.from_pretrained(load_directory)
        model.to(device)
        model.eval()

        df = pd.read_csv("processed_data_main.csv")
        if "labels" not in df.columns:
            raise KeyError("Missing 'labels' column in CSV")
        if "intent" not in df.columns:
            df["intent"] = df["labels"].astype(str)

        print("✅ BERT and intent mapping loaded.")
    except Exception as e:
        print("❌ Failed to load BERT or data:", e)
        raise e

def predict_intent(text):
    global model, tokenizer, df

    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    intent_mapping = dict(zip(df["labels"].unique(), df["intent"].unique()))
    return intent_mapping.get(predicted_class, "Unknown Intent")
