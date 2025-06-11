from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from google import genai
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()

model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ Error: GEMINI_API_KEY not set.")
    exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)


def load_bert_model():
    global model, tokenizer, df

    print("⏳ Loading BERT intent classification model...")

    try:
        load_directory = "Halo536/bert-intent-model"
        tokenizer = BertTokenizer.from_pretrained(load_directory)
        model = BertForSequenceClassification.from_pretrained(load_directory)
        model.to(device)
        model.eval()

        print("✅ Model loaded and moved to device:", device)

        print("⏳ Loading intent mapping CSV...")
        df = pd.read_csv("processed_data_main.csv")

        if "labels" not in df.columns:
            raise KeyError("Missing 'labels' column in CSV")
        if "intent" not in df.columns:
            print("⚠️ 'intent' column not found, using 'labels' as fallback")
            df["intent"] = df["labels"].astype(str)

        print("✅ Intent mapping loaded.")
    except Exception as e:
        print("❌ Failed to load model or data:", e)
        exit(1)

def predict_intent(text):
    global model, tokenizer, df

    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    intent_mapping = dict(zip(df["labels"].unique(), df["intent"].unique()))
    intent_label = intent_mapping.get(predicted_class, "Unknown Intent")

    return intent_label


@app.route('/')
@cross_origin()
def home():
    return jsonify({"message": "Welcome to the Home route"})


@app.route('/generate-intent', methods=['POST'])
@cross_origin()
def generate_text():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Invalid request: "prompt" not present in data'})

        user_prompt = data['prompt']
        print(f"📝 Received prompt: '{user_prompt}'")

        final_prompt = 'Categorize this in this category in 2 words only question related to credit cards and debit cards are consirdered accounts questions for the faws any theretical question that can have the same answer for all user is considered faq question and anything other than these are complaint questions. Also no formating on the text just plain text: (FAQ Question), (Complaint Question), (Account Question), (Greetings Question) : ' + user_prompt

        response = client.models.generate_content(model="gemini-2.0-flash", contents=final_prompt)
        text = response.text

        if "complaint" in text.lower():
            return jsonify({"prompt": user_prompt, "ai_response": "qa_complaint"})

        if "greetings" in text.lower():
            return jsonify({"prompt": user_prompt, "ai_response": "qa_greetings"})

        text = predict_intent(user_prompt)  
        return jsonify({"prompt": user_prompt, "ai_response": text})
    except Exception as e:
        print("❌ Error in /generate-text:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_bert_model()
    print("🚀 Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
