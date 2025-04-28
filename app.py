from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import os
import speech_recognition as sr
import requests
from dotenv import load_dotenv
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

# Load env
load_dotenv("deepkey.env")
MISTRAL_KEY = os.getenv("MISTRAL_KEY")

# Lazy loading vars
tokenizer = None
model = None
translator_pipe = None

dictionary = {
    'eng': "eng_Latn",
    'ceb': "ceb_Latn"
}

def nllb_model(text, direction):
    try:
        load_translation_model()
        src_lang = dictionary["ceb"] if direction == "ceb_to_eng" else dictionary["eng"]
        tgt_lang = dictionary["eng"] if direction == "ceb_to_eng" else dictionary["ceb"]
        translated_text = translator_pipe(text, src_lang=src_lang, tgt_lang=tgt_lang, max_length=400)
        return translated_text[0]['translation_text']
    except Exception as e:
        logging.error(f"Translation Error: {e}", exc_info=True)
        return "Translation failed"


def load_translation_model():
    global tokenizer, model, translator_pipe
    if translator_pipe is None:
        model_name = "Splintir/Nllb_dialecto"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"  # optional
        )

        translator_pipe = pipeline(
            "translation", 
            model=model, 
            tokenizer=tokenizer,
            device=0  # if using GPU, or device=-1 for CPU
        )

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_input = request.get_json().get("text", "")
    direction = "ceb_to_eng"
    try:
        translated_text = nllb_model(user_input, direction)
    except Exception as e:
        print("Translation Error:", e)
        return jsonify({"error": "Translation failed"}), 500

    try:
        chatbot_response = get_chatbot_response(translated_text)
    except Exception as e:
        print("Chatbot API Error:", e)
        return jsonify({"error": "Chatbot request failed"}), 500

    return jsonify({
        "original_text": user_input,
        "translated_text": translated_text,
        "chatbot_response": chatbot_response  # Changed the key name
    })


def get_chatbot_response(text):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "mistral-tiny",
        "messages": [{"role": "user", "content": text}]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "No response")
    except requests.exceptions.RequestException as e:
        logging.error(f"Chatbot API Error: {e}", exc_info=True)
        return "Chatbot request failed"


@app.route("/")
def home():
    return "Dialecto API is live."

def init_models():
    load_translation_model()
    print("Models initialized successfully.")

if __name__ == "__main__":
    init_models()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
