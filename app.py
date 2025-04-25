from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import os
import speech_recognition as sr
import requests
from dotenv import load_dotenv
from flask_cors import CORS

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

def load_translation_model():
    global tokenizer, model, translator_pipe
    if translator_pipe is None:
        model_name = "Splintir/Nllb_dialecto"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        translator_pipe = pipeline("translation", model=model, tokenizer=tokenizer)

def nllb_model(text, direction):
    load_translation_model()
    src_lang = dictionary["ceb"] if direction == "ceb_to_eng" else dictionary["eng"]
    tgt_lang = dictionary["eng"] if direction == "ceb_to_eng" else dictionary["ceb"]
    translated_text = translator_pipe(text, src_lang=src_lang, tgt_lang=tgt_lang, max_length=400)
    return translated_text[0]['translation_text']

def get_audio_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Could not request results"

@app.route("/microphone", methods=["POST"])
def microphone():
    text = get_audio_input()
    return jsonify({"text": text})

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_input = request.get_json().get("text", "")
    
    return jsonify({
        "translated_text": "debug",
        "chatbot_response(echo)": {user_input}
    })

def get_chatbot_response(text):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "mistral-tiny",
        "messages": [{"role": "user", "content": text}]
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    response_json = response.json()
    return response_json.get("choices", [{}])[0].get("message", {}).get("content", "No response")

@app.route("/")
def home():
    return "Dialecto API is live."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
