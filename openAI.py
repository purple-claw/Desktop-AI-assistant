from flask import Flask, request, jsonify, render_template
import json
import pyttsx3
import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import openai
import random

app = Flask(__name__)

# Load pre-trained word embeddings (e.g., GloVe)
print("Loading word embeddings...")
word_vectors = api.load("glove-wiki-gigaword-100")

# Load knowledge base from JSON file
def load_knowledge_base(filename):
    with open(filename, 'r') as file:
        return json.load(file)

knowledge_base = load_knowledge_base('data.json')

# Text-to-Speech function (For desktop use; disabled in Flask)
def say(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Text-to-Speech Error: {e}")

# NLP: Process and find the best response
def process_query(query):
    words = [word for word in query.split() if word in word_vectors]
    if not words:
        return "I'm sorry, I don't understand that."

    query_vector = np.mean([word_vectors[word] for word in words], axis=0)
    best_match = None
    max_similarity = -1

    for question, answer in knowledge_base.items():
        question_words = [word for word in question.split() if word in word_vectors]
        if not question_words:
            continue

        question_vector = np.mean([word_vectors[word] for word in question_words], axis=0)
        if np.any(np.isnan(query_vector)) or np.any(np.isnan(question_vector)):
            continue

        similarity = cosine_similarity([query_vector], [question_vector])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = answer

    return best_match if max_similarity > 0.5 else "I'm sorry, I don't understand that."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("query", "")
    if "time" in query.lower():
        current_time = datetime.now().strftime("%H:%M:%S")
        response = f"The time is {current_time}"
    else:
        response = process_query(query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
#/home/vijaya136/mysite/flask_app.py