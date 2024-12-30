import json
import pyttsx3
import speech_recognition as sr
import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import webbrowser
import os
from datetime import datetime
import openai
import random
from time import strftime

# Load pre-trained word embeddings (e.g., GloVe)
print("Loading word embeddings...")
word_vectors = api.load("glove-wiki-gigaword-100")


# Load knowledge base from JSON file
def load_knowledge_base(filename):
    with open(filename, 'r') as file:
        return json.load(file)


# Load the knowledge base
knowledge_base = load_knowledge_base('data.json')


# Text-to-Speech function
def say(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        say(f"I can't hear you, Can you repeat it......: {e}")


# Speech-to-Text function
def take_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            print("Recognizing...")
            query = recognizer.recognize_google(audio, language="en-in")
            print(f"Vijaya said: {query}")
            return query.lower()
        except Exception as e:
            print(f"Speech recognition error - I can't hear you, Can you repeat it: {e}")
            return ""


# NLP: Process and find the best response
def process_query(query):
    # Tokenize user query
    words = [word for word in query.split() if word in word_vectors]

    if not words:
        return "I'm sorry, I don't understand that."

    query_vector = np.mean([word_vectors[word] for word in words], axis=0)

    # Compare with knowledge base
    best_match = None
    max_similarity = -1

    for question, answer in knowledge_base.items():
        question_words = [word for word in question.split() if word in word_vectors]
        if not question_words:
            continue

        question_vector = np.mean([word_vectors[word] for word in question_words], axis=0)

        # Check if both vectors are valid
        if np.any(np.isnan(query_vector)) or np.any(np.isnan(question_vector)):
            continue

        similarity = cosine_similarity([query_vector], [question_vector])[0][0]

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = answer

    return best_match if max_similarity > 0.5 else "I'm sorry, I don't understand that."


# Main AI Assistant loop
def main():
    say("Hello! I am your Desktop AI Assistant. How can I help you?")

    while True:
        query = take_command()
        if not query:
            continue

        if "exit" in query:
            say("Goodbye! Have a great day!")
            break

        sites = [
            ["YouTube", "https://www.youtube.com/"],
            ["Google", "https://www.google.com/"],
            ["WhatsApp", "https://web.whatsapp.com/"],
            ["code", "https://leetcode.com/"],
            ["Hackerrank", "https://www.hackerrank.com/"],
            ["Hackerearth", "https://www.hackerearth.com/"],
            ["Chat GPT", "https://chatgpt.com/"],
            ["GitHub", "https://github.com/"],
            ["LinkedIn", "https://www.linkedin.com/in/"]
        ]

        # Open websites

        for site in sites:
            if f"open {site[0]}".lower() in query.lower():
                say(f"Opening {site[0]}, Vijaya!")
                webbrowser.open(site[1])
        # todo: Add a feature to play a specific song
        # Play music
        if "open music" in query.lower():
            # musicPath = r"C:\Users\vijaya\Downloads\Samayama.mp3"
            musicPath = "C:\\Users\\vijaya\\Downloads\\Samayama.mp3"

            if os.path.exists(musicPath):
                os.startfile(musicPath)
            else:
                say("File not found. Please check the file path.")

        # Tell the time
        if "time" in query.lower():
            current_time = datetime.now().strftime("%H:%M:%S")
            say(f"The time is {current_time}")

        response = process_query(query)
        print(f"Assistant: {response}")
        say(response)


# Run the assistant
if __name__ == "__main__":
    main()