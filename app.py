from flask import Flask, request, jsonify, render_template
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# Load IS456 data (using relative path for portability)
with open("is456_data.json", "r") as f:
    is_data = json.load(f)

# Load Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create list of clause texts
texts = [item["content"] for item in is_data]
embeddings = model.encode(texts, batch_size=16, convert_to_numpy=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def get_answer(query):
    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, k=1)
    result = is_data[I[0][0]]
    return result

# Route to serve the HTML interface
@app.route("/")
def home():
    return render_template("index.html")

# API route to handle the chat messages
@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    answer = get_answer(user_query)
    return jsonify(answer)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)