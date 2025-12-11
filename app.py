from fastapi import FastAPI, UploadFile, File, Form
import json
import time
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

DATA_DIR = "data"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def embed(texts, model):
    return model.encode(texts)

def evaluate_relevance(response, context_embeds, model):
    resp_embed = embed([response], model)
    scores = cosine_similarity(resp_embed, context_embeds)[0]
    return float(np.max(scores))

def check_hallucination(response, context_text):
    hallucinated_points = []
    sentences = response.split(".")
    for s in sentences:
        s = s.strip()
        if len(s) < 5:
            continue
        if s.lower() not in context_text.lower():
            hallucinated_points.append(s)
    return hallucinated_points

@app.get("/")
def home():
    return {"message": "Evaluation API is running"}

@app.post("/evaluate")
async def evaluate(
    chat_file: UploadFile = File(...),
    context_file: UploadFile = File(...)
):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chat = json.load(chat_file.file)
    context = json.load(context_file.file)

    assistant_response = None
    for msg in reversed(chat.get("conversation_turns", [])):
        if msg["role"] == "AI/Chatbot":
            assistant_response = msg["message"]
            break

    context_texts = [c["text"] for c in context["data"]["vector_data"] if "text" in c]
    context_embeds = embed(context_texts, model)

    start = time.time()
    relevance_score = evaluate_relevance(assistant_response, context_embeds, model)
    latency = time.time() - start

    concatenated_context = " ".join(context_texts)
    hallucinations = check_hallucination(assistant_response, concatenated_context)

    return {
        "response": assistant_response,
        "relevance_score": relevance_score,
        "hallucinations_found": hallucinations,
        "latency_seconds": latency
    }
