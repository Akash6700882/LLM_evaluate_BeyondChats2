import json
import time
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Path to your data folder
DATA_DIR = "data"

def load_json(path):
    """Load JSON file safely."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def embed(texts, model):
    """Get embeddings for a list of texts."""
    return model.encode(texts)

def evaluate_relevance(response, context_embeds, model):
    """Check similarity between response and context embeddings."""
    resp_embed = embed([response], model)
    scores = cosine_similarity(resp_embed, context_embeds)[0]
    return float(np.max(scores))

def check_hallucination(response, context_text):
    """Detect hallucinated points outside context."""
    hallucinated_points = []
    sentences = response.split(".")
    for s in sentences:
        s = s.strip()
        if len(s) < 5:
            continue
        if s.lower() not in context_text.lower():
            hallucinated_points.append(s)
    return hallucinated_points

def get_matching_context_file(chat_file):
    """
    Try to find the matching context file in DATA_DIR automatically.
    Looks for any file containing the same number as chat_file.
    """
    match = re.search(r"(\d+)", chat_file)
    if not match:
        return None
    num = match.group(1)
    for f in os.listdir(DATA_DIR):
        if "context_vectors" in f and num in f:
            return os.path.join(DATA_DIR, f)
    return None

def main():
    print("Loading model…")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Loop through all chat JSON files in the data folder
    for file_name in os.listdir(DATA_DIR):
        if "chat-conversation" in file_name and file_name.endswith(".json"):
            chat_path = os.path.join(DATA_DIR, file_name)
            context_path = get_matching_context_file(file_name)

            if not context_path or not os.path.exists(context_path):
                print(f"Context file not found for {file_name}, skipping.")
                continue

            chat = load_json(chat_path)
            context = load_json(context_path)

            # Extract last assistant response
            assistant_response = None
            for msg in reversed(chat.get("conversation_turns", [])):
                if msg.get("role") == "AI/Chatbot":
                    assistant_response = msg.get("message")
                    break

            if assistant_response is None:
                print(f"No assistant response found in {file_name}, skipping.")
                continue

            # Prepare context embeddings safely
            context_texts = [c["text"] for c in context.get("data", {}).get("vector_data", []) if "text" in c]
            if not context_texts:
                print(f"No valid context texts in {context_path}, skipping.")
                continue

            context_embeds = embed(context_texts, model)

            # Evaluate relevance & latency
            start = time.time()
            relevance_score = evaluate_relevance(assistant_response, context_embeds, model)
            latency = time.time() - start

            # Hallucination check
            concatenated_context = " ".join(context_texts)
            hallucinations = check_hallucination(assistant_response, concatenated_context)

            # Prepare result
            result = {
                "chat_file": file_name,
                "response": assistant_response,
                "relevance_score": round(relevance_score, 4),
                "hallucinations_found": hallucinations,
                "latency_seconds": round(latency, 4)
            }

            # Save evaluation result
            output_file = os.path.join(f"evaluation_result_{file_name}")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)

            print(f"Saved evaluation for {file_name} -> {output_file}")

if __name__ == "__main__":
    main()
