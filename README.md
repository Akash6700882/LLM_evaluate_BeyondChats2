
# LLM Response Evaluation Pipeline

## Project Description
This project is a Python-based evaluation pipeline designed to automatically assess AI/LLM-generated responses for **reliability, relevance, and factual accuracy**. It evaluates responses against a given context and measures:

- **Response Relevance & Completeness** – How closely the AI response aligns with the context vectors retrieved from a knowledge base.
- **Hallucination / Factual Accuracy** – Detects sentences in the response that are unsupported or outside the context.
- **Latency & Costs** – Tracks the time taken to compute embeddings and similarity scores.

The pipeline outputs a structured JSON file for each chat conversation, including the response, relevance score, detected hallucinations, and latency.

---

## Features
- Automatically loads chat and context JSONs.
- Uses **SentenceTransformer (`all-MiniLM-L6-v2`)** for semantic embeddings.
- Computes **cosine similarity** to evaluate response relevance.
- Detects hallucinations in responses.
- Measures evaluation latency for performance analysis.
- Scalable design for real-time evaluation.

---

## Technologies Used
- Python 3
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- scikit-learn for cosine similarity
- NumPy for numerical operations

---

## Setup Instructions

1. Clone the repository:

```bash
git clone <your-github-repo-url>
cd beyondchats-evaluator
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the evaluation pipeline:

```bash
python evaluate_pipeline.py
```

> Ensure your `data/` folder contains chat JSON files (`sample-chat-conversation-01.json`) and context JSON files (`sample-context_vectors-01.json`).

---

## Architecture

1. Load chat and context JSON files.
2. Extract the last assistant response from the conversation.
3. Convert context texts into embeddings.
4. Compute cosine similarity between response and context to determine relevance.
5. Detect hallucinated sentences outside the context.
6. Measure latency for evaluation.
7. Save results as JSON.

---

## Design Decisions

* **Embeddings + Cosine Similarity:** Provides semantic understanding rather than exact string matches.
* **Lightweight Model (`all-MiniLM-L6-v2`)**: Fast inference and low computation cost.
* **Scalability:** Supports batch embedding processing for large datasets and real-time evaluation.

---

## Future Improvements

* Use **semantic similarity for hallucination detection** instead of substring matching.
* Add **parallel processing** for large-scale evaluation.
* Extend to evaluate **multi-turn conversations** and overall completeness.

---

## Output

Each chat evaluation produces a JSON file:

```json
{
    "chat_file": "sample-chat-conversation-01.json",
    "response": "AI assistant response text...",
    "relevance_score": 0.8732,
    "hallucinations_found": ["sentence not in context..."],
    "latency_seconds": 0.0456
}
```

This format makes it easy to integrate into analytics dashboards or further evaluation pipelines.

---

##AUTHOR 
M Akash 
IIT Guwahati
9392847778 ||  codes.akash@gmail.com

## License

This project is for educational and internship submission purposes. All code is your property and should not be used without consent.

```

---

If you want, I can **also create a `requirements.txt` and a folder structure example** ready to push to GitHub so your repo is submission-ready.  

