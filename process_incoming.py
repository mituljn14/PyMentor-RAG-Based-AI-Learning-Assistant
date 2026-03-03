import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests


# -------- Embedding Function --------
def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]


# -------- LLM Inference Function --------
def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    return r.json()


# -------- Load Vector DB --------
df = joblib.load('embeddings.joblib')


print("AI Tutor Ready. Ask questions about your course.")
print("Type 'exit' to quit.\n")


# -------- Chat Loop --------
while True:
    incoming_query = input("Ask a Question: ")

    if incoming_query.lower() in ["exit", "quit", "q"]:
        print("Goodbye 👋")
        break

    # Intent detection (long or short)
    long_mode = incoming_query.lower().startswith(
        ("explain", "describe", "how", "in detail")
    )
    mode_text = "FULL detailed explanation" if long_mode else "SHORT precise answer"

    # Create embedding for query
    try:
        question_embedding = create_embedding([incoming_query])[0]
    except Exception as e:
        print("Embedding API error:", e)
        continue

    # Similarity search
    similarities = cosine_similarity(
        np.vstack(df['embedding']),
        [question_embedding]
    ).flatten()

    top_results = 5
    max_indx = similarities.argsort()[::-1][:top_results]
    new_df = df.loc[max_indx]

    # Prompt for LLM
    prompt = f'''
You are an AI tutor for a Python course.

DATA:
{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

USER QUESTION:
"{incoming_query}"

TASK MODE:
{mode_text}

RULES:
- If FULL mode → explain deeply with theory + examples.
- If SHORT mode → 2–4 lines only.

In both cases you MUST:
- Mention video number and title.
- Mention exact timestamps.
- Explain what is taught at those timestamps.
- Guide user where to watch.

Do NOT use outside knowledge.
Only use the given video data.
'''

    # Get answer
    response = inference(prompt)["response"]

    print("\nAnswer:\n")
    print(response)

    # Save last prompt
    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    # Save chat history
    with open("history.txt", "a", encoding="utf-8") as f:
        f.write("\n\nQ: " + incoming_query)
        f.write("\nA: " + response)   
