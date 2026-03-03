from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests

app = Flask(__name__)

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
    return r.json()["response"]

# -------- Load Vector DB --------
df = joblib.load('embeddings.joblib')


@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    incoming_query = ""

    if request.method == "POST":
        incoming_query = request.form["question"]

        long_mode = incoming_query.lower().startswith(
            ("explain", "describe", "how", "in detail")
        )
        mode_text = "FULL detailed explanation" if long_mode else "SHORT precise answer"

        try:
            question_embedding = create_embedding([incoming_query])[0]

            similarities = cosine_similarity(
                np.vstack(df['embedding']),
                [question_embedding]
            ).flatten()

            top_results = 5
            max_indx = similarities.argsort()[::-1][:top_results]
            new_df = df.loc[max_indx]

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

            response = inference(prompt)

            # Save logs (same as your CLI)
            with open("prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)

            with open("history.txt", "a", encoding="utf-8") as f:
                f.write("\n\nQ: " + incoming_query)
                f.write("\nA: " + response)

        except Exception as e:
            response = f"Error: {str(e)}"

    return render_template("index.html", answer=response, question=incoming_query)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
