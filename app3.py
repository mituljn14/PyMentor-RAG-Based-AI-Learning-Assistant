from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import sqlite3
from datetime import datetime
import google.generativeai as genai
import os
import time

app = Flask(__name__)

# -------- 🔐 Gemini API Setup (SAFE) --------
# Set this in terminal: export GEMINI_API_KEY=your_key
GEMINI_API_KEY = "AIzaSyAaaae7bq1PL2ZMCeTfT5zPzjk-wdl1zSw" # 🔑 Replace with your key genai.configure(api_key=GEMINI_API_KEY)

# -------- ✅ FIXED MODELS --------
EMBEDDING_MODEL = "models/embedding-001"   # stable embedding model
GENERATION_MODEL = "gemini-1.5-pro"        # ✅ no more 404 error


# -------- DB Setup --------
def init_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            mode TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_to_db(question, answer, mode):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (question, answer, mode, timestamp) VALUES (?, ?, ?, ?)",
        (question, answer, mode, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()


def get_history(limit=20):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, question, answer, mode, timestamp FROM chat_history ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def delete_history_entry(entry_id):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()


# -------- ✅ FIXED EMBEDDING FUNCTION --------
def create_embedding(text):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document"   # keep consistent
    )
    return result["embedding"]


# -------- ✅ FIXED LLM FUNCTION --------
def inference(prompt):
    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt)
    return response.text


# -------- Load Vector DB --------
print("Loading embeddings.joblib...")
df = joblib.load('embeddings.joblib')
print(f"Loaded {len(df)} rows | Dim: {len(df.iloc[0]['embedding'])}")

init_db()


@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    incoming_query = ""
    mode_label = ""

    if request.method == "POST":
        incoming_query = request.form.get("question", "").strip()

        if incoming_query:
            long_mode = incoming_query.lower().startswith(
                ("explain", "describe", "how", "in detail")
            )

            mode_label = "Detailed" if long_mode else "Concise"
            mode_text = "FULL detailed explanation" if long_mode else "SHORT precise answer"

            try:
                # -------- Embed Query --------
                question_embedding = create_embedding(incoming_query)

                # -------- Similarity Search --------
                similarities = cosine_similarity(
                    np.vstack(df['embedding']),
                    [question_embedding]
                ).flatten()

                top_k = 5
                top_indices = similarities.argsort()[::-1][:top_k]
                new_df = df.loc[top_indices]

                # -------- Prompt --------
                prompt = f"""
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
- Mention video number, title, timestamps.
- Use ONLY given data.
"""

                # -------- LLM Call --------
                response = inference(prompt)

                # -------- Save --------
                save_to_db(incoming_query, response, mode_label)

            except Exception as e:
                response = f"Error: {str(e)}"

    history = get_history(20)

    return render_template(
        "index2.html",
        answer=response,
        question=incoming_query,
        mode_label=mode_label,
        history=history
    )


@app.route("/delete/<int:entry_id>", methods=["POST"])
def delete_entry(entry_id):
    delete_history_entry(entry_id)
    return redirect(url_for("home"))


@app.route("/clear_history", methods=["POST"])
def clear_history():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True, threaded=True)