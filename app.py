from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
import sqlite3
from datetime import datetime

app = Flask(__name__)

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

                # Save to SQLite DB
                save_to_db(incoming_query, response, mode_label)

                # Save logs
                with open("prompt.txt", "w", encoding="utf-8") as f:
                    f.write(prompt)

                with open("history.txt", "a", encoding="utf-8") as f:
                    f.write("\n\nQ: " + incoming_query)
                    f.write("\nA: " + response)

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
    from flask import redirect, url_for
    return redirect(url_for("home"))


@app.route("/clear_history", methods=["POST"])
def clear_history():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()
    from flask import redirect, url_for
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
