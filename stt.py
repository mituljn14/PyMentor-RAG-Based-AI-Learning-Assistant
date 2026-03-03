import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import whisper

model = whisper.load_model("large-v2")

result = model.transcribe(
    audio="video/1.mp3",
    language="hi",
    task="translate"
)

chunks = []

for seg in result["segments"]:
    chunks.append({
        "start": seg["start"],
        "end": seg["end"],
        "text": seg["text"]
    })

print(chunks)

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)
