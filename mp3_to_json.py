import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import whisper
import json
import textwrap

model = whisper.load_model("large-v2")

audios = os.listdir("audios")

MAX_CHARS = 150   # approx 3 lines (you can tune: 80–150)

for audio in audios:
    if audio.endswith(".mp3"):
        name = audio.replace(".mp3", "")
        title = name
        number = name.replace("part", "")

        print("Title:", title, "Number:", number)

        result = model.transcribe(
            audio=f"audios/{audio}",
            language="hi",
            task="translate",
            word_timestamps=False
        )

        chunks = []
        for segment in result["segments"]:
            small_texts = textwrap.wrap(segment["text"], MAX_CHARS)

            seg_duration = segment["end"] - segment["start"]
            step = seg_duration / len(small_texts)

            for i, txt in enumerate(small_texts):
                start = segment["start"] + i * step
                end = start + step

                chunks.append({
                    "number": number,
                    "title": title,
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "text": txt
                })

        chunks_with_metadata = {
            "chunks": chunks,
            "text": result["text"]
        }

        with open(f"newjsons/{audio}.json", "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)
