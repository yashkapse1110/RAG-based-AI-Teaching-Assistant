import whisper
import json

model = whisper.load_model("large-v2")

result = model.transcribe(audio = "audios/output10.mp3", language = "hi", task="translate",fp16=False)

chunk =[]
for segment in result["segments"]:
    chunk.append({"start": segment["start"],"end": segment["end"],"text": segment["text"]})



with open("output.json", "w") as f:
    json.dump(chunk,f)
