from openai import OpenAI
from pathlib import Path


def transcribe(audio_path: Path, api_key: str) -> str:
    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text"
        )
    return response
