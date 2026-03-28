"""Subagente 2 — Transcriptor: MP3 → texto con timestamps"""
from openai import OpenAI, AuthenticationError
from pathlib import Path

# Modelos Whisper en el SSD para no llenar la partición interna
WHISPER_MODEL_DIR = "/mnt/ssd/linux/models"


def run(audio_path: Path, api_key: str) -> str:
    try:
        client = OpenAI(api_key=api_key)
        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="text"
            )
        return response
    except (AuthenticationError, Exception) as e:
        print(f"   ⚠️  API falló ({type(e).__name__}), usando Whisper local (tiny)...")
        return _local(audio_path)


def _local(audio_path: Path) -> str:
    import whisper
    # "tiny" para mayor velocidad en CPU — sacrifica algo de precisión
    model = whisper.load_model("tiny", download_root=WHISPER_MODEL_DIR)
    result = model.transcribe(str(audio_path), fp16=False)
    return result["text"]
