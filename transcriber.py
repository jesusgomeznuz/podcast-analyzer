from openai import OpenAI, AuthenticationError
from pathlib import Path


def transcribe(audio_path: Path, api_key: str) -> str:
    # Intenta primero con la API de OpenAI (rápido, en la nube)
    try:
        client = OpenAI(api_key=api_key)
        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        return response
    except (AuthenticationError, Exception) as e:
        print(f"   ⚠️  API falló ({type(e).__name__}), usando Whisper local...")
        return _transcribe_local(audio_path)


def _transcribe_local(audio_path: Path) -> str:
    import whisper
    # Modelos se descargan al SSD para no llenar la partición interna
    model = whisper.load_model("base", download_root="/mnt/ssd/linux/models")
    result = model.transcribe(str(audio_path))
    return result["text"]
