"""Subagente 2 — Transcriptor: MP3 → texto con diarización (WhisperX)"""
import os
from pathlib import Path

MODEL_DIR = "/home/chuy/models/whisper"


def run(audio_path: Path, api_key: str = None) -> str:
    """Transcribe con WhisperX. Usa cache si ya existe el transcript."""
    cache_path = audio_path.with_suffix(".transcript.txt")
    if cache_path.exists():
        print(f"   → Cache encontrado, cargando {cache_path.name}")
        return cache_path.read_text(encoding="utf-8")

    transcript = _transcribe(audio_path)
    cache_path.write_text(transcript, encoding="utf-8")
    return transcript


def _transcribe(audio_path: Path) -> str:
    import whisperx

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Transcripción con detección automática de idioma
    model = whisperx.load_model(
        "medium", device="cpu", compute_type="int8",
        download_root=MODEL_DIR
    )
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=8)
    detected_lang = result.get("language", "es")

    # 2. Alineación word-level
    align_model, metadata = whisperx.load_align_model(
        language_code=detected_lang, device="cpu"
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device="cpu"
    )

    # 3. Diarización (quién habla cuándo)
    HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")
    if HF_TOKEN:
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers
        diarize_model = DiarizationPipeline(token=HF_TOKEN, device="cpu")
        diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=10)
        result = assign_word_speakers(diarize_segments, result, fill_nearest=True)

    # 4. Formatear como texto con hablantes
    lines = []
    for seg in result["segments"]:
        speaker = seg.get("speaker", "SPEAKER_?")
        text = seg.get("text", "").strip()
        start = seg.get("start", 0)
        lines.append(f"[{speaker} {start:.1f}s] {text}")

    return "\n".join(lines)
