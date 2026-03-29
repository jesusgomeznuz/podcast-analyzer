"""Subagente 2 — Transcriptor: MP3 → texto con diarización
Estrategia: Mac M4 via SSH (mlx-whisper, Metal) → fallback ThinkPad CPU (WhisperX)
"""
import os
import subprocess
import tempfile
from pathlib import Path

MODEL_DIR = "/home/chuy/models/whisper"
MAC_HOST = "jesus@192.168.1.131"
MAC_CONDA_ENV = "filtro"
MAC_CONDA = "/Users/jesus/miniconda3/etc/profile.d/conda.sh"


def run(audio_path: Path, api_key: str = None) -> str:
    """Transcribe con diarización. Usa caché si ya existe."""
    cache_path = audio_path.with_suffix(".transcript.txt")
    if cache_path.exists():
        print(f"   → Cache encontrado, cargando {cache_path.name}")
        return cache_path.read_text(encoding="utf-8")

    if _mac_available():
        print("   → Mac M4 detectada, usando mlx-whisper + Metal")
        transcript = _transcribe_mac(audio_path)
    else:
        print("   → Mac no disponible, usando WhisperX en CPU (ThinkPad)")
        transcript = _transcribe_local(audio_path)

    cache_path.write_text(transcript, encoding="utf-8")
    return transcript


def _mac_available() -> bool:
    """Verifica si la Mac está encendida y accesible por SSH."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             MAC_HOST, "echo ok"],
            capture_output=True, text=True, timeout=8
        )
        return result.returncode == 0
    except Exception:
        return False


def _transcribe_mac(audio_path: Path) -> str:
    """Transcribe en Mac M4 via SSH usando mlx-whisper + pyannote diarización."""
    # Subir audio a Mac
    remote_audio = f"/tmp/{audio_path.name}"
    subprocess.run(
        ["scp", str(audio_path), f"{MAC_HOST}:{remote_audio}"],
        check=True
    )

    HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")

    PYTHON = f"/Users/jesus/miniconda3/envs/{MAC_CONDA_ENV}/bin/python3"
    script = f"""
export PATH="/opt/homebrew/bin:$PATH"
{PYTHON} - <<'PYEOF'
import mlx_whisper, sys, warnings
warnings.filterwarnings("ignore")

audio_path = "{remote_audio}"
hf_token = "{HF_TOKEN}"

# 1. Transcripción con mlx-whisper large-v3-turbo (Metal)
result = mlx_whisper.transcribe(
    audio_path,
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    word_timestamps=True
)
segments = result.get("segments", [])

# 2. Diarización con pyannote directamente
if hf_token:
    try:
        import torch
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        # Convertir a WAV mono 16kHz para pyannote (soundfile no lee MP3)
        import subprocess as sp, torch, soundfile as sf, os
        wav_path = audio_path.replace(".mp3", "_diar.wav")
        sp.run(["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000",
                wav_path], capture_output=True, check=True)
        waveform, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform.T)
        audio_dict = {{"waveform": waveform, "sample_rate": sample_rate}}
        output = pipeline(audio_dict, min_speakers=2, max_speakers=10)
        os.remove(wav_path)
        # pyannote 4.x retorna DiarizeOutput — extraer Annotation
        diarization = getattr(output, "speaker_diarization", output)

        # Asignar hablante a cada segmento por overlap de tiempo
        def get_speaker(start, end, diarization):
            best_spk, best_overlap = "SPEAKER_?", 0
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                overlap = min(turn.end, end) - max(turn.start, start)
                if overlap > best_overlap:
                    best_overlap, best_spk = overlap, speaker
            return best_spk

        for seg in segments:
            seg["speaker"] = get_speaker(seg["start"], seg["end"], diarization)

    except Exception as e:
        print(f"Diarización falló: {{e}}", file=sys.stderr)

# 3. Formatear
lines = []
for seg in segments:
    speaker = seg.get("speaker", "SPEAKER_?")
    text = seg.get("text", "").strip()
    start = seg.get("start", 0)
    lines.append(f"[{{speaker}} {{start:.1f}}s] {{text}}")

print("\\n".join(lines))
PYEOF
"""

    result = subprocess.run(
        ["ssh", MAC_HOST, script],
        capture_output=True, text=True, timeout=7200
    )

    # Limpiar archivo temporal en Mac
    subprocess.run(["ssh", MAC_HOST, f"rm -f {remote_audio}"], capture_output=True)

    if result.returncode != 0:
        raise RuntimeError(f"Mac transcripción falló: {result.stderr[-500:]}")

    # Filtrar stderr (warnings) — stdout tiene el transcript
    transcript = result.stdout.strip()
    if not transcript:
        raise RuntimeError("Mac retornó transcript vacío")

    return transcript


def _transcribe_local(audio_path: Path) -> str:
    """Fallback: transcripción local con WhisperX en CPU."""
    import whisperx

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    model = whisperx.load_model(
        "medium", device="cpu", compute_type="int8",
        download_root=MODEL_DIR
    )
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=8)
    detected_lang = result.get("language", "es")

    align_model, metadata = whisperx.load_align_model(
        language_code=detected_lang, device="cpu"
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device="cpu"
    )

    HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")
    if HF_TOKEN:
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers
        diarize_model = DiarizationPipeline(token=HF_TOKEN, device="cpu")
        diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=10)
        result = assign_word_speakers(diarize_segments, result, fill_nearest=True)

    lines = []
    for seg in result["segments"]:
        speaker = seg.get("speaker", "SPEAKER_?")
        text = seg.get("text", "").strip()
        start = seg.get("start", 0)
        lines.append(f"[{speaker} {start:.1f}s] {text}")

    return "\n".join(lines)
