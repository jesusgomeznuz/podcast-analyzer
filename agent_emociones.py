"""Subagente — Análisis emocional: emoción categórica + dimensional por segmento de hablante
Estrategia: Mac M4 via SSH → fallback ThinkPad CPU
"""
import os
import re
import sys
import time
import subprocess
import tempfile
import torch
import librosa
import numpy as np
from pathlib import Path

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")
WAVLM_MODEL_DIR = "/tmp/vox-profile-release"
MAC_HOST = "jesus@192.168.1.131"
MAC_PY = "/Users/jesus/miniconda3/envs/podcast-analyzer/bin/python3"
MAC_SCRIPTS_DIR = Path(__file__).parent / "mac_scripts"

# Mapeo de etiquetas emotion2vec (chino/inglés) → español
EMOTION_MAP = {
    "中立/neutral": "neutral",
    "开心/happy": "feliz",
    "难过/sad": "triste",
    "愤怒/angry": "enojado",
    "厌恶/disgusted": "disgustado",
    "恐惧/fearful": "temeroso",
    "惊讶/surprised": "sorprendido",
    "吃惊/surprised": "sorprendido",
    "其他/other": "otro",
    "未知/unknown": "desconocido",
    "neutral": "neutral",
    "happy": "feliz",
    "sad": "triste",
    "angry": "enojado",
    "disgusted": "disgustado",
    "fearful": "temeroso",
    "surprised": "sorprendido",
}


def _mac_available() -> bool:
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             MAC_HOST, "echo ok"],
            capture_output=True, text=True, timeout=8
        )
        return result.returncode == 0
    except Exception:
        return False


def _run_on_mac(audio_path: Path, transcript: str) -> str:
    remote_audio = f"/tmp/{audio_path.name}"
    remote_transcript = f"/tmp/{audio_path.stem}_transcript.txt"
    remote_script = "/tmp/emociones_mac.py"

    # SCP audio
    subprocess.run(["scp", str(audio_path), f"{MAC_HOST}:{remote_audio}"], check=True)

    # SCP transcript
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(transcript)
        local_transcript = f.name
    subprocess.run(["scp", local_transcript, f"{MAC_HOST}:{remote_transcript}"], check=True)
    os.unlink(local_transcript)

    # SCP script
    subprocess.run(["scp", str(MAC_SCRIPTS_DIR / "emociones_mac.py"), f"{MAC_HOST}:{remote_script}"], check=True)

    # Run on Mac
    cmd = f'export PATH="/opt/homebrew/bin:/usr/bin:/bin:$PATH" && {MAC_PY} {remote_script} {remote_audio} {remote_transcript} "{HF_TOKEN}"'
    result = subprocess.run(["ssh", MAC_HOST, cmd], capture_output=True, text=True, timeout=7200)

    # Cleanup
    subprocess.run(["ssh", MAC_HOST, f"rm -f {remote_audio} {remote_transcript} {remote_script}"], capture_output=True)

    if result.returncode != 0:
        raise RuntimeError(f"Mac emociones falló: {result.stderr[-500:]}")

    output = result.stdout.strip()
    if not output:
        raise RuntimeError("Mac retornó output vacío")
    return output


def parse_transcript_segments(transcript: str) -> list:
    """Parsea el transcript de WhisperX en segmentos con hablante y tiempo."""
    segments = []
    for line in transcript.strip().split("\n"):
        m = re.match(r"\[([^\] ]+) ([\d.]+)s\] (.+)", line)
        if m:
            segments.append({
                "speaker": m.group(1),
                "start": float(m.group(2)),
                "text": m.group(3),
            })
    return segments


def run(audio_path: Path, transcript: str) -> str:
    """Analiza emociones por segmento de hablante. Usa Mac M4 si disponible."""
    if _mac_available():
        try:
            print("   → Mac M4 detectada, usando Neural Engine para emociones")
            return _run_on_mac(audio_path, transcript)
        except Exception as e:
            print(f"   → Mac falló ({e}), usando CPU local")

    print("   → Procesando emociones en CPU local (ThinkPad)")
    return _run_local(audio_path, transcript)


def _run_local(audio_path: Path, transcript: str) -> str:
    """Fallback: análisis local en ThinkPad CPU."""
    t0 = time.time()
    y, sr = librosa.load(str(audio_path), sr=16000)
    segments = parse_transcript_segments(transcript)

    if not segments:
        return "No hay segmentos para analizar."

    for i, seg in enumerate(segments):
        if i + 1 < len(segments):
            seg["end"] = segments[i + 1]["start"]
        else:
            seg["end"] = len(y) / sr

    results_emotion2vec = _emotion2vec(y, sr, segments)
    results_wavlm = _wavlm_dimensional(y, sr, segments)

    elapsed = time.time() - t0
    lines = [f"# Análisis Emocional ({elapsed:.1f}s)\n"]
    lines.append("## Por segmento de hablante\n")

    for i, seg in enumerate(segments):
        e2v = results_emotion2vec[i] if i < len(results_emotion2vec) else "—"
        wlm = results_wavlm[i] if i < len(results_wavlm) else "—"
        lines.append(
            f"[{seg['speaker']} {seg['start']:.1f}s] {seg['text'][:60]}\n"
            f"  → emoción: {e2v} | arousal/valence/dominance: {wlm}\n"
        )

    return "\n".join(lines)


def _emotion2vec(y: np.ndarray, sr: int, segments: list) -> list:
    from funasr import AutoModel
    model = AutoModel(model="iic/emotion2vec_plus_base", disable_update=True)
    results = []
    for seg in segments:
        start_s = int(seg["start"] * sr)
        end_s = int(seg["end"] * sr)
        chunk = y[start_s:end_s]
        if len(chunk) < sr // 2:
            results.append("muy corto")
            continue
        try:
            rec = model.generate(chunk.astype(np.float32), sample_rate=sr,
                                 granularity="utterance", extract_embedding=False)
            if rec:
                raw_label = rec[0]["labels"][np.argmax(rec[0]["scores"])]
                label = EMOTION_MAP.get(raw_label, raw_label)
                score = max(rec[0]["scores"])
                results.append(f"{label} ({score:.2f})")
            else:
                results.append("—")
        except Exception:
            results.append("error")
    return results


def _wavlm_dimensional(y: np.ndarray, sr: int, segments: list) -> list:
    sys.path.insert(0, WAVLM_MODEL_DIR)
    from src.model.emotion.wavlm_emotion_dim import WavLMWrapper
    model = WavLMWrapper.from_pretrained(
        "tiantiaf/wavlm-large-msp-podcast-emotion-dim"
    )
    model.eval()
    results = []
    for seg in segments:
        start_s = int(seg["start"] * sr)
        end_s = int(seg["end"] * sr)
        chunk = y[start_s:end_s]
        if len(chunk) < sr:
            results.append("muy corto")
            continue
        chunk = chunk[:sr * 15]
        data = torch.tensor(chunk).unsqueeze(0).float()
        try:
            with torch.no_grad():
                arousal, valence, dominance = model(data)
            results.append(
                f"A={arousal.item():.2f} V={valence.item():.2f} D={dominance.item():.2f}"
            )
        except Exception:
            results.append("error")
    return results
