"""Subagente — Análisis emocional: emoción categórica + dimensional por segmento de hablante"""
import os
import re
import sys
import time
import torch
import librosa
import numpy as np
from pathlib import Path

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")
WAVLM_MODEL_DIR = "/tmp/vox-profile-release"

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


def parse_transcript_segments(transcript: str) -> list:
    """Parsea el transcript de WhisperX en segmentos con hablante y tiempo."""
    segments = []
    for line in transcript.strip().split("\n"):
        m = re.match(r"\[(\w+_\d+) ([\d.]+)s\] (.+)", line)
        if m:
            segments.append({
                "speaker": m.group(1),
                "start": float(m.group(2)),
                "text": m.group(3),
            })
    return segments


def run(audio_path: Path, transcript: str) -> str:
    """Analiza emociones por segmento de hablante. Retorna texto enriquecido."""
    t0 = time.time()
    y, sr = librosa.load(str(audio_path), sr=16000)
    segments = parse_transcript_segments(transcript)

    if not segments:
        return "No hay segmentos para analizar."

    # Asignar tiempos de fin a cada segmento
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
        # Max 15s recomendado
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
