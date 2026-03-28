"""Subagente — Prosodia y features acústicos por segmento de hablante"""
import re
import time
import librosa
import numpy as np
import opensmile
from pathlib import Path


def parse_transcript_segments(transcript: str) -> list:
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
    t0 = time.time()
    y, sr = librosa.load(str(audio_path))
    segments = parse_transcript_segments(transcript)

    if not segments:
        return "No hay segmentos para analizar."

    # Asignar tiempos de fin
    duration = len(y) / sr
    for i, seg in enumerate(segments):
        seg["end"] = segments[i + 1]["start"] if i + 1 < len(segments) else duration

    # opensmile global (features del archivo completo para comparación)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    global_features = smile.process_file(str(audio_path))

    lines = ["# Análisis de Prosodia y Acústica\n"]

    # Features globales
    cols = {
        "F0semitoneFrom27.5Hz_sma3nz_amean": "pitch",
        "loudness_sma3_amean": "volumen",
        "jitterLocal_sma3nz_amean": "jitter",
        "shimmerLocaldB_sma3nz_amean": "shimmer",
        "HNRdBACF_sma3nz_amean": "HNR",
    }
    lines.append("## Features globales (archivo completo)\n")
    for col, name in cols.items():
        if col in global_features.columns:
            lines.append(f"- {name}: {global_features[col].iloc[0]:.3f}")

    # Por segmento de hablante
    lines.append("\n## Por segmento de hablante\n")
    try:
        import parselmouth
        from parselmouth.praat import call
        use_parselmouth = True
    except ImportError:
        use_parselmouth = False

    for seg in segments:
        start_s = int(seg["start"] * sr)
        end_s = int(seg["end"] * sr)
        chunk = y[start_s:end_s]
        if len(chunk) < sr // 2:
            continue

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        onsets = librosa.onset.onset_detect(y=chunk, sr=sr)
        seg_dur = max(len(chunk) / sr, 0.01)
        speech_rate = len(onsets) / seg_dur

        mean_pitch = 0
        if use_parselmouth and len(chunk) > sr // 4:
            try:
                snd = parselmouth.Sound(chunk, sampling_frequency=float(sr))
                pitch_obj = call(snd, "To Pitch", 0, 75, 600)
                pv = pitch_obj.selected_array["frequency"]
                pv = pv[pv > 0]
                mean_pitch = float(np.mean(pv)) if len(pv) > 0 else 0
            except Exception:
                pass

        lines.append(
            f"[{seg['speaker']} {seg['start']:.1f}s] {seg['text'][:50]}\n"
            f"  → RMS={rms:.3f} | velocidad={speech_rate:.1f}syl/s | pitch={mean_pitch:.0f}Hz\n"
        )

    elapsed = time.time() - t0
    lines.append(f"\n_Tiempo procesamiento: {elapsed:.1f}s_")
    return "\n".join(lines)
