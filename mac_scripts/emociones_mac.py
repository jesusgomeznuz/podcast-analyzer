"""Mac M4 emotion analysis script — runs on Mac via SSH dispatch.
Usage: python3 emociones_mac.py <audio_path> <transcript_path> [hf_token]
"""
import sys
import os
import re
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/vox-profile-release")

import torch
import librosa
import numpy as np

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

audio_path = sys.argv[1]
transcript_path = sys.argv[2]
hf_token = sys.argv[3] if len(sys.argv) > 3 else ""

transcript = open(transcript_path, encoding="utf-8").read()


def parse_transcript_segments(transcript):
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


def emotion2vec(y, sr, segments):
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


def wavlm_dimensional(y, sr, segments):
    from src.model.emotion.wavlm_emotion_dim import WavLMWrapper
    model = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-msp-podcast-emotion-dim")
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
            results.append(f"A={arousal.item():.2f} V={valence.item():.2f} D={dominance.item():.2f}")
        except Exception:
            results.append("error")
    return results


t0 = time.time()
y, sr = librosa.load(audio_path, sr=16000)
segments = parse_transcript_segments(transcript)

if not segments:
    print("No hay segmentos para analizar.")
    sys.exit(0)

for i, seg in enumerate(segments):
    if i + 1 < len(segments):
        seg["end"] = segments[i + 1]["start"]
    else:
        seg["end"] = len(y) / sr

results_e2v = emotion2vec(y, sr, segments)
results_wlm = wavlm_dimensional(y, sr, segments)

elapsed = time.time() - t0
print(f"# Análisis Emocional ({elapsed:.1f}s)\n")
print("## Por segmento de hablante\n")
for i, seg in enumerate(segments):
    e2v = results_e2v[i] if i < len(results_e2v) else "—"
    wlm = results_wlm[i] if i < len(results_wlm) else "—"
    print(f"[{seg['speaker']} {seg['start']:.1f}s] {seg['text'][:60]}")
    print(f"  → emoción: {e2v} | arousal/valence/dominance: {wlm}")
    print()
