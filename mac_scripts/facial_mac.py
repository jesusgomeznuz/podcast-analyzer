"""Mac M4 facial analysis script — runs on Mac via SSH dispatch.
Usage: python3 facial_mac.py <video_path> <transcript_path>
"""
import sys
import re
import time
import warnings

warnings.filterwarnings("ignore")
os_import = __import__("os")

import cv2
import numpy as np
from collections import defaultdict

video_path = sys.argv[1]
transcript_path = sys.argv[2]
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


def extract_frame_at(video_path, timestamp):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_num = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def analyze_frame(frame):
    try:
        from deepface import DeepFace
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            silent=True,
        )
        if result:
            return result[0]["emotion"]
    except Exception:
        pass
    return None


t0 = time.time()
segments = parse_transcript_segments(transcript)

if not segments:
    print("No hay segmentos para analizar.")
    sys.exit(0)

cap = cv2.VideoCapture(video_path)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
video_duration = total_frames / fps
cap.release()

for i, seg in enumerate(segments):
    seg["end"] = segments[i + 1]["start"] if i + 1 < len(segments) else video_duration

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
speaker_emotions = defaultdict(lambda: defaultdict(list))
speaker_segments = defaultdict(int)

for seg in segments:
    mid = (seg["start"] + seg["end"]) / 2
    frame = extract_frame_at(video_path, mid)
    if frame is None:
        continue
    emotions = analyze_frame(frame)
    if emotions is None:
        continue
    for e in EMOTIONS:
        speaker_emotions[seg["speaker"]][e].append(emotions.get(e, 0))
    speaker_segments[seg["speaker"]] += 1

elapsed = time.time() - t0
lines = ["# Análisis Facial por Hablante\n"]

if not speaker_emotions:
    lines.append("No se detectaron rostros en los segmentos.")
else:
    for spk in sorted(speaker_emotions.keys()):
        data = speaker_emotions[spk]
        n = speaker_segments[spk]
        lines.append(f"## {spk} ({n} segmentos analizados)\n")
        avgs = {e: np.mean(data[e]) for e in EMOTIONS if data[e]}
        if avgs:
            dominant = max(avgs, key=avgs.get)
            lines.append(f"**Emoción dominante**: {dominant} ({avgs[dominant]:.1f}%)\n")
            for e in sorted(avgs, key=avgs.get, reverse=True):
                bar = "█" * int(avgs[e] / 5)
                lines.append(f"- {e:10s}: {avgs[e]:5.1f}% {bar}")
        lines.append("")

lines.append(f"\n_Tiempo procesamiento: {elapsed:.1f}s_")
print("\n".join(lines))
