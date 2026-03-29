"""Subagente — Análisis facial por hablante usando DeepFace + timestamps de WhisperX
Estrategia: Mac M4 via SSH → fallback ThinkPad CPU
"""
import os
import re
import time
import subprocess
import tempfile
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path

MAC_HOST = "jesus@192.168.1.131"
MAC_PY = "/Users/jesus/miniconda3/envs/podcast-analyzer/bin/python3"
MAC_SCRIPTS_DIR = Path(__file__).parent / "mac_scripts"


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


def _run_on_mac(video_path: Path, transcript: str) -> str:
    remote_video = f"/tmp/{video_path.name}"
    remote_transcript = f"/tmp/{video_path.stem}_facial_transcript.txt"
    remote_script = "/tmp/facial_mac.py"

    print(f"   → Subiendo video a Mac ({video_path.stat().st_size // 1024 // 1024}MB)...")
    subprocess.run(["scp", str(video_path), f"{MAC_HOST}:{remote_video}"], check=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(transcript)
        local_transcript = f.name
    subprocess.run(["scp", local_transcript, f"{MAC_HOST}:{remote_transcript}"], check=True)
    os.unlink(local_transcript)

    subprocess.run(["scp", str(MAC_SCRIPTS_DIR / "facial_mac.py"), f"{MAC_HOST}:{remote_script}"], check=True)

    cmd = f'export PATH="/opt/homebrew/bin:/usr/bin:/bin:$PATH" && {MAC_PY} {remote_script} {remote_video} {remote_transcript}'
    result = subprocess.run(["ssh", MAC_HOST, cmd], capture_output=True, text=True, timeout=7200)

    subprocess.run(["ssh", MAC_HOST, f"rm -f {remote_video} {remote_transcript} {remote_script}"], capture_output=True)

    if result.returncode != 0:
        raise RuntimeError(f"Mac facial falló: {result.stderr[-500:]}")

    output = result.stdout.strip()
    if not output:
        raise RuntimeError("Mac retornó output vacío")
    return output


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


def extract_frame_at(video_path: Path, timestamp: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_num = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def analyze_frame(frame: np.ndarray) -> dict | None:
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


def run(video_path: Path, transcript: str) -> str:
    if _mac_available():
        try:
            print("   → Mac M4 detectada, procesando análisis facial en Mac")
            return _run_on_mac(video_path, transcript)
        except Exception as e:
            print(f"   → Mac falló ({e}), usando CPU local")

    print("   → Procesando análisis facial en CPU local (ThinkPad)")
    return _run_local(video_path, transcript)


def _run_local(video_path: Path, transcript: str) -> str:
    t0 = time.time()
    segments = parse_transcript_segments(transcript)

    if not segments:
        return "No hay segmentos para analizar."

    if not video_path.exists():
        return f"Video no encontrado: {video_path}"

    cap = cv2.VideoCapture(str(video_path))
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
    return "\n".join(lines)
