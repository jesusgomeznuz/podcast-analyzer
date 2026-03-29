"""Subagente — Prosodia y features acústicos por segmento de hablante
Estrategia: Mac M4 via SSH → fallback ThinkPad CPU
"""
import os
import re
import time
import subprocess
import tempfile
import librosa
import numpy as np
import opensmile
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


def _run_on_mac(audio_path: Path, transcript: str) -> str:
    remote_audio = f"/tmp/{audio_path.name}"
    remote_transcript = f"/tmp/{audio_path.stem}_prosodia_transcript.txt"
    remote_script = "/tmp/prosodia_mac.py"

    subprocess.run(["scp", str(audio_path), f"{MAC_HOST}:{remote_audio}"], check=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(transcript)
        local_transcript = f.name
    subprocess.run(["scp", local_transcript, f"{MAC_HOST}:{remote_transcript}"], check=True)
    os.unlink(local_transcript)

    subprocess.run(["scp", str(MAC_SCRIPTS_DIR / "prosodia_mac.py"), f"{MAC_HOST}:{remote_script}"], check=True)

    cmd = f'export PATH="/opt/homebrew/bin:/usr/bin:/bin:$PATH" && {MAC_PY} {remote_script} {remote_audio} {remote_transcript}'
    result = subprocess.run(["ssh", MAC_HOST, cmd], capture_output=True, text=True, timeout=3600)

    subprocess.run(["ssh", MAC_HOST, f"rm -f {remote_audio} {remote_transcript} {remote_script}"], capture_output=True)

    if result.returncode != 0:
        raise RuntimeError(f"Mac prosodia falló: {result.stderr[-500:]}")

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


def run(audio_path: Path, transcript: str) -> str:
    if _mac_available():
        try:
            print("   → Mac M4 detectada, procesando prosodia en Mac")
            return _run_on_mac(audio_path, transcript)
        except Exception as e:
            print(f"   → Mac falló ({e}), usando CPU local")

    print("   → Procesando prosodia en CPU local (ThinkPad)")
    return _run_local(audio_path, transcript)


def _run_local(audio_path: Path, transcript: str) -> str:
    t0 = time.time()
    y, sr = librosa.load(str(audio_path))
    segments = parse_transcript_segments(transcript)

    if not segments:
        return "No hay segmentos para analizar."

    duration = len(y) / sr
    for i, seg in enumerate(segments):
        seg["end"] = segments[i + 1]["start"] if i + 1 < len(segments) else duration

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    global_features = smile.process_file(str(audio_path))

    lines = ["# Análisis de Prosodia y Acústica\n"]

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
