"""Subagente — Dinámica conversacional: quién habla cuánto, interrupciones, dominio"""
import re
import time
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


def run(transcript: str, audio_duration: float = None) -> str:
    t0 = time.time()
    segments = parse_transcript_segments(transcript)

    if not segments:
        return "No hay segmentos para analizar."

    # Asignar tiempos de fin
    for i, seg in enumerate(segments):
        seg["end"] = segments[i + 1]["start"] if i + 1 < len(segments) else (
            audio_duration or segments[-1]["start"] + 5
        )
        seg["duration"] = seg["end"] - seg["start"]

    speakers = {}
    prev_speaker = None
    interruptions = 0

    for seg in segments:
        spk = seg["speaker"]
        speakers.setdefault(spk, {"words": 0, "turns": 0, "duration": 0.0, "segments": []})
        speakers[spk]["words"] += len(seg["text"].split())
        speakers[spk]["duration"] += seg["duration"]
        speakers[spk]["segments"].append(seg)
        if prev_speaker and prev_speaker != spk:
            speakers[spk]["turns"] += 1
            # Interrupción: turno muy corto del hablante previo (<2s)
            if prev_speaker in speakers and speakers[prev_speaker]["segments"]:
                last_dur = speakers[prev_speaker]["segments"][-1]["duration"]
                if last_dur < 2.0:
                    interruptions += 1
        prev_speaker = spk

    total_words = sum(s["words"] for s in speakers.values())
    total_duration = sum(s["duration"] for s in speakers.values())

    elapsed = time.time() - t0
    lines = ["# Dinámica Conversacional\n"]

    for spk, data in sorted(speakers.items()):
        word_pct = data["words"] / total_words * 100 if total_words else 0
        time_pct = data["duration"] / total_duration * 100 if total_duration else 0
        avg_turn = data["duration"] / max(data["turns"], 1)
        lines.append(
            f"**{spk}**: {data['words']} palabras ({word_pct:.1f}%) | "
            f"{data['duration']:.1f}s hablando ({time_pct:.1f}%) | "
            f"{data['turns']} turnos | turno promedio {avg_turn:.1f}s"
        )

    lines.append(f"\n**Interrupciones estimadas**: {interruptions}")
    lines.append(f"**Hablantes**: {len(speakers)}")
    lines.append(f"**Total segmentos**: {len(segments)}")

    # Determinar quién domina
    if speakers:
        dominant = max(speakers.items(), key=lambda x: x[1]["words"])
        lines.append(f"\n**Hablante dominante**: {dominant[0]} ({dominant[1]['words']} palabras, {dominant[1]['words']/total_words*100:.1f}%)")

    lines.append(f"\n_Tiempo procesamiento: {elapsed:.3f}s_")
    return "\n".join(lines)
