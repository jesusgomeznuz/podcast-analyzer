#!/usr/bin/env python3
"""Orquestador — Podcast Analyzer Fase 2"""
import argparse
import librosa
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import agent_descargador
import agent_transcriptor
import agent_emociones
import agent_prosodia
import agent_dinamica
import agent_analizador
import agent_veredicto
import agent_reportero

load_dotenv()

TMP_DIR = Path(__file__).parent / "tmp"
REPORTS_DIR = Path(__file__).parent / "reports"


def main():
    parser = argparse.ArgumentParser(description="Podcast Analyzer")
    parser.add_argument("url", help="URL del video (YouTube, Instagram, TikTok)")
    args = parser.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    report_dir = REPORTS_DIR / today
    report_dir.mkdir(parents=True, exist_ok=True)

    print("─" * 55)
    print(f"🎙️  Podcast Analyzer — {datetime.now().strftime('%H:%M')}")
    print(f"📺  URL: {args.url}")
    print("─" * 55)

    print("\n[1/7] 🔽 Descargando audio...")
    audio_path = agent_descargador.run(args.url, TMP_DIR)
    size_mb = audio_path.stat().st_size / 1024 / 1024
    print(f"      → {audio_path.name} ({size_mb:.1f} MB)")

    print("\n[2/7] 🎙️  Transcribiendo con diarización (WhisperX)...")
    transcript = agent_transcriptor.run(audio_path)
    (report_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    print(f"      → {len(transcript):,} chars | {report_dir / 'transcript.txt'}")

    # Duración del audio para dinámica conversacional
    audio_duration = librosa.get_duration(path=str(audio_path))

    print("\n[3/7] 💬 Analizando dinámica conversacional...")
    dinamica = agent_dinamica.run(transcript, audio_duration)
    (report_dir / "dinamica.md").write_text(dinamica, encoding="utf-8")
    print(f"      → dinamica.md")

    print("\n[4/7] 😤 Analizando emociones (emotion2vec + WavLM)...")
    emociones = agent_emociones.run(audio_path, transcript)
    (report_dir / "emociones.md").write_text(emociones, encoding="utf-8")
    print(f"      → emociones.md")

    print("\n[5/7] 🎛️  Analizando prosodia y acústica...")
    prosodia = agent_prosodia.run(audio_path, transcript)
    (report_dir / "prosodia.md").write_text(prosodia, encoding="utf-8")
    print(f"      → prosodia.md")

    print("\n[6/7] 🤖 Analizando contenido + veredicto (Claude)...")
    # Combinar todos los análisis para el analizador
    contexto_completo = f"{transcript}\n\n---\n{dinamica}\n\n---\n{prosodia}"
    analysis = agent_analizador.run(contexto_completo, report_dir / "analysis.md")
    veredicto = agent_veredicto.run(analysis, report_dir / "veredicto.md")
    print(f"      → analysis.md + veredicto.md")

    print("\n[7/7] 📋 Reporte final...")
    agent_reportero.run(args.url, audio_path.name, transcript,
                        analysis, veredicto, report_dir)
    print(f"      → estado.md + Telegram")

    print("\n" + "─" * 55)
    print(f"✅ Completado — {datetime.now().strftime('%H:%M')}")
    print(f"📁 {report_dir}/")
    print("─" * 55)


if __name__ == "__main__":
    main()
