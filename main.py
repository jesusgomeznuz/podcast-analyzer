#!/usr/bin/env python3
"""Orquestador — coordina los subagentes del Podcast Analyzer (Fase 2)"""
import argparse
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import agent_descargador
import agent_transcriptor
import agent_emociones
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

    print("─" * 50)
    print(f"🎙️  Podcast Analyzer — {datetime.now().strftime('%H:%M')}")
    print(f"📺  URL: {args.url}")
    print("─" * 50)

    print("\n[1/6] 🔽 Descargando audio...")
    audio_path = agent_descargador.run(args.url, TMP_DIR)
    print(f"      → {audio_path.name} ({audio_path.stat().st_size / 1024 / 1024:.1f} MB)")

    print("\n[2/6] 🎙️  Transcribiendo con diarización (WhisperX)...")
    transcript = agent_transcriptor.run(audio_path)
    print(f"      → {len(transcript):,} caracteres")
    (report_dir / "transcript.txt").write_text(transcript, encoding="utf-8")

    print("\n[3/6] 😤 Analizando emociones (emotion2vec + WavLM)...")
    emotion_report = agent_emociones.run(audio_path, transcript)
    (report_dir / "emociones.md").write_text(emotion_report, encoding="utf-8")
    print(f"      → emociones.md generado")

    print("\n[4/6] 🤖 Analizando contenido...")
    analysis = agent_analizador.run(transcript, report_dir / "analysis.md")
    print(f"      → analysis.md ({len(analysis):,} chars)")

    print("\n[5/6] 🏆 Generando veredicto...")
    veredicto = agent_veredicto.run(analysis, report_dir / "veredicto.md")
    print(f"      → veredicto.md ({len(veredicto):,} chars)")

    print("\n[6/6] 📋 Generando reporte final...")
    agent_reportero.run(args.url, audio_path.name, transcript,
                        analysis, veredicto, report_dir)
    print(f"      → estado.md + notificación Telegram")

    print("\n" + "─" * 50)
    print(f"✅ Completado — {datetime.now().strftime('%H:%M')}")
    print(f"📁 {report_dir}/")
    print("─" * 50)


if __name__ == "__main__":
    main()
