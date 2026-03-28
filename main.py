#!/usr/bin/env python3
"""Orquestador — coordina los 5 subagentes del Podcast Analyzer"""
import argparse
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import agent_descargador
import agent_transcriptor
import agent_analizador
import agent_veredicto
import agent_reportero

load_dotenv()

TMP_DIR = Path(__file__).parent / "tmp"
REPORTS_DIR = Path(__file__).parent / "reports"


def main():
    parser = argparse.ArgumentParser(description="Podcast Analyzer")
    parser.add_argument("url", help="URL de YouTube del podcast/debate")
    args = parser.parse_args()

    openai_key = os.environ["OPENAI_API_KEY"]
    today = datetime.now().strftime("%Y-%m-%d")
    report_dir = REPORTS_DIR / today
    report_dir.mkdir(parents=True, exist_ok=True)

    print("─" * 50)
    print(f"🎙️  Podcast Analyzer — {datetime.now().strftime('%H:%M')}")
    print(f"📺  URL: {args.url}")
    print("─" * 50)

    print("\n[1/5] 🔽 Subagente Descargador...")
    audio_path = agent_descargador.run(args.url, TMP_DIR)
    print(f"      → {audio_path.name} ({audio_path.stat().st_size / 1024 / 1024:.1f} MB)")

    print("\n[2/5] 🎙️  Subagente Transcriptor...")
    transcript = agent_transcriptor.run(audio_path, openai_key)
    print(f"      → {len(transcript):,} caracteres")

    print("\n[3/5] 🤖 Subagente Analizador...")
    analysis = agent_analizador.run(transcript, report_dir / "analysis.md")
    print(f"      → análisis listo ({len(analysis):,} chars)")

    print("\n[4/5] 🏆 Subagente Veredicto...")
    veredicto = agent_veredicto.run(analysis, report_dir / "veredicto.md")
    print(f"      → veredicto listo ({len(veredicto):,} chars)")

    print("\n[5/5] 📋 Subagente Reportero...")
    agent_reportero.run(args.url, audio_path.name, transcript,
                        analysis, veredicto, report_dir)
    print(f"      → estado.md generado + notificación Telegram")

    print("\n" + "─" * 50)
    print(f"✅ Completado — {datetime.now().strftime('%H:%M')}")
    print(f"📁 {report_dir}/")
    print("─" * 50)


if __name__ == "__main__":
    main()
