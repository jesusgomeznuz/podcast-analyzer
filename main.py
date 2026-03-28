#!/usr/bin/env python3
import argparse
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from downloader import download_audio
from transcriber import transcribe
from analyzer import analyze
from reporter import write_report

load_dotenv()

TMP_DIR = Path(__file__).parent / "tmp"
REPORTS_DIR = Path(__file__).parent / "reports"


def main():
    parser = argparse.ArgumentParser(description="Podcast Analyzer")
    parser.add_argument("url", help="URL de YouTube del podcast")
    args = parser.parse_args()

    openai_key = os.environ["OPENAI_API_KEY"]
    today = datetime.now().strftime("%Y-%m-%d")
    report_dir = REPORTS_DIR / today
    report_dir.mkdir(parents=True, exist_ok=True)

    print("⬇️  Descargando audio...")
    audio_path = download_audio(args.url, TMP_DIR)
    print(f"   → {audio_path.name}")

    print("🎙️  Transcribiendo con Whisper...")
    transcript = transcribe(audio_path, openai_key)
    print(f"   → {len(transcript):,} caracteres")

    print("🤖 Analizando con Claude Code...")
    analysis = analyze(transcript, report_dir / "analysis.md")
    print("   → análisis listo")

    write_report(args.url, transcript, analysis, audio_path.name, report_dir)
    print(f"\n✅ Reporte en {report_dir}/")
    print(f"   estado.md · transcript.txt · analysis.md")


if __name__ == "__main__":
    main()
