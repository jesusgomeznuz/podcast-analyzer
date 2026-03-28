#!/usr/bin/env python3
"""Orquestador — Podcast Analyzer Fase 2"""
# IMPORTANTE: load_dotenv() debe correr ANTES de los imports de agentes
# para que os.environ.get("HUGGINGFACE_TOKEN") funcione en tiempo de módulo
from dotenv import load_dotenv
load_dotenv()

import argparse
import librosa
from datetime import datetime
from pathlib import Path

import agent_descargador
import agent_transcriptor
import agent_emociones
import agent_prosodia
import agent_dinamica
import agent_facial
import agent_analizador
import agent_veredicto
import agent_reportero
import agent_visualizador
import agent_html_report

TMP_DIR = Path(__file__).parent / "tmp"
REPORTS_DIR = Path(__file__).parent / "reports"


def main():
    parser = argparse.ArgumentParser(description="Podcast Analyzer")
    parser.add_argument("url", help="URL del video (YouTube, Instagram, TikTok)")
    parser.add_argument("--no-facial", action="store_true", help="Saltar análisis facial")
    args = parser.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")

    print("─" * 55)
    print(f"🎙️  Podcast Analyzer — {datetime.now().strftime('%H:%M')}")
    print(f"📺  URL: {args.url}")
    print("─" * 55)

    print("\n[1/8] 🔽 Descargando audio...")
    audio_path = agent_descargador.run(args.url, TMP_DIR)

    # Usar video ID como subdirectorio para evitar colisiones entre videos
    video_id = audio_path.stem  # e.g. "DQUdGB_CdQ7"
    report_dir = REPORTS_DIR / today / video_id
    report_dir.mkdir(parents=True, exist_ok=True)
    size_mb = audio_path.stat().st_size / 1024 / 1024
    print(f"      → {audio_path.name} ({size_mb:.1f} MB)")

    print("\n[2/8] 🎙️  Transcribiendo con diarización (WhisperX)...")
    transcript = agent_transcriptor.run(audio_path)
    (report_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    print(f"      → {len(transcript):,} chars | {report_dir / 'transcript.txt'}")

    # Duración del audio para dinámica conversacional
    audio_duration = librosa.get_duration(path=str(audio_path))

    print("\n[3/8] 💬 Analizando dinámica conversacional...")
    dinamica = agent_dinamica.run(transcript, audio_duration)
    (report_dir / "dinamica.md").write_text(dinamica, encoding="utf-8")
    print(f"      → dinamica.md")

    print("\n[4/8] 😤 Analizando emociones (emotion2vec + WavLM)...")
    emociones = agent_emociones.run(audio_path, transcript)
    (report_dir / "emociones.md").write_text(emociones, encoding="utf-8")
    print(f"      → emociones.md")

    print("\n[5/8] 🎛️  Analizando prosodia y acústica...")
    prosodia = agent_prosodia.run(audio_path, transcript)
    (report_dir / "prosodia.md").write_text(prosodia, encoding="utf-8")
    print(f"      → prosodia.md")

    print("\n[6/8] 👁️  Análisis facial (DeepFace)...")
    if args.no_facial:
        print("      → (omitido con --no-facial)")
        facial = "Análisis facial omitido."
    else:
        try:
            video_path = agent_descargador.run_video(args.url, TMP_DIR)
            print(f"      → Video: {video_path.name}")
            facial = agent_facial.run(video_path, transcript)
        except Exception as e:
            facial = f"Análisis facial no disponible: {e}"
            print(f"      → Error: {e}")
    (report_dir / "facial.md").write_text(facial, encoding="utf-8")
    print(f"      → facial.md")

    print("\n[7/8] 🤖 Analizando contenido + veredicto (Claude)...")
    contexto_completo = f"{transcript}\n\n---\n{dinamica}\n\n---\n{prosodia}"
    analysis = agent_analizador.run(contexto_completo, report_dir / "analysis.md")
    veredicto = agent_veredicto.run(analysis, report_dir / "veredicto.md")
    print(f"      → analysis.md + veredicto.md")

    print("\n[8/9] 📋 Reporte final...")
    agent_reportero.run(args.url, audio_path.name, transcript,
                        analysis, veredicto, report_dir)
    print(f"      → estado.md + Telegram")

    print("\n[9/9] 📊 Generando gráficas y reporte HTML...")
    charts = agent_visualizador.run(report_dir)
    print(f"      → {len(charts)} gráficas generadas")
    html_path = agent_html_report.run(args.url, report_dir, charts)
    print(f"      → {html_path}")

    print("\n" + "─" * 55)
    print(f"✅ Completado — {datetime.now().strftime('%H:%M')}")
    print(f"📁 {report_dir}/")
    print(f"🌐 {html_path}")
    print("─" * 55)


if __name__ == "__main__":
    main()
