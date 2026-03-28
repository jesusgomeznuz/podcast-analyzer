"""Subagente 5 — Reportero: consolida todo y notifica por Telegram"""
import os
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path


def run(url: str, audio_name: str, transcript: str, analysis: str,
        veredicto: str, report_dir: Path):
    (report_dir / "transcript.txt").write_text(transcript, encoding="utf-8")

    estado = f"""# Estado — {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Corrida completada ✅

| Campo | Valor |
|-------|-------|
| URL | {url} |
| Audio | {audio_name} |
| Transcript | {len(transcript):,} chars |
| Análisis | {len(analysis):,} chars |
| Veredicto | {len(veredicto):,} chars |

## Archivos
- `transcript.txt` — transcripción completa
- `analysis.md` — análisis con Claude
- `veredicto.md` — veredicto final

## Veredicto (preview)

{veredicto[:500]}

## Análisis (preview)

{analysis[:400]}
"""
    (report_dir / "estado.md").write_text(estado, encoding="utf-8")
    _notify_telegram(veredicto, analysis, report_dir)


def _notify_telegram(veredicto: str, analysis: str, report_dir: Path):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("   ⚠️  Sin credenciales Telegram, saltando notificación.")
        return

    # Extrae el resumen en una línea del veredicto si existe
    resumen = ""
    for line in veredicto.splitlines():
        if "redes" in line.lower() or "resumen" in line.lower():
            resumen = line.strip("- #*").strip()
            break

    msg = f"""🎙️ Podcast Analyzer — Análisis completado

📁 Reporte: {report_dir}

🏆 VEREDICTO:
{veredicto[:600]}

📊 ANÁLISIS (preview):
{analysis[:400]}

---
Reporte completo en estado.md"""

    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": msg[:4000]
    }).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage", data=data
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        print("   → Notificación enviada por Telegram")
    except Exception as e:
        print(f"   ⚠️  Telegram falló: {e}")
