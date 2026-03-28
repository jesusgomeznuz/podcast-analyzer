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
    _copy_to_vault(report_dir)
    _notify_telegram(veredicto, analysis, report_dir)


def _copy_to_vault(report_dir: Path):
    """Copia el reporte como nota Markdown al Vault de Obsidian."""
    vault_dir = Path("/home/chuy/Documents/Obsidian Vault/Podcast Analyzer/Reportes")
    vault_dir.mkdir(parents=True, exist_ok=True)

    date_part = report_dir.parent.name  # e.g. 2026-03-28
    video_id = report_dir.name          # e.g. DQUdGB_CdQ7
    output = vault_dir / f"{date_part}_{video_id}.md"

    lines = [f"# Reporte: {video_id} ({date_part})\n"]
    for section in ["transcript.txt", "dinamica.md", "emociones.md", "prosodia.md",
                    "facial.md", "analysis.md", "veredicto.md", "estado.md"]:
        f = report_dir / section
        if f.exists():
            lines += ["\n---\n", f"## {section}\n\n", f.read_text(encoding="utf-8")]

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"   → Vault: {output}")


def _notify_telegram(veredicto: str, analysis: str, report_dir: Path):
    token = os.environ.get("AGENT_TELEGRAM_TOKEN")
    chat_id = os.environ.get("AGENT_TELEGRAM_CHAT_ID")
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
