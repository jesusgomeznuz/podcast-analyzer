from datetime import datetime
from pathlib import Path


def write_report(url: str, transcript: str, analysis: str, audio_name: str, report_dir: Path):
    (report_dir / "transcript.txt").write_text(transcript, encoding="utf-8")

    estado = f"""# Estado — {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Corrida completada ✅

| Campo | Valor |
|-------|-------|
| URL | {url} |
| Audio | {audio_name} |
| Transcript | {len(transcript):,} caracteres |
| Análisis | {len(analysis):,} caracteres |

## Archivos generados

- `transcript.txt` — transcripción completa (Whisper)
- `analysis.md` — análisis (Claude Code)

## Análisis (preview)

{analysis[:800]}
"""
    (report_dir / "estado.md").write_text(estado, encoding="utf-8")
