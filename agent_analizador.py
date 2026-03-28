"""Subagente 3 — Analizador: transcript → análisis estructurado (Claude Code CLI)"""
import subprocess
from pathlib import Path

PROMPT = """Analiza el siguiente transcript de debate/podcast y produce un reporte en español con:

1. **Participantes** identificados (nombres si se pueden inferir)
2. **Temas principales** discutidos (lista)
3. **Momentos clave** con posición aproximada
4. **Tono y emociones** por participante (frustración, confianza, nerviosismo, etc.)
5. **Interrupciones y dinámicas** de conversación notables
6. **Argumentos más fuertes** de cada parte

Sé conciso y usa markdown. Máximo 800 palabras.

---
TRANSCRIPT:

{transcript}
"""


def run(transcript: str, output_path: Path) -> str:
    prompt = PROMPT.format(transcript=transcript[:60000])
    result = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "text"],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"Claude Code (analizador) falló:\n{result.stderr}")
    analysis = result.stdout.strip()
    output_path.write_text(analysis, encoding="utf-8")
    return analysis
