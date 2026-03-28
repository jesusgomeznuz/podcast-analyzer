# Subagente de análisis — lanza Claude Code CLI como subprocess
# No requiere ANTHROPIC_API_KEY, usa la auth de Claude Code
import subprocess
from pathlib import Path


PROMPT_TEMPLATE = """Analiza el siguiente transcript de podcast/debate y produce un reporte en español con:

1. **Resumen ejecutivo** (3-5 oraciones)
2. **Temas principales** (lista)
3. **Momentos clave** (con indicación aproximada de posición si se puede inferir)
4. **Tono y emociones** detectadas
5. **Citas destacadas** (máximo 3)

Escribe el reporte en formato markdown. Sé conciso y útil.

---
TRANSCRIPT:

{transcript}
"""


def analyze(transcript: str, output_path: Path) -> str:
    prompt = PROMPT_TEMPLATE.format(transcript=transcript[:50000])  # límite seguro
    result = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "text"],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"Claude Code falló:\n{result.stderr}")
    analysis = result.stdout.strip()
    output_path.write_text(analysis, encoding="utf-8")
    return analysis
