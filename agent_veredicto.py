"""Subagente 4 — Veredicto: análisis → quién ganó el debate (Claude Code CLI)"""
import subprocess
from pathlib import Path

PROMPT = """Basándote en el siguiente análisis de debate, emite un veredicto objetivo:

1. **Veredicto**: ¿Quién ganó el debate y por qué? (3-4 oraciones directas)
2. **Métricas estimadas**:
   - Tiempo de habla relativo (%)
   - Argumentos sólidos vs. ataques personales (proporción estimada)
   - Momentos de ventaja clara por participante
3. **Punto de quiebre**: el momento decisivo que inclinó la balanza
4. **Resumen en una línea** apto para publicar en redes sociales

Sé directo y objetivo. No más de 300 palabras.

---
ANÁLISIS:

{analysis}
"""


def run(analysis: str, output_path: Path) -> str:
    prompt = PROMPT.format(analysis=analysis)
    result = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "text"],
        capture_output=True, text=True, timeout=180
    )
    if result.returncode != 0:
        raise RuntimeError(f"Claude Code (veredicto) falló:\n{result.stderr}")
    veredicto = result.stdout.strip()
    output_path.write_text(veredicto, encoding="utf-8")
    return veredicto
