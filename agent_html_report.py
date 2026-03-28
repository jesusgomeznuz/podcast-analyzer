"""Subagente — Reporte HTML: consolida todo en un documento navegable"""
import base64
import re
from datetime import datetime
from pathlib import Path


def _img_b64(path: Path) -> str:
    """Convierte imagen PNG a data URI base64."""
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def _md_to_html(text: str) -> str:
    """Conversión muy básica de Markdown a HTML."""
    lines = []
    in_table = False
    for line in text.split("\n"):
        if line.startswith("### "):
            lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("# "):
            lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("> "):
            lines.append(f'<blockquote>{line[2:]}</blockquote>')
        elif line.startswith("|"):
            if not in_table:
                lines.append('<table>')
                in_table = True
            if re.match(r"\|[-| :]+\|", line):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            row = "".join(f"<td>{c}</td>" for c in cells)
            lines.append(f"<tr>{row}</tr>")
        else:
            if in_table:
                lines.append("</table>")
                in_table = False
            html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
            html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
            html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)
            if html.strip():
                lines.append(f"<p>{html}</p>")
    if in_table:
        lines.append("</table>")
    return "\n".join(lines)


def run(url: str, report_dir: Path, charts: list) -> Path:
    video_id = report_dir.name
    date = report_dir.parent.name

    # Leer contenidos
    def read(name):
        p = report_dir / name
        return p.read_text(encoding="utf-8") if p.exists() else ""

    transcript = read("transcript.txt")
    dinamica = read("dinamica.md")
    emociones = read("emociones.md")
    prosodia = read("prosodia.md")
    facial = read("facial.md")
    analysis = read("analysis.md")
    veredicto = read("veredicto.md")

    # Insertar gráficas
    chart_html = ""
    for chart_path in charts:
        title = chart_path.stem.replace("grafica_", "").replace("_", " ").title()
        img = _img_b64(chart_path)
        chart_html += f"""
        <div class="chart-card">
            <h3>{title}</h3>
            <img src="{img}" alt="{title}" loading="lazy">
        </div>"""

    # Transcript con colores por hablante
    speakers = sorted(set(re.findall(r"\[(\S+) [\d.]+s\]", transcript)))
    COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
    spk_colors = {spk: COLORS[i % len(COLORS)] for i, spk in enumerate(speakers)}

    transcript_html = ""
    for line in transcript.strip().split("\n"):
        m = re.match(r"\[(\S+) ([\d.]+)s\] (.+)", line)
        if m:
            spk, ts, txt = m.group(1), m.group(2), m.group(3)
            color = spk_colors.get(spk, "#555")
            transcript_html += f"""
            <div class="seg">
                <span class="spk" style="color:{color}">{spk}</span>
                <span class="ts">{ts}s</span>
                <span class="txt">{txt}</span>
            </div>"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Podcast Analyzer — {video_id}</title>
<style>
  :root {{
    --bg: #0f0f13; --card: #1a1a24; --accent: #6c63ff;
    --text: #e2e2ea; --sub: #8888a0; --border: #2e2e40;
    --blue: #2196F3; --red: #F44336; --green: #4CAF50;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Segoe UI", system-ui, sans-serif; background: var(--bg);
         color: var(--text); line-height: 1.6; }}
  header {{ background: linear-gradient(135deg, #1a1a30, #2d1b69);
            padding: 2rem; border-bottom: 1px solid var(--border); }}
  header h1 {{ font-size: 1.6rem; color: #a89dff; }}
  header .meta {{ color: var(--sub); font-size: 0.9rem; margin-top: 0.4rem; }}
  header a {{ color: var(--accent); text-decoration: none; word-break: break-all; }}
  nav {{ display: flex; gap: 0.5rem; padding: 1rem 2rem;
         background: var(--card); border-bottom: 1px solid var(--border);
         flex-wrap: wrap; }}
  nav a {{ color: var(--accent); text-decoration: none; padding: 0.3rem 0.8rem;
           border-radius: 20px; border: 1px solid var(--accent); font-size: 0.85rem; }}
  nav a:hover {{ background: var(--accent); color: white; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
  section {{ margin-bottom: 3rem; }}
  section h2 {{ color: #a89dff; font-size: 1.3rem; margin-bottom: 1rem;
                padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); }}
  .card {{ background: var(--card); border: 1px solid var(--border);
           border-radius: 10px; padding: 1.5rem; margin-bottom: 1rem; }}
  .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
             gap: 1.5rem; }}
  .chart-card {{ background: var(--card); border: 1px solid var(--border);
                  border-radius: 10px; padding: 1rem; }}
  .chart-card h3 {{ color: var(--sub); font-size: 0.9rem; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 0.8rem; }}
  .chart-card img {{ width: 100%; border-radius: 6px; }}
  .seg {{ display: grid; grid-template-columns: 120px 60px 1fr;
          gap: 0.5rem; padding: 0.3rem 0; border-bottom: 1px solid var(--border);
          font-size: 0.88rem; align-items: start; }}
  .spk {{ font-weight: 600; font-size: 0.8rem; }}
  .ts {{ color: var(--sub); font-family: monospace; }}
  .txt {{ color: var(--text); }}
  p {{ margin-bottom: 0.7rem; color: var(--text); }}
  h1,h2,h3 {{ color: #a89dff; margin: 1rem 0 0.5rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; margin: 1rem 0; }}
  td,th {{ padding: 0.5rem 0.8rem; border: 1px solid var(--border); }}
  tr:nth-child(even) {{ background: rgba(255,255,255,0.03); }}
  blockquote {{ border-left: 3px solid var(--accent); padding: 0.5rem 1rem;
                background: rgba(108,99,255,0.08); border-radius: 0 6px 6px 0;
                font-style: italic; margin: 1rem 0; color: #c4befc; }}
  code {{ background: rgba(255,255,255,0.08); padding: 0.1rem 0.3rem; border-radius: 3px; }}
  strong {{ color: #c4befc; }}
  .veredicto {{ font-size: 1rem; }}
  footer {{ text-align: center; color: var(--sub); padding: 2rem; font-size: 0.8rem;
            border-top: 1px solid var(--border); }}
</style>
</head>
<body>

<header>
  <h1>🎙️ Podcast Analyzer</h1>
  <div class="meta">
    <strong>{video_id}</strong> · {date} ·
    <a href="{url}" target="_blank">{url}</a>
  </div>
</header>

<nav>
  <a href="#graficas">📊 Gráficas</a>
  <a href="#veredicto">🏆 Veredicto</a>
  <a href="#analisis">🤖 Análisis</a>
  <a href="#dinamica">💬 Dinámica</a>
  <a href="#emociones">😤 Emociones</a>
  <a href="#prosodia">🎛️ Prosodia</a>
  <a href="#facial">👁️ Facial</a>
  <a href="#transcript">📝 Transcript</a>
</nav>

<div class="container">

  <section id="graficas">
    <h2>📊 Visualizaciones</h2>
    <div class="charts">
      {chart_html if chart_html else '<p style="color:var(--sub)">No se generaron gráficas.</p>'}
    </div>
  </section>

  <section id="veredicto">
    <h2>🏆 Veredicto</h2>
    <div class="card veredicto">
      {_md_to_html(veredicto)}
    </div>
  </section>

  <section id="analisis">
    <h2>🤖 Análisis</h2>
    <div class="card">
      {_md_to_html(analysis)}
    </div>
  </section>

  <section id="dinamica">
    <h2>💬 Dinámica Conversacional</h2>
    <div class="card">
      {_md_to_html(dinamica)}
    </div>
  </section>

  <section id="emociones">
    <h2>😤 Emociones (audio)</h2>
    <div class="card">
      {_md_to_html(emociones[:3000])}
      {'<p style="color:var(--sub)">... (truncado)</p>' if len(emociones) > 3000 else ''}
    </div>
  </section>

  <section id="prosodia">
    <h2>🎛️ Prosodia y Acústica</h2>
    <div class="card">
      {_md_to_html(prosodia[:3000])}
      {'<p style="color:var(--sub)">... (truncado)</p>' if len(prosodia) > 3000 else ''}
    </div>
  </section>

  <section id="facial">
    <h2>👁️ Análisis Facial</h2>
    <div class="card">
      {_md_to_html(facial)}
    </div>
  </section>

  <section id="transcript">
    <h2>📝 Transcripción</h2>
    <div class="card">
      {transcript_html}
    </div>
  </section>

</div>

<footer>
  Generado por Podcast Analyzer · {datetime.now().strftime("%Y-%m-%d %H:%M")}
</footer>

</body>
</html>"""

    out = report_dir / "reporte.html"
    out.write_text(html, encoding="utf-8")
    return out
