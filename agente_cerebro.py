#!/usr/bin/env python3
"""
agente_cerebro.py — Alimenta el cerebro por creador en Obsidian.

Uso:
    python3 agente_cerebro.py <youtube_url> <nombre_creador>

Ejemplo:
    python3 agente_cerebro.py https://youtu.be/2HD_DWKhLcs "Roberto Mtz"

Flujo:
    1. yt-dlp: descarga audio MP3 + heatmap
    2. Mac M4: transcripción con timestamps (mlx-whisper)
    3. youtube-comment-downloader: top 100 comentarios
    4. Claude: analiza todo y extrae rasgos, temas, fandom, invitado
    5. Obsidian: actualiza Perfil del creador + Perfil del fandom + crea nota de episodio
"""

import sys
import json
import os
import subprocess
import re
import time
import unicodedata
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('/home/chuy/agent/.env')

# ─── Config ─────────────────────────────────────────────────────────────────

MAC_USER = "jesus"
MAC_HOST = "192.168.1.131"
MAC_PYTHON = "/Users/jesus/miniconda3/envs/podcast-analyzer/bin/python3"
MAC_PATH = "PATH=/opt/homebrew/bin:/usr/bin:/bin"

VAULT = Path("/home/chuy/Documents/Obsidian Vault")
CREATORS_DIR = VAULT / "Podcast Analyzer" / "Creadores de contenido"
CEREBRO_CACHE = Path("/home/chuy/podcast-analyzer/cerebro_cache")
TRANSCRIBE_SCRIPT = Path("/home/chuy/podcast-analyzer/mac_scripts/transcribe_plain_mac.py")
CLAUDE_USAGE_FILE = Path("/home/chuy/agent/claude_usage.json")
TELEGRAM_TOKEN = os.getenv("AGENT_TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("AGENT_TELEGRAM_CHAT_ID", "5253002519")

# ─── Helpers ─────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[cerebro] {msg}", flush=True)


def run_claude(prompt: str, timeout: int = 120) -> str:
    result = subprocess.run(
        ["claude", "-p", "--dangerously-skip-permissions"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    output = result.stdout.strip()
    # Strip markdown code blocks if present
    if output.startswith("```"):
        lines = output.split("\n")
        output = "\n".join(lines[1:-1]) if len(lines) > 2 else output
    return output


# ─── Step 1: Download audio + heatmap ────────────────────────────────────────

def download_audio_and_heatmap(url: str, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_path = cache_dir / "audio.mp3"
    meta_path = cache_dir / "meta.json"

    # Use cached meta if available (avoids yt-dlp when YouTube rate-limits)
    if meta_path.exists() and audio_path.exists():
        with open(meta_path) as f:
            cached = json.load(f)
        if cached.get("uploader"):
            log("Meta + audio ya en caché.")
            return cached
        log("Cache sin uploader — re-fetching metadata...")

    # Fetch metadata — try with cookies if plain request fails
    log("Obteniendo metadata del video...")
    def run_ytdlp_meta(extra_args=None):
        cmd = ["yt-dlp", "--print-json", "--skip-download"] + (extra_args or []) + [url]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        return r.stdout.strip()

    raw = run_ytdlp_meta()
    if not raw:
        log("YouTube bloqueó sin cookies — intentando con Firefox...")
        raw = run_ytdlp_meta(["--cookies-from-browser", "firefox"])
    if not raw:
        raise RuntimeError("yt-dlp no pudo obtener metadata (bot detection). Exporta cookies manualmente.")

    yt_meta = json.loads(raw)
    video_id = yt_meta["id"]
    title = yt_meta["title"]
    views = yt_meta.get("view_count", 0)
    duration_s = yt_meta.get("duration", 0)
    uploader = yt_meta.get("uploader") or yt_meta.get("channel", "")
    upload_date_raw = yt_meta.get("upload_date", "")
    fecha_video = (f"{upload_date_raw[:4]}-{upload_date_raw[4:6]}-{upload_date_raw[6:]}"
                   if len(upload_date_raw) == 8 else "")

    # Extract heatmap peaks
    heatmap_raw = yt_meta.get("heatmap", [])
    peaks = sorted(heatmap_raw, key=lambda x: x.get("value", 0), reverse=True)[:8]
    heatmap = []
    for p in peaks:
        start = p.get("start_time", 0)
        val = p.get("value", 0)
        m = int(start // 60)
        s = int(start % 60)
        heatmap.append({"time": f"{m}m{s:02d}s", "pct": round(val * 100), "start_s": start})

    with open(cache_dir / "heatmap.json", "w") as f:
        json.dump(heatmap, f, ensure_ascii=False, indent=2)

    # Download audio
    if not audio_path.exists():
        log("Descargando audio...")
        dl_cmd = ["yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "5",
                  "-o", str(audio_path), url]
        dl_result = subprocess.run(dl_cmd, capture_output=True, timeout=600)
        if dl_result.returncode != 0:
            log("Descarga falló sin cookies — reintentando con Firefox...")
            dl_cmd_cookies = dl_cmd[:-1] + ["--cookies-from-browser", "firefox", url]
            subprocess.run(dl_cmd_cookies, capture_output=True, timeout=600)
    else:
        log("Audio ya en caché.")

    result_meta = {
        "video_id": video_id,
        "title": title,
        "uploader": uploader,
        "views": views,
        "duration_min": duration_s // 60,
        "url": url,
        "fecha_video": fecha_video,
        "heatmap": heatmap,
        "audio_path": str(audio_path),
    }

    # Save meta to cache
    with open(meta_path, "w") as f:
        json.dump(result_meta, f, ensure_ascii=False, indent=2)

    log(f"Video: {title} | {views:,} vistas | {duration_s // 60:.0f} min")
    return result_meta


# ─── Step 2: Transcribe on Mac M4 ────────────────────────────────────────────

def transcribe_on_mac(audio_path: str, video_id: str, cache_dir: Path, duration_min: int = 0) -> dict:
    transcript_path = cache_dir / "transcript.json"
    if transcript_path.exists() and transcript_path.stat().st_size > 1000:
        log("Transcript ya en caché.")
        with open(transcript_path) as f:
            return json.load(f)

    # Kill any stale transcription on Mac before starting
    subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", f"{MAC_USER}@{MAC_HOST}",
         "pkill -f transcribe_plain_mac 2>/dev/null; sleep 1; true"],
        capture_output=True, timeout=15
    )

    log("Enviando audio a Mac M4 para transcripción...")
    remote_audio = f"/tmp/audio_{video_id}.mp3"
    remote_script = "/tmp/transcribe_plain_mac.py"

    subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no",
         str(TRANSCRIBE_SCRIPT), f"{MAC_USER}@{MAC_HOST}:{remote_script}"],
        capture_output=True, timeout=30
    )
    scp_result = subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=no",
         audio_path, f"{MAC_USER}@{MAC_HOST}:{remote_audio}"],
        capture_output=True, timeout=300
    )
    if scp_result.returncode != 0:
        raise RuntimeError(f"SCP failed: {scp_result.stderr.decode()[:200]}")

    # Dynamic timeout: at least 10 min, or 20s per minute of audio
    ssh_timeout = max(600, duration_min * 20)
    log(f"Transcribiendo... (timeout: {ssh_timeout // 60} min para {duration_min} min de audio)")
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no",
         f"{MAC_USER}@{MAC_HOST}",
         f"{MAC_PATH} {MAC_PYTHON} {remote_script} {remote_audio}"],
        capture_output=True, text=True, timeout=ssh_timeout
    )

    if not result.stdout.strip():
        stderr_hint = result.stderr.strip()[:200] if result.stderr else "no stderr"
        raise RuntimeError(f"Mac M4 devolvió output vacío. stderr: {stderr_hint}")

    transcript = json.loads(result.stdout)
    with open(transcript_path, "w") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    log(f"Transcripción: {len(transcript['segments'])} segmentos, {transcript['duration_s'] / 60:.1f} min")

    # Cleanup Mac
    subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", f"{MAC_USER}@{MAC_HOST}",
         f"rm -f {remote_audio}"],
        capture_output=True, timeout=15
    )

    return transcript


# ─── Step 3: Get comments ─────────────────────────────────────────────────────

def get_comments(url: str, cache_dir: Path) -> list:
    comments_path = cache_dir / "comments.json"
    if comments_path.exists():
        log("Comentarios ya en caché.")
        with open(comments_path) as f:
            return json.load(f)

    log("Descargando comentarios...")
    from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
    d = YoutubeCommentDownloader()
    comments = []
    for c in d.get_comments_from_url(url, sort_by=SORT_BY_POPULAR):
        comments.append({"text": c["text"], "votes": c.get("votes", 0)})
        if len(comments) >= 100:
            break

    with open(comments_path, "w") as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)

    log(f"Comentarios: {len(comments)} descargados")
    return comments


# ─── Step 4: Cross-reference heatmap with transcript ─────────────────────────

def cross_reference_heatmap(heatmap: list, segments: list) -> list:
    def find_text_at(target_s, window=30):
        matches = [s for s in segments
                   if s["start"] >= target_s - window and s["start"] <= target_s + window]
        if not matches:
            matches = sorted(segments, key=lambda s: abs(s["start"] - target_s))[:4]
        return " ".join(s["text"].strip() for s in sorted(matches, key=lambda s: s["start"]))

    result = []
    for h in heatmap:
        text = find_text_at(h["start_s"])
        result.append(f"[{h['time']} — {h['pct']}%] {text[:200]}")
    return result


# ─── Step 5: Claude analysis ──────────────────────────────────────────────────

def _read_existing_nodes(directory: Path) -> list[dict]:
    """Lee títulos y slugs de notas existentes en una carpeta."""
    nodes = []
    if not directory.exists():
        return nodes
    for md in sorted(directory.glob("*.md")):
        slug = md.stem
        name = slug
        try:
            for line in md.read_text(encoding="utf-8").splitlines():
                if line.startswith("# "):
                    name = line[2:].strip()
                    break
        except Exception:
            pass
        nodes.append({"slug": slug, "name": name})
    return nodes


def analyze_with_claude(meta: dict, transcript: dict, comments: list, heatmap_context: list, creator_name: str, creator_dir: Path = None) -> dict:
    analysis_path = Path(meta["audio_path"]).parent / "analysis.json"
    if analysis_path.exists():
        log("Análisis ya en caché.")
        with open(analysis_path) as f:
            return json.load(f)

    log("Analizando con Claude...")
    top_comments = "\n".join(f"[{c['votes']}] {c['text']}" for c in comments[:20])
    peaks_text = "\n".join(heatmap_context[:6])
    full_text = transcript.get("full_text", "")[:40000]

    # Load existing nodes so Claude can reuse slugs instead of creating duplicates
    existing_rasgos_lines = ""
    existing_temas_lines = ""
    existing_temas_slugs = []
    existing_gags_lines = ""
    existing_fandom_lines = ""
    existing_personas_lines = ""
    existing_datos_lines = ""
    if creator_dir:
        existing_rasgos = _read_existing_nodes(creator_dir / "Rasgos")
        if existing_rasgos:
            existing_rasgos_lines = "\n".join(f'  - slug: "{r["slug"]}" → {r["name"]}' for r in existing_rasgos)
        existing_temas = _read_existing_nodes(creator_dir / "Temas")
        if existing_temas:
            existing_temas_slugs = [t["slug"] for t in existing_temas]
            existing_temas_lines = "\n".join(f'  - slug: "{t["slug"]}" → {t["name"]}' for t in existing_temas)
        existing_gags = _read_existing_nodes(creator_dir / "RunningGags")
        if existing_gags:
            existing_gags_lines = "\n".join(f'  - slug: "{g["slug"]}" → {g["name"]}' for g in existing_gags)
        existing_fandom = _read_existing_nodes(creator_dir / "Fandom")
        if existing_fandom:
            existing_fandom_lines = "\n".join(f'  - slug: "{f["slug"]}" → {f["name"]}' for f in existing_fandom)
        existing_personas = _read_existing_nodes(creator_dir / "Personas")
        if existing_personas:
            existing_personas_lines = "\n".join(f'  - slug: "{p["slug"]}" → {p["name"]}' for p in existing_personas)
        existing_datos = _read_existing_nodes(creator_dir / "DatosPersonales" / creator_name)
        if existing_datos:
            existing_datos_lines = "\n".join(f'  - slug: "{d["slug"]}" → {d["name"]}' for d in existing_datos)

    existing_context = ""
    if existing_rasgos_lines:
        existing_context += f"""
## Rasgos ya documentados de {creator_name}
{existing_rasgos_lines}
→ Si observas el mismo rasgo, devuelve el slug EXACTO de la lista en el campo "slug". Solo crea uno nuevo si no hay match semántico claro.

"""
    if existing_temas_lines:
        existing_context += f"""## Temas ya documentados
{existing_temas_lines}
→ Si el tema nuevo coincide semánticamente con uno existente, devuelve su slug EXACTO. Solo crea uno nuevo si no hay match claro.
→ Para temas_relacionados: usa ÚNICAMENTE slugs de esta lista. No inventes slugs.

"""
    if existing_gags_lines:
        existing_context += f"""## Running gags ya documentados
{existing_gags_lines}
→ Si detectas el mismo gag, devuelve el slug EXACTO. Solo crea uno nuevo si es distinto.

"""
    if existing_fandom_lines:
        existing_context += f"""## Comportamientos de fandom ya documentados
{existing_fandom_lines}
→ Si detectas el mismo comportamiento, devuelve el slug EXACTO. Solo crea uno nuevo si es distinto.

"""
    if existing_personas_lines:
        existing_context += f"""## Personas ya documentadas
{existing_personas_lines}
→ Si la persona ya existe, devuelve el slug EXACTO. Solo crea uno nuevo si es una persona nueva.

"""
    if existing_datos_lines:
        existing_context += f"""## Datos personales ya documentados de {creator_name}
{existing_datos_lines}
→ Si el dato ya existe, devuelve el slug EXACTO. Solo crea uno nuevo si es un dato genuinamente nuevo.

"""

    prompt = f"""Eres un analista de cerebros por creador de contenido. Extraes información verificable para alimentar el perfil de {creator_name} en Obsidian.
{existing_context}

FORMATO DE SALIDA: Responde SOLO con JSON válido. Las cadenas de texto que incluyas en los campos "fuente_exacta", "observacion", "descripcion" y "dato" deben seguir convenciones de Obsidian Markdown cuando sea útil — usa [[wikilinks]] SOLO para referenciar personas (ej: [[Personas/franco-escamilla|Franco Escamilla]]) o temas (ej: [[Temas/proceso-creativo|proceso creativo]]) que ya existan como nodos. NUNCA uses el nombre del podcast como wikilink (no escribas [[Creativo]] ni [[CREATIVO]] — el podcast no es un nodo del grafo). Escribe en español neutro sin markdown adicional (sin **, sin #, sin listas).

REGLA CRÍTICA — temas en dos niveles:
- tema_especifico: lo particular de este episodio ("payaso como terapia transformadora según Franco Escamilla")
- tema_general: el concepto abstracto y reutilizable en 2-4 palabras ("arte-como-terapia")
- El slug SIEMPRE deriva del tema_general. Nunca del específico.
- Si tema_general es demasiado específico para repetirse, generaliza más.
- temas_relacionados: SOLO slugs que aparecen literalmente en la lista "Temas ya documentados" de arriba. Si no hay lista o ninguno aplica, usa [].

REGLA CRÍTICA — deduplicación:
- Si el rasgo, tema, gag, dato, persona o comportamiento de fandom ya existe en las listas de arriba, devuelve el slug EXACTO tal como aparece.
- Nunca generes un slug nuevo para algo que ya existe con otro slug.

REGLA CRÍTICA — anti-poisoning:
- Solo incluye hechos con evidencia directa (cita textual o "MM:SS — descripción del momento")
- Si no tienes fuente exacta, NO lo incluyas
- No parafrasees ni interpretes — documenta lo que ocurrió
- hosts_identificados: identifica quiénes son los hosts del canal (pueden ser 1 o más). Usa el contexto del video, el título, o tu conocimiento general del canal
- Para datos_personales: SOLO datos de los hosts (no del invitado). Solo lo que cada host dice en primera persona sobre sí mismo. Si alguien más lo dice sobre él, NO lo incluyas. Cada dato debe tener el nombre exacto del host en "sujeto"

## Video
Canal: {meta.get('uploader', '')}
Título: {meta['title']}
Vistas: {meta['views']:,}
Duración: {meta['duration_min']} min
URL: {meta['url']}

## Transcript
{full_text}

## Momentos de alto impacto (heatmap most-replayed)
{peaks_text}

## Top comentarios de la audiencia
{top_comments}

---

Responde SOLO con JSON:

{{
  "rasgos_creador": [
    {{
      "slug": "slug-exacto-si-ya-existe-O-nuevo-en-kebab-case-sin-acentos",
      "rasgo": "descripción concisa del rasgo de {creator_name}",
      "fuente_exacta": "cita textual entre comillas O 'MM:SS — descripción del momento'"
    }}
  ],
  "temas_recurrentes": [
    {{
      "tema_especifico": "descripción exacta y particular de CÓMO se trató el tema en ESTE episodio — pegada al lenguaje del invitado",
      "tema_general": "concepto abstracto y reutilizable en 2-4 palabras — lo suficientemente amplio para que OTROS episodios puedan referenciarlo",
      "slug": "tema-general-en-kebab-case-sin-acentos — derivado de tema_general, NO de tema_especifico",
      "campo_semantico": "territorio amplio al que pertenece este tema — 2-3 palabras en español, ej: 'Proceso creativo', 'Negocio creativo', 'Salud y vida', 'Tecnología', 'Identidad y cultura', 'Arte y expresión', 'Escritura', 'Sistemas personales' — usa uno existente si aplica, crea uno nuevo solo si ninguno encaja",
      "fuente_exacta": "cita textual entre comillas O 'MM:SS — descripción del momento'"
    }}
  ],
  "datos_fandom": [
    {{
      "slug": "comportamiento-slug",
      "observacion": "qué hace o dice la audiencia",
      "fuente_exacta": "texto exacto del comentario con votos, ej: '[813] texto completo del comentario'"
    }}
  ],
  "running_gags": [
    {{
      "slug": "nombre-slug",
      "descripcion": "qué es el running gag o catchphrase",
      "fuente_exacta": "cita textual o momento exacto donde aparece"
    }}
  ],
  "personas_frecuentes": [
    {{
      "nombre": "nombre completo",
      "slug": "nombre-slug",
      "relacion": "cómo se relaciona con {creator_name} en este episodio"
    }}
  ],
  "hosts_identificados": ["nombre del host 1", "nombre del host 2 si aplica"],
  "datos_personales": [
    {{
      "sujeto": "nombre del HOST que compartió el dato (no el invitado)",
      "sujeto_slug": "nombre-slug",
      "dato": "descripción concisa del dato personal que el host compartió sobre SÍ MISMO",
      "slug": "dato-slug",
      "fuente_exacta": "cita textual O 'MM:SS — descripción del momento'",
      "nota_temporal": "si el dato implica un cambio en el tiempo, explicar (vacío si no aplica)"
    }}
  ],
  "invitado": {{
    "nombre": "nombre del invitado",
    "slug": "nombre-slug",
    "profesion": "profesión breve",
    "datos_clave": ["hecho verificable — fuente exacta"]
  }},
  "episodio_titulo": "CREATIVO #N — Nombre Invitado",
  "resumen_episodio": "1 oración: quién, qué tema central, tono"
}}

Solo JSON. Sin texto antes ni después.
"""

    # Dynamic timeout: 3 min base + 1 min per 10 min of audio, max 20 min
    analysis_timeout = min(1200, 180 + meta.get("duration_min", 60) * 6)
    output = run_claude(prompt, timeout=analysis_timeout)

    try:
        analysis = json.loads(output)
    except json.JSONDecodeError:
        # Try to extract JSON from output
        match = re.search(r'\{.*\}', output, re.DOTALL)
        if match:
            analysis = json.loads(match.group())
        else:
            raise ValueError(f"Claude no devolvió JSON válido: {output[:200]}")

    with open(analysis_path, "w") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    return analysis


# ─── Step 6: Update Obsidian notes ───────────────────────────────────────────

def _set_campo_semantico(nota_path: Path, campo: str):
    """Add or update campo_semantico in a tema note's frontmatter."""
    if not nota_path.exists() or not campo:
        return
    content = nota_path.read_text(encoding="utf-8")
    if f"campo_semantico: {campo}" in content:
        return
    if "campo_semantico:" in content:
        content = re.sub(r"campo_semantico:.*\n", f"campo_semantico: {campo}\n", content)
    else:
        content = content.replace("---\ntipo: tema\n---", f"---\ntipo: tema\ncampo_semantico: {campo}\n---", 1)
    nota_path.write_text(content, encoding="utf-8")


def _generate_campos_hub(creator_dir: Path, temas_dir: Path) -> list[str]:
    """Group temas by campo_semantico. Create/update Campos/ hub notes. Return list of campo slugs."""
    campos: dict[str, list[tuple[str, str]]] = {}
    for md in sorted(temas_dir.glob("*.md")):
        slug = md.stem
        content = md.read_text(encoding="utf-8")
        campo_match = re.search(r"campo_semantico:\s*(.+)", content)
        if not campo_match:
            continue
        campo = campo_match.group(1).strip()
        name = slug
        for line in content.splitlines():
            if line.startswith("# "):
                name = line[2:].strip()
                break
        campos.setdefault(campo, []).append((slug, name))

    if not campos:
        return []

    campos_dir = creator_dir / "Campos"
    campos_dir.mkdir(exist_ok=True)
    campo_slugs = []
    for campo, temas in sorted(campos.items()):
        campo_slug = slugify(campo)
        campo_path = campos_dir / f"{campo_slug}.md"
        temas_links = "\n".join(f"- [[Temas/{s}|{n}]]" for s, n in sorted(temas))
        campo_path.write_text(f"""---
tipo: campo-semantico
nombre: {campo}
---

# {campo}

## Temas
{temas_links}
""", encoding="utf-8")
        campo_slugs.append((campo_slug, campo))
    return campo_slugs


def _upsert_hub_node(node_path: Path, node_name: str, node_type: str, episode_link: str, fuente_exacta: str, fecha_video: str = ""):
    """Create or update a hub node (Tema, Fandom, RunningGag, Persona) with a new episode link."""
    date_prefix = f"({fecha_video}) " if fecha_video else ""
    new_entry = f"- {episode_link} — {date_prefix}{fuente_exacta}"
    if node_path.exists():
        existing = node_path.read_text(encoding="utf-8")
        if episode_link in existing:
            return  # already linked, skip
        # Insert new entry inside "## Episodios donde aparece" block,
        # before any subsequent ## section (e.g. ## Temas relacionados)
        lines = existing.rstrip().split("\n")
        insert_idx = len(lines)  # default: end of file
        in_episodios = False
        for i, line in enumerate(lines):
            if line.strip().startswith("## Episodios donde aparece"):
                in_episodios = True
                continue
            if in_episodios and line.startswith("## "):
                insert_idx = i
                break
        lines.insert(insert_idx, new_entry)
        node_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        node_path.write_text(f"""---
tipo: {node_type}
---

# {node_name}

## Episodios donde aparece
{new_entry}
""", encoding="utf-8")


def update_obsidian(creator_name: str, meta: dict, analysis: dict):
    from datetime import date

    creator_dir = CREATORS_DIR / creator_name
    temas_dir = creator_dir / "Temas"
    fandom_dir = creator_dir / "Fandom"
    gags_dir = creator_dir / "RunningGags"
    personas_dir = creator_dir / "Personas"
    episodios_dir = creator_dir / "Episodios"
    rasgos_dir = creator_dir / "Rasgos"

    for d in [creator_dir, temas_dir, fandom_dir, gags_dir, personas_dir, episodios_dir, rasgos_dir]:
        d.mkdir(parents=True, exist_ok=True)

    fecha_video = meta.get("fecha_video", "")

    episode_title = analysis.get("episodio_titulo", meta["title"])
    episode_link = f"[[Episodios/{episode_title}]]"
    today = date.today().isoformat()

    log(f"Creando nota de episodio: {episode_title}")

    # ── 1. Episode note ──────────────────────────────────────────────────────
    heatmap_rows = "\n".join(
        f"| {h['time']} | {h['pct']}% |"
        for h in meta["heatmap"]
        if h["pct"] >= 40
    ) or "| — | — |"

    invitado = analysis.get("invitado", {})
    datos_clave = "\n".join(f"- {d}" for d in invitado.get("datos_clave", []))

    temas_links = " · ".join(
        f"[[Temas/{t.get('slug', slugify(t.get('tema_general', t.get('tema',''))))}|{t.get('tema_general', t.get('tema',''))}]]"
        for t in analysis.get("temas_recurrentes", [])
    )
    gags_links = " · ".join(
        f"[[RunningGags/{g['slug']}|{g['descripcion'][:30]}]]"
        for g in analysis.get("running_gags", [])
    )

    episode_note = f"""---
tipo: episodio
podcast: {creator_name}
invitado: {invitado.get('nombre', '')}
url: {meta['url']}
vistas: {meta['views']}
duracion_min: {meta['duration_min']}
fecha_video: {fecha_video}
fecha_analisis: {today}
tags:
  - episodio
---

# {episode_title}

> {analysis.get('resumen_episodio', '')}

Host: [[Perfil del creador]] | Invitado: [[Personas/{invitado.get('slug', slugify(invitado.get('nombre', 'invitado')))}|{invitado.get('nombre', '')}]]

**Temas:** {temas_links}
{"**Running gags:** " + gags_links if gags_links else ""}

---

## Momentos de alto impacto (heatmap most-replayed)

| Tiempo | Rewatch % |
|---|---|
{heatmap_rows}

---

## Datos del invitado

{datos_clave}
"""
    (episodios_dir / f"{episode_title}.md").write_text(episode_note, encoding="utf-8")

    # ── 2. Temas/ hub nodes ──────────────────────────────────────────────────
    log("Actualizando nodos de Temas/...")
    tema_slugs = []
    for tema in analysis.get("temas_recurrentes", []):
        # Support both old schema (tema) and new schema (tema_general + tema_especifico)
        tema_general = tema.get("tema_general") or tema.get("tema", "")
        tema_especifico = tema.get("tema_especifico") or tema_general
        slug = tema.get("slug") or slugify(tema_general)
        if not slug:
            continue
        campo = tema.get("campo_semantico", "")
        tema_slugs.append(f"[[Temas/{slug}|{tema_general}]]")
        # Entry shows the specific framing + source
        fuente = tema.get("fuente_exacta", "—")
        entry_detail = f"{tema_especifico} — {fuente}" if tema_especifico != tema_general else fuente
        _upsert_hub_node(
            node_path=temas_dir / f"{slug}.md",
            node_name=tema_general,
            node_type="tema",
            episode_link=episode_link,
            fuente_exacta=entry_detail,
            fecha_video=fecha_video,
        )
        # Set campo_semantico in frontmatter
        if campo:
            _set_campo_semantico(temas_dir / f"{slug}.md", campo)
        # Note: temas_relacionados cross-links intentionally removed.
        # Cross-connections between temas emerge organically when multiple episodes
        # reference the same temas — not from Claude-generated synthetic links.

    # ── 3. Fandom/ hub nodes ─────────────────────────────────────────────────
    log("Actualizando nodos de Fandom/...")
    for dato in analysis.get("datos_fandom", []):
        slug = dato.get("slug") or slugify(dato.get("observacion", "fandom")[:40])
        if not slug:
            continue
        _upsert_hub_node(
            node_path=fandom_dir / f"{slug}.md",
            node_name=dato["observacion"],
            node_type="comportamiento-fandom",
            episode_link=episode_link,
            fuente_exacta=dato.get("fuente_exacta", "—"),
            fecha_video=fecha_video,
        )

    # ── 4. RunningGags/ hub nodes ────────────────────────────────────────────
    log("Actualizando RunningGags/...")
    for gag in analysis.get("running_gags", []):
        slug = gag.get("slug") or slugify(gag.get("descripcion", "gag")[:40])
        if not slug:
            continue
        _upsert_hub_node(
            node_path=gags_dir / f"{slug}.md",
            node_name=gag["descripcion"],
            node_type="running-gag",
            episode_link=episode_link,
            fuente_exacta=gag.get("fuente_exacta", "—"),
            fecha_video=fecha_video,
        )

    # ── 5. DatosPersonales/ hub nodes (subcarpeta por host) ─────────────────
    log("Actualizando DatosPersonales/...")
    datos_base_dir = creator_dir / "DatosPersonales"
    datos_base_dir.mkdir(exist_ok=True)
    for dato in analysis.get("datos_personales", []):
        slug = dato.get("slug") or slugify(dato.get("dato", "dato")[:50])
        if not slug:
            continue
        sujeto = dato.get("sujeto", creator_name)
        sujeto_slug = dato.get("sujeto_slug") or slugify(sujeto)
        nota_temporal = dato.get("nota_temporal", "")
        fuente = dato.get("fuente_exacta", "—")
        if nota_temporal:
            fuente = f"{fuente} | {nota_temporal}"
        # One subfolder per host
        host_dir = datos_base_dir / sujeto
        host_dir.mkdir(exist_ok=True)
        node_path = host_dir / f"{slug}.md"
        date_prefix = f"({fecha_video}) " if fecha_video else ""
        new_entry = f"- {episode_link} — {date_prefix}{fuente}"
        if node_path.exists():
            existing = node_path.read_text(encoding="utf-8")
            if episode_link not in existing:
                node_path.write_text(existing.rstrip() + "\n" + new_entry + "\n", encoding="utf-8")
        else:
            node_path.write_text(f"""---
tipo: dato-personal
sujeto: {sujeto}
---

# {dato['dato']}

## Mencionado en
{new_entry}
""", encoding="utf-8")

    # ── 6. Personas/ hub nodes ───────────────────────────────────────────────
    log("Actualizando Personas/...")
    for persona in analysis.get("personas_frecuentes", []):
        slug = persona.get("slug") or slugify(persona.get("nombre", "persona"))
        if not slug:
            continue
        _upsert_hub_node(
            node_path=personas_dir / f"{slug}.md",
            node_name=persona["nombre"],
            node_type="persona",
            episode_link=episode_link,
            fuente_exacta=persona.get("relacion", "—"),
            fecha_video=fecha_video,
        )
    # Also add invitado as persona
    if invitado.get("nombre"):
        inv_slug = invitado.get("slug") or slugify(invitado["nombre"])
        _upsert_hub_node(
            node_path=personas_dir / f"{inv_slug}.md",
            node_name=invitado["nombre"],
            node_type="persona-invitado",
            episode_link=episode_link,
            fuente_exacta=invitado.get("profesion", "—"),
            fecha_video=fecha_video,
        )

    # ── 6. Rasgos/ hub nodes (rasgos observados del creador) ────────────────
    log("Actualizando Rasgos/...")
    for rasgo in analysis.get("rasgos_creador", []):
        rasgo_texto = rasgo.get("rasgo", "")
        if not rasgo_texto:
            continue
        slug = rasgo.get("slug") or slugify(rasgo_texto[:50])
        fuente = rasgo.get("fuente_exacta", "—")
        _upsert_hub_node(
            node_path=rasgos_dir / f"{slug}.md",
            node_name=rasgo_texto,
            node_type="rasgo-creador",
            episode_link=episode_link,
            fuente_exacta=fuente,
            fecha_video=fecha_video,
        )

    # ── 7. Campos semánticos hub notes ──────────────────────────────────────────
    log("Generando Campos semánticos...")
    campo_slugs = _generate_campos_hub(creator_dir, temas_dir)

    # ── 8. Perfil del creador — minimal hub: only episodes + campos ──────────────
    def _node_links(directory: Path, prefix: str) -> str:
        nodes = _read_existing_nodes(directory)
        if not nodes:
            return f"_Sin notas aún en {prefix}/_"
        return "\n".join(f"- [[{prefix}/{n['slug']}|{n['name']}]]" for n in nodes)

    perfil_path = creator_dir / "Perfil del creador.md"
    log("Actualizando Perfil del creador...")

    episodios_links = _node_links(episodios_dir, "Episodios")
    if campo_slugs:
        campos_links = "\n".join(f"- [[Campos/{cs}|{cn}]]" for cs, cn in campo_slugs)
    else:
        campos_links = _node_links(temas_dir, "Temas")  # fallback if no campos yet

    perfil_path.write_text(f"""---
tipo: perfil-creador
creador: {creator_name}
tags:
  - creador
---

# {creator_name} — Perfil del creador

> Punto de entrada al cerebro. Navega por campos semánticos o entra directo a un episodio.
> Los nodos con más conexiones en el grafo = temas nucleares del creador.

## Episodios analizados
{episodios_links}

## Campos semánticos
{campos_links}
""", encoding="utf-8")

    # ── 8. Perfil del fandom — always regenerate with real links ────────────────
    fandom_perfil_path = creator_dir / "Perfil del fandom.md"
    log("Actualizando Perfil del fandom...")
    fandom_perfil_path.write_text(f"""---
tipo: perfil-fandom
creador: {creator_name}
tags:
  - fandom
---

# {creator_name} — Perfil del fandom

## Comportamientos de la audiencia
{_node_links(fandom_dir, "Fandom")}
""", encoding="utf-8")

    log(f"Obsidian actualizado: {creator_dir}")
    return episode_title


# ─── Queue management ────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s-]', '', text.lower())
    text = re.sub(r'[-\s]+', '-', text).strip('-')
    return text or "sin-titulo"


def creator_slug(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def queue_path(creator_name: str) -> Path:
    return CEREBRO_CACHE / creator_slug(creator_name) / "queue.json"


def load_queue(creator_name: str) -> list:
    p = queue_path(creator_name)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


def save_queue(creator_name: str, queue: list):
    p = queue_path(creator_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(queue, f, ensure_ascii=False, indent=2)


def get_usage() -> dict:
    """Lee uso real de Claude Code desde claude_usage.json (escrito por el status line).

    Retorna dict con:
      session_pct: % usado del límite de 5 horas (None si no disponible)
      weekly_pct:  % usado del límite semanal de 7 días (None si no disponible)
      session_resets_at: epoch cuando resetea la sesión (None si no disponible)
      weekly_resets_at:  epoch cuando resetea el weekly (None si no disponible)
      source: "statusline" | "estimate"
    """
    if CLAUDE_USAGE_FILE.exists():
        try:
            data = json.loads(CLAUDE_USAGE_FILE.read_text())
            age_s = time.time() - data.get("updated_at", 0)
            # Archivo válido si tiene menos de 5 minutos
            if age_s < 300:
                return {
                    "session_pct": data.get("session_pct"),
                    "weekly_pct": data.get("weekly_pct"),
                    "session_resets_at": data.get("session_resets_at"),
                    "weekly_resets_at": data.get("weekly_resets_at"),
                    "source": "statusline",
                }
        except Exception:
            pass

    # Fallback: estimación por JSONL (solo contexto, no límite de sesión real)
    project_dir = Path.home() / ".claude" / "projects" / "-home-chuy-agent"
    jsonls = sorted(project_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not jsonls:
        return {"session_pct": None, "weekly_pct": None, "session_resets_at": None, "weekly_resets_at": None, "source": "none"}
    total_chars = 0
    with open(jsonls[0]) as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "message" in obj:
                    content = obj["message"].get("content", "")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                total_chars += len(block.get("text", ""))
                    elif isinstance(content, str):
                        total_chars += len(content)
            except Exception:
                pass
    pct = round((total_chars / 3) / 200_000 * 100, 1)
    return {"session_pct": pct, "weekly_pct": None, "session_resets_at": None, "weekly_resets_at": None, "source": "estimate"}


def telegram_notify(msg: str):
    """Envía mensaje a Jesús por Telegram."""
    if not TELEGRAM_TOKEN:
        log(f"[telegram] {msg}")
        return
    subprocess.run([
        "curl", "-s", "-X", "POST",
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": msg}),
    ], capture_output=True, timeout=15)


def cmd_queue(channel_url: str, creator_name: str, top_n: int = 20):
    """Fetch top N most-viewed videos from channel and save to queue."""
    log(f"Obteniendo top {top_n} videos del canal...")
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(view_count)s\t%(duration)s\t%(title)s", channel_url],
        capture_output=True, text=True, timeout=600
    )
    videos = []
    for line in result.stdout.strip().split("\n"):
        parts = line.strip().split("\t", 3)
        if len(parts) == 4:
            vid_id, views_str, dur_str, title = parts
            try:
                duration_s = int(float(dur_str)) if dur_str not in ("NA", "None", "") else 0
                videos.append({
                    "id": vid_id,
                    "views": int(float(views_str)) if views_str not in ("NA", "None", "") else 0,
                    "duration_min": duration_s // 60,
                    "title": title
                })
            except (ValueError, TypeError):
                pass

    videos.sort(key=lambda x: x["views"], reverse=True)
    top = videos[:top_n]

    existing = load_queue(creator_name)
    existing_ids = {v["id"] for v in existing}

    # Filter out shorts/clips (< 10 min)
    new_entries = []
    for v in top:
        if v["id"] in existing_ids:
            continue
        duration_min = v.get("duration_min", 0)
        if duration_min < 10:
            log(f"  Skip (short {duration_min}m): {v['title'][:50]}")
            continue
        new_entries.append({
            "id": v["id"],
            "title": v["title"],
            "views": v["views"],
            "duration_min": duration_min,
            "url": f"https://www.youtube.com/watch?v={v['id']}",
            "status": "pending",
        })

    queue = existing + new_entries
    save_queue(creator_name, queue)

    log(f"Cola guardada: {len(new_entries)} nuevos + {len(existing_ids)} existentes")
    print(json.dumps({
        "status": "ok",
        "creator": creator_name,
        "total_in_queue": len(queue),
        "new_added": len(new_entries),
        "top_10": [{"id": v["id"], "views": v["views"], "title": v["title"][:50]} for v in top[:10]],
    }, ensure_ascii=False, indent=2))


def schedule_cron_resume(creator_name: str, resets_at: int | None):
    """Programa un cron para retomar el batch cuando resetea la sesión."""
    if resets_at is None:
        log("No hay hora de reset disponible — no se puede programar cron automático")
        return
    reset_dt = time.gmtime(resets_at)
    # Agregar 5 minutos de margen
    resume_epoch = resets_at + 300
    resume_dt = time.gmtime(resume_epoch)
    cron_min = resume_dt.tm_min
    cron_hour = resume_dt.tm_hour
    cron_cmd = f'cd /home/chuy/podcast-analyzer && python3 agente_cerebro.py process "{creator_name}" >> /home/chuy/agent/batch.log 2>&1'
    cron_line = f"{cron_min} {cron_hour} * * * {cron_cmd}"

    # Escribir cron temporal (se ejecuta una sola vez hoy)
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    existing = result.stdout if result.returncode == 0 else ""
    # Quitar líneas anteriores de auto-resume para este creador
    clean = "\n".join(l for l in existing.splitlines() if f'process "{creator_name}"' not in l)
    new_crontab = clean.rstrip() + "\n" + cron_line + "\n"
    subprocess.run(["crontab", "-"], input=new_crontab, text=True)
    log(f"Cron programado: {cron_min:02d}:{cron_hour:02d} UTC (reset + 5 min)")


def cmd_process(creator_name: str, session_threshold: float = 80.0, weekly_threshold: float = 90.0):
    """Procesa videos pendientes en cola.

    Detiene si:
    - session_pct >= session_threshold (límite de 5 horas)
    - weekly_pct >= weekly_threshold (límite semanal de 7 días)

    Al detenerse por sesión: programa cron para auto-reanudar cuando resetea.
    """
    queue = load_queue(creator_name)
    pending = [v for v in queue if v["status"] in ("pending", "error")]

    if not pending:
        log("Cola vacía — nada que procesar.")
        print(json.dumps({"status": "empty", "creator": creator_name}))
        return

    remaining_count = len([v for v in queue if v["status"] == "pending"])
    done_count = len([v for v in queue if v["status"] == "done"])
    log(f"Cola: {len(pending)} pendientes, {done_count} listos de {len(queue)} total")
    telegram_notify(f"Cerebro batch iniciado — {creator_name}\nCola: {len(pending)} pendientes, {done_count} ya procesados")

    processed = []

    for video in pending:
        usage = get_usage()
        sess_pct = usage["session_pct"]
        week_pct = usage["weekly_pct"]
        src = usage["source"]

        log(f"Usage ({src}): sesión={sess_pct}% | semanal={week_pct}% | video: {video['title'][:50]}")

        # Check weekly limit first (no hay auto-resume posible)
        if week_pct is not None and week_pct >= weekly_threshold:
            msg = (
                f"Cerebro batch — límite semanal alcanzado\n"
                f"Semanal: {week_pct:.0f}% (umbral {weekly_threshold:.0f}%)\n"
                f"Procesados hoy: {len(processed)}\n"
                f"Pendientes: {len(pending) - len(processed)}\n"
                f"Reanuda solo al resetear el weekly"
            )
            log(msg)
            telegram_notify(msg)
            print(json.dumps({
                "status": "weekly_limit",
                "weekly_pct": week_pct,
                "processed": len(processed),
                "remaining": len(pending) - len(processed),
            }, ensure_ascii=False))
            return

        # Check session limit (auto-resume via cron)
        if sess_pct is not None and sess_pct >= session_threshold:
            resets_at = usage.get("session_resets_at")
            reset_str = ""
            if resets_at:
                reset_dt = time.gmtime(resets_at)
                reset_str = f"\nReset a las {reset_dt.tm_hour:02d}:{reset_dt.tm_min:02d} UTC"
            msg = (
                f"Cerebro batch — límite de sesión alcanzado\n"
                f"Sesión: {sess_pct:.0f}% (umbral {session_threshold:.0f}%)\n"
                f"Procesados: {len(processed)}\n"
                f"Pendientes: {len(pending) - len(processed)}"
                f"{reset_str}\n"
                f"Programando cron para auto-reanudar..."
            )
            log(msg)
            telegram_notify(msg)
            schedule_cron_resume(creator_name, resets_at)
            print(json.dumps({
                "status": "session_limit",
                "session_pct": sess_pct,
                "processed": len(processed),
                "remaining": len(pending) - len(processed),
                "session_resets_at": resets_at,
            }, ensure_ascii=False))
            return

        # Procesar este video
        try:
            url = video["url"]
            provisional_id = extract_video_id(url)
            cache_dir = CEREBRO_CACHE / provisional_id
            meta = download_audio_and_heatmap(url, cache_dir)

            if meta["video_id"] != provisional_id:
                real_dir = CEREBRO_CACHE / meta["video_id"]
                if cache_dir.exists():
                    cache_dir.rename(real_dir)
                cache_dir = real_dir
                meta["audio_path"] = str(cache_dir / "audio.mp3")

            transcript = transcribe_on_mac(meta["audio_path"], meta["video_id"], cache_dir, meta["duration_min"])
            comments = get_comments(url, cache_dir)
            heatmap_context = cross_reference_heatmap(meta["heatmap"], transcript["segments"])
            analysis = analyze_with_claude(meta, transcript, comments, heatmap_context, creator_name, CREATORS_DIR / creator_name)
            episode_title = update_obsidian(creator_name, meta, analysis)

            for q in queue:
                if q["id"] == video["id"]:
                    q["status"] = "done"
                    q["episode_title"] = episode_title
            save_queue(creator_name, queue)

            processed.append({"id": video["id"], "episode": episode_title, "views": video["views"]})
            log(f"✓ {episode_title}")
            post_usage = get_usage()
            sess = post_usage.get("session_pct")
            week = post_usage.get("weekly_pct")
            ctx = post_usage.get("context_used_pct")
            def fmt(v): return f"{v:.0f}%" if v is not None else "?"
            usage_line = f"Sesión: {fmt(sess)} | Semanal: {fmt(week)} | Ctx: {fmt(ctx)}"
            telegram_notify(
                f"✓ {creator_name}: {episode_title}\n"
                f"({len(processed)}/{len(pending)} procesados)\n"
                f"{usage_line}"
            )

        except Exception as e:
            log(f"✗ Error en {video['id']}: {e}")
            for q in queue:
                if q["id"] == video["id"]:
                    q["status"] = "error"
                    q["error"] = str(e)
            save_queue(creator_name, queue)
            telegram_notify(f"✗ Error procesando {video['title'][:50]}\n{str(e)[:200]}")

    final_msg = (
        f"Cerebro batch terminado — {creator_name}\n"
        f"Procesados: {len(processed)}/{len(pending)}\n"
        f"Cola completa: {done_count + len(processed)}/{len(queue)}"
    )
    telegram_notify(final_msg)
    print(json.dumps({
        "status": "done",
        "creator": creator_name,
        "processed": len(processed),
        "episodes": processed,
    }, ensure_ascii=False, indent=2))


# ─── Main ─────────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from any URL format."""
    m = re.search(r'(?:youtu\.be/|[?&]v=)([A-Za-z0-9_-]{11})', url)
    if m:
        return m.group(1)
    # Last path segment as fallback
    return url.rstrip("/").split("/")[-1].split("?")[0]


def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print('  python3 agente_cerebro.py <url> "Creador"           # analiza un video')
        print('  python3 agente_cerebro.py queue <channel_url> "Creador" [N]  # top N más vistos')
        print('  python3 agente_cerebro.py process "Creador"          # procesa la cola')
        sys.exit(1)

    subcommand = sys.argv[1]

    if subcommand == "queue":
        if len(sys.argv) < 4:
            print('Uso: python3 agente_cerebro.py queue <channel_url> "Creador" [N]')
            sys.exit(1)
        channel_url = sys.argv[2]
        creator_name = sys.argv[3]
        top_n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
        cmd_queue(channel_url, creator_name, top_n)
        return

    if subcommand == "process":
        if len(sys.argv) < 3:
            print('Uso: python3 agente_cerebro.py process "Creador" [session_threshold] [weekly_threshold]')
            sys.exit(1)
        creator_name = sys.argv[2]
        session_thr = float(sys.argv[3]) if len(sys.argv) > 3 else 80.0
        weekly_thr = float(sys.argv[4]) if len(sys.argv) > 4 else 90.0
        cmd_process(creator_name, session_threshold=session_thr, weekly_threshold=weekly_thr)
        return

    # Default: single video
    if len(sys.argv) < 3:
        print("Uso: python3 agente_cerebro.py <youtube_url> <nombre_creador>")
        sys.exit(1)

    url = sys.argv[1]
    creator_name = sys.argv[2]

    log(f"=== Cerebro: {creator_name} ===")

    # Safety threshold check before consuming tokens
    SESSION_THRESHOLD = 80.0
    WEEKLY_THRESHOLD = 90.0
    usage = get_usage()
    sess_pct = usage.get("session_pct")
    week_pct = usage.get("weekly_pct")
    if sess_pct is not None and sess_pct >= SESSION_THRESHOLD:
        msg = f"Abortado: sesión en {sess_pct:.0f}% (umbral {SESSION_THRESHOLD:.0f}%). Espera a que resetee."
        log(msg)
        telegram_notify(msg)
        sys.exit(1)
    if week_pct is not None and week_pct >= WEEKLY_THRESHOLD:
        msg = f"Abortado: semanal en {week_pct:.0f}% (umbral {WEEKLY_THRESHOLD:.0f}%). Espera al reset semanal."
        log(msg)
        telegram_notify(msg)
        sys.exit(1)
    log(f"Usage: sesión={sess_pct}% | semanal={week_pct}% — OK para continuar")

    # Step 1: Download (use provisional ID from URL, will be confirmed by yt-dlp)
    provisional_id = extract_video_id(url)
    cache_dir = CEREBRO_CACHE / provisional_id
    meta = download_audio_and_heatmap(url, cache_dir)

    # If yt-dlp gave a different ID (rare), rename cache dir
    if meta["video_id"] != provisional_id:
        real_dir = CEREBRO_CACHE / meta["video_id"]
        cache_dir.rename(real_dir)
        cache_dir = real_dir
        meta["audio_path"] = str(cache_dir / "audio.mp3")

    # Step 2: Transcribe
    transcript = transcribe_on_mac(meta["audio_path"], meta["video_id"], cache_dir)

    # Step 3: Comments
    comments = get_comments(url, cache_dir)

    # Step 4: Cross-reference heatmap
    heatmap_context = cross_reference_heatmap(meta["heatmap"], transcript["segments"])

    # Step 5: Analyze
    analysis = analyze_with_claude(meta, transcript, comments, heatmap_context, creator_name, CREATORS_DIR / creator_name)

    # Step 6: Update Obsidian
    episode_title = update_obsidian(creator_name, meta, analysis)

    log(f"=== Listo: {episode_title} ===")
    print(json.dumps({
        "status": "ok",
        "creator": creator_name,
        "episode": episode_title,
        "video_id": meta["video_id"],
        "views": meta["views"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
