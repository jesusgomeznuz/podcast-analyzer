"""Subagente 1 — Descargador: URL → MP3 (y opcionalmente MP4 para análisis facial)"""
import subprocess
from pathlib import Path

VENV_BIN = Path(__file__).parent / "venv" / "bin"


def _get_video_id(url: str) -> str | None:
    result = subprocess.run(
        [str(VENV_BIN / "yt-dlp"), "--get-id", url],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        vid = result.stdout.strip().split("\n")[0]
        return vid if vid else None
    return None


def run(url: str, output_dir: Path) -> Path:
    """Descarga audio como MP3. Retorna el archivo correcto por video ID."""
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = _get_video_id(url)
    if video_id:
        cached = output_dir / f"{video_id}.mp3"
        if cached.exists():
            return cached

    output_template = str(output_dir / "%(id)s.%(ext)s")
    subprocess.run(
        [str(VENV_BIN / "yt-dlp"), "-x", "--audio-format", "mp3",
         "--audio-quality", "5",
         "-o", output_template, url],
        capture_output=True, text=True, check=True
    )

    if video_id:
        mp3 = output_dir / f"{video_id}.mp3"
        if mp3.exists():
            return mp3

    # Fallback: latest MP3 (solo si no pudimos obtener el ID)
    mp3s = sorted(output_dir.glob("*.mp3"), key=lambda p: p.stat().st_mtime)
    if not mp3s:
        raise RuntimeError("No se encontró MP3 después de descargar.")
    return mp3s[-1]


def run_video(url: str, output_dir: Path) -> Path:
    """Descarga video como MP4 (para análisis facial). Usa caché si ya existe."""
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = _get_video_id(url)
    if video_id:
        cached = output_dir / f"{video_id}.mp4"
        if cached.exists():
            return cached

    output_template = str(output_dir / "%(id)s.%(ext)s")
    subprocess.run(
        [str(VENV_BIN / "yt-dlp"),
         "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
         "--merge-output-format", "mp4",
         "-o", output_template, url],
        capture_output=True, text=True, check=True
    )

    if video_id:
        mp4 = output_dir / f"{video_id}.mp4"
        if mp4.exists():
            return mp4

    mp4s = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not mp4s:
        raise RuntimeError("No se encontró MP4 después de descargar.")
    return mp4s[-1]
