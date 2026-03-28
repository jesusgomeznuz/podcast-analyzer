import subprocess
from pathlib import Path

VENV_BIN = Path(__file__).parent / "venv" / "bin"


def download_audio(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")
    subprocess.run(
        [str(VENV_BIN / "yt-dlp"), "-x", "--audio-format", "mp3", "-o", output_template, url],
        capture_output=True, text=True, check=True
    )
    mp3s = sorted(output_dir.glob("*.mp3"), key=lambda p: p.stat().st_mtime)
    if not mp3s:
        raise RuntimeError("No se encontró MP3 después de descargar.")
    return mp3s[-1]
