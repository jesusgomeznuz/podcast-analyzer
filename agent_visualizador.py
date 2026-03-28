"""Subagente — Visualizaciones: genera gráficas PNG a partir de los análisis"""
import re
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # sin display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ─── Paleta de colores por hablante ────────────────────────────────────────────
SPEAKER_COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]


def _speaker_color(speaker: str, speakers: list) -> str:
    idx = speakers.index(speaker) if speaker in speakers else 0
    return SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]


# ─── Parser de emociones ────────────────────────────────────────────────────────
def parse_emociones(emociones_md: str) -> list:
    """Parsea emociones.md → lista de dicts con speaker, start, emocion, arousal, valence, dominance."""
    segments = []
    # Formato: [SPEAKER_00 0.0s] texto\n  → emoción: alegría | arousal/valence/dominance: 0.52/0.48/0.61
    pattern = re.compile(
        r"\[(\S+) ([\d.]+)s\] (.+?)\n\s+→ emoción: ([\w\s]+?)(?:\s+\([\d.]+\))? \| arousal/valence/dominance: (?:A=([\d.]+) V=([\d.]+) D=([\d.]+)|([\d.]+)/([\d.]+)/([\d.]+))",
        re.MULTILINE
    )
    for m in pattern.finditer(emociones_md):
        # Support both formats: A=x V=y D=z and x/y/z
        a = m.group(5) or m.group(8)
        v = m.group(6) or m.group(9)
        d = m.group(7) or m.group(10)
        if not (a and v and d):
            continue
        segments.append({
            "speaker": m.group(1),
            "start": float(m.group(2)),
            "text": m.group(3),
            "emocion": m.group(4).strip(),
            "arousal": float(a),
            "valence": float(v),
            "dominance": float(d),
        })
    return segments


def parse_dinamica(dinamica_md: str) -> dict:
    """Parsea dinamica.md → dict {speaker: {words, pct_words, duration, pct_duration, turns}}"""
    speakers = {}
    pattern = re.compile(
        r"\*\*(\S+)\*\*: (\d+) palabras \(([\d.]+)%\) \| ([\d.]+)s hablando \(([\d.]+)%\) \| (\d+) turnos"
    )
    for m in pattern.finditer(dinamica_md):
        speakers[m.group(1)] = {
            "words": int(m.group(2)),
            "pct_words": float(m.group(3)),
            "duration": float(m.group(4)),
            "pct_duration": float(m.group(5)),
            "turns": int(m.group(6)),
        }
    return speakers


# ─── Gráfica 1: Línea de tiempo emocional (arousal + valence) ─────────────────
def plot_emotion_timeline(segments: list, report_dir: Path) -> Path | None:
    if not segments:
        return None

    speakers = sorted(set(s["speaker"] for s in segments))
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Línea de tiempo emocional por hablante", fontsize=14, fontweight="bold")

    dims = [("arousal", "Arousal"), ("valence", "Valence"), ("dominance", "Dominance")]
    for ax, (dim, label) in zip(axes, dims):
        for spk in speakers:
            segs = [s for s in segments if s["speaker"] == spk]
            if not segs:
                continue
            xs = [s["start"] for s in segs]
            ys = [s[dim] for s in segs]
            color = _speaker_color(spk, speakers)
            ax.plot(xs, ys, "o-", color=color, label=spk, linewidth=1.5,
                    markersize=4, alpha=0.85)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_ylabel(label, fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Tiempo (s)", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    out = report_dir / "grafica_timeline_emocional.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


# ─── Gráfica 2: Distribución de emociones categóricas por hablante ────────────
def plot_emotion_distribution(segments: list, report_dir: Path) -> Path | None:
    if not segments:
        return None

    speakers = sorted(set(s["speaker"] for s in segments))
    all_emotions = sorted(set(s["emocion"] for s in segments if s["emocion"] != "—"))
    if not all_emotions:
        return None

    fig, axes = plt.subplots(1, len(speakers), figsize=(6 * len(speakers), 5), squeeze=False)
    fig.suptitle("Distribución de emociones por hablante", fontsize=14, fontweight="bold")

    for i, spk in enumerate(speakers):
        ax = axes[0][i]
        segs = [s for s in segments if s["speaker"] == spk and s["emocion"] != "—"]
        if not segs:
            ax.set_title(spk)
            continue

        counts = {e: sum(1 for s in segs if s["emocion"] == e) for e in all_emotions}
        total = sum(counts.values()) or 1
        pcts = {e: v / total * 100 for e, v in counts.items()}
        sorted_emos = sorted(pcts.items(), key=lambda x: x[1], reverse=True)
        labels, values = zip(*sorted_emos) if sorted_emos else ([], [])

        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        bars = ax.barh(labels, values, color=colors)
        ax.set_xlabel("%", fontsize=9)
        ax.set_title(spk, fontsize=11, fontweight="bold",
                     color=_speaker_color(spk, speakers))
        ax.set_xlim(0, 105)
        for bar, val in zip(bars, values):
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%", va="center", fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = report_dir / "grafica_emociones_categoricas.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


# ─── Gráfica 3: Dinámica de habla ──────────────────────────────────────────────
def plot_talk_dynamics(speakers_data: dict, report_dir: Path) -> Path | None:
    if not speakers_data:
        return None

    speakers = list(speakers_data.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Dinámica conversacional", fontsize=14, fontweight="bold")
    colors = [_speaker_color(s, speakers) for s in speakers]

    # Pie: tiempo de habla
    durations = [speakers_data[s]["pct_duration"] for s in speakers]
    wedges, texts, autotexts = axes[0].pie(
        durations, labels=speakers, autopct="%1.0f%%",
        colors=colors, startangle=90,
        textprops={"fontsize": 10}
    )
    for at in autotexts:
        at.set_fontsize(9)
    axes[0].set_title("Tiempo de habla", fontsize=11)

    # Bar: palabras
    words = [speakers_data[s]["words"] for s in speakers]
    bars = axes[1].bar(speakers, words, color=colors)
    axes[1].set_title("Palabras", fontsize=11)
    axes[1].set_ylabel("Cantidad")
    for bar, val in zip(bars, words):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     str(val), ha="center", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    # Bar: turnos
    turns = [speakers_data[s]["turns"] for s in speakers]
    bars = axes[2].bar(speakers, turns, color=colors)
    axes[2].set_title("Turnos de habla", fontsize=11)
    axes[2].set_ylabel("Cantidad")
    for bar, val in zip(bars, turns):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(val), ha="center", fontsize=9)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = report_dir / "grafica_dinamica_conversacional.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


# ─── Gráfica 4: Arousal vs Valence scatter (espacio emocional) ─────────────────
def plot_emotion_space(segments: list, report_dir: Path) -> Path | None:
    if not segments:
        return None

    speakers = sorted(set(s["speaker"] for s in segments))
    fig, ax = plt.subplots(figsize=(8, 7))

    for spk in speakers:
        segs = [s for s in segments if s["speaker"] == spk]
        xs = [s["valence"] for s in segs]
        ys = [s["arousal"] for s in segs]
        color = _speaker_color(spk, speakers)
        ax.scatter(xs, ys, c=color, label=spk, alpha=0.6, s=60, edgecolors="white", linewidth=0.5)

        # Centroide
        if xs and ys:
            cx, cy = np.mean(xs), np.mean(ys)
            ax.scatter([cx], [cy], c=color, s=200, marker="*",
                       edgecolors="black", linewidth=0.8, zorder=5)
            ax.annotate(f"{spk}\n(μ)", (cx, cy), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=color)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Valence (negativo ← → positivo)", fontsize=10)
    ax.set_ylabel("Arousal (calmado ← → activado)", fontsize=10)
    ax.set_title("Espacio emocional (arousal vs valence)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Etiquetas de cuadrantes
    for x, y, txt in [(0.25, 0.75, "Enojo/Miedo"), (0.75, 0.75, "Alegría/Excitación"),
                      (0.25, 0.25, "Tristeza"), (0.75, 0.25, "Calma/Satisfacción")]:
        ax.text(x, y, txt, ha="center", va="center", fontsize=8,
                color="gray", alpha=0.6, fontstyle="italic")

    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()

    out = report_dir / "grafica_espacio_emocional.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


# ─── Punto de entrada ──────────────────────────────────────────────────────────
def run(report_dir: Path) -> list:
    """Genera todas las gráficas para un reporte dado. Retorna lista de paths."""
    emociones_path = report_dir / "emociones.md"
    dinamica_path = report_dir / "dinamica.md"

    emociones_md = emociones_path.read_text(encoding="utf-8") if emociones_path.exists() else ""
    dinamica_md = dinamica_path.read_text(encoding="utf-8") if dinamica_path.exists() else ""

    segments = parse_emociones(emociones_md)
    speakers_data = parse_dinamica(dinamica_md)

    charts = []
    for fn, args in [
        (plot_emotion_timeline, (segments, report_dir)),
        (plot_emotion_space, (segments, report_dir)),
        (plot_emotion_distribution, (segments, report_dir)),
        (plot_talk_dynamics, (speakers_data, report_dir)),
    ]:
        try:
            path = fn(*args)
            if path:
                charts.append(path)
        except Exception as e:
            print(f"   ⚠️  {fn.__name__}: {e}")

    return charts
