"""Live chord player and soft-output visualisation for the Streamlit demo."""

from __future__ import annotations

import base64
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
AUDIO_FPS = 10.0
SYMBOLIC_FPM = 20.0
PLAYBACK_SR = 22050

SOFT_KEYS = ("root", "bass", "pitch")
SOFT_TITLES = {"root": "Root", "bass": "Bass", "pitch": "Pitch class"}
SOFT_PALETTES = {
    "root": (72, 140, 255),
    "bass": (255, 166, 77),
    "pitch": (64, 196, 122),
}
INITIAL_WINDOW = {"audio": 30.0, "symbolic": 16.0}
ROW_HEIGHT = 210  # ~15 px per pitch class, enough for all 12 tick labels
WHITE_KEY_BAND = "rgba(128,128,128,0.14)"  # neutral grey, reads on light and dark themes


# --------------------------------------------------------------------------- #
# Soft-output helpers
# --------------------------------------------------------------------------- #
def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=0, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=0, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def _normalize_probability_matrix(array: np.ndarray, key: str) -> np.ndarray:
    matrix = np.asarray(array, dtype=np.float32)
    if key in {"root", "bass"}:
        if matrix.shape[0] != 12:
            matrix = matrix.T
        return _softmax(matrix)

    if key == "pitch":
        if matrix.shape[0] != 12:
            matrix = matrix.T
        return _sigmoid(matrix)

    raise ValueError(f"Unsupported soft-output key: {key!r}")


def parse_lab(lab_path: Path) -> list[dict]:
    """Read a tab-separated ``start end label`` file into a list of segments."""
    segments = []
    for line in Path(lab_path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        segments.append({"start": float(parts[0]), "end": float(parts[1]), "label": parts[2]})
    return segments


def build_soft_output_figure(npz_path: Path, input_kind: str):
    """Stacked root/bass/pitch-class heatmaps sharing one time axis."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    soft = dict(np.load(npz_path))
    keys = [key for key in SOFT_KEYS if soft.get(key) is not None]
    if not keys:
        raise ValueError(f"No soft outputs found in {npz_path}")

    fps = AUDIO_FPS if input_kind == "audio" else SYMBOLIC_FPM
    unit = "s" if input_kind == "audio" else "measures"

    fig = make_subplots(
        rows=len(keys),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[f"{SOFT_TITLES[key]} probability" for key in keys],
    )

    duration = 0.0
    for row, key in enumerate(keys, start=1):
        matrix = _normalize_probability_matrix(soft[key], key)
        times = np.arange(matrix.shape[1]) / fps
        duration = max(duration, matrix.shape[1] / fps)
        r, g, b = SOFT_PALETTES[key]
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=times,
                y=PITCH_NAMES,
                zmin=0.0,
                zmax=1.0,
                colorscale=[[0.0, f"rgba({r},{g},{b},0.04)"], [1.0, f"rgb({r},{g},{b})"]],
                showscale=False,
                name=SOFT_TITLES[key],
                hovertemplate=f"t=%{{x:.2f}} {unit}<br>%{{y}}<br>p=%{{z:.3f}}<extra>{SOFT_TITLES[key]}</extra>",
            ),
            row=row,
            col=1,
        )

    window = min(INITIAL_WINDOW[input_kind], duration)
    fig.update_xaxes(range=[0.0, window], showgrid=False, zeroline=False)
    fig.update_xaxes(
        title_text=f"Time ({unit})",
        rangeslider=dict(visible=True, thickness=0.08),
        row=len(keys),
        col=1,
    )
    # Label every pitch class (Plotly otherwise thins out category ticks) and
    # set the white keys apart: bold labels plus a faint band behind their rows.
    fig.update_yaxes(
        showgrid=False,
        fixedrange=True,
        tickfont=dict(size=10),
        tickmode="array",
        tickvals=PITCH_NAMES,
        ticktext=[name if "#" in name else f"<b>{name}</b>" for name in PITCH_NAMES],
    )
    for row in range(1, len(keys) + 1):
        for idx, name in enumerate(PITCH_NAMES):
            if "#" not in name:
                fig.add_hrect(
                    y0=idx - 0.5,
                    y1=idx + 0.5,
                    fillcolor=WHITE_KEY_BAND,
                    line_width=0,
                    layer="below",
                    row=row,
                    col=1,
                )
    fig.update_layout(
        height=ROW_HEIGHT * len(keys) + 120,
        margin=dict(l=48, r=16, t=32, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        dragmode="pan",
        hovermode="closest",
    )
    return fig


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
def encode_audio_for_playback(audio_path: Path) -> tuple[str, str]:
    """Return (base64, mime) of a compact mono rendition of the audio for the browser."""
    import librosa
    import soundfile as sf

    audio, sr = librosa.load(str(audio_path), sr=PLAYBACK_SR, mono=True)
    buffer = io.BytesIO()
    try:
        sf.write(buffer, audio, sr, format="MP3")
        mime = "audio/mpeg"
    except Exception:  # noqa: BLE001 - libsndfile without MP3 support
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV", subtype="PCM_16")
        mime = "audio/wav"
    return base64.b64encode(buffer.getvalue()).decode("ascii"), mime


@lru_cache(maxsize=1)
def _plotly_js() -> str:
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


def _script_json(value) -> str:
    """JSON that is safe to inline inside a <script> tag."""
    text = value if isinstance(value, str) else json.dumps(value)
    return text.replace("</", "<\\/")


# --------------------------------------------------------------------------- #
# Live player component
# --------------------------------------------------------------------------- #
_PLAYER_TEMPLATE = r"""
<style>
  :root {
    --fg: #31333f; --muted: #6b6f7b; --card: #f0f2f6; --border: rgba(49,51,63,0.15);
    --accent: #ff4b4b; --bg: #ffffff;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --fg: #fafafa; --muted: #a3a8b8; --card: #262730; --border: rgba(250,250,250,0.15);
      --accent: #ff4b4b; --bg: #0e1117;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "Source Sans Pro", system-ui, sans-serif; color: var(--fg); background: transparent; }
  audio { width: 100%; margin-bottom: 12px; }
  .cards { display: grid; grid-template-columns: 1fr 2fr 1fr; gap: 10px; margin-bottom: 12px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px;
          display: flex; flex-direction: column; justify-content: center; min-width: 0; }
  .card .cap { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
  .card .lbl { font-family: ui-monospace, Menlo, monospace; font-weight: 600; overflow: hidden;
               text-overflow: ellipsis; white-space: nowrap; }
  .card.side .lbl { font-size: 18px; color: var(--muted); }
  .card.now { border-color: var(--accent); }
  .card.now .lbl { font-size: 34px; }
  .card .time { font-size: 12px; color: var(--muted); }
  .strip-wrap { position: relative; overflow-x: auto; overflow-y: hidden; border: 1px solid var(--border);
                border-radius: 8px; height: 46px; margin-bottom: 8px; }
  .strip { position: relative; height: 100%; }
  .seg { position: absolute; top: 4px; bottom: 4px; border-radius: 4px; cursor: pointer; overflow: hidden;
         font: 11px ui-monospace, Menlo, monospace; color: #fff; padding: 2px 4px; white-space: nowrap;
         text-overflow: ellipsis; border: 1px solid rgba(0,0,0,0.12); opacity: .75; }
  .seg.active { opacity: 1; outline: 2px solid var(--fg); outline-offset: -1px; }
  .playhead { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--accent); pointer-events: none; }
  .plot-wrap { position: relative; }
  #plot-playhead { display: none; z-index: 10; }
</style>

<audio id="audio" controls preload="auto" src="data:__MIME__;base64,__AUDIO__"></audio>

<div class="cards">
  <div class="card side"><span class="cap">Previous</span><span class="lbl" id="prev">–</span></div>
  <div class="card now"><span class="cap">Now</span><span class="lbl" id="now">–</span><span class="time" id="now-time"></span></div>
  <div class="card side"><span class="cap">Next</span><span class="lbl" id="next">–</span></div>
</div>

<div class="strip-wrap" id="strip-wrap"><div class="strip" id="strip"><div class="playhead" id="strip-playhead"></div></div></div>

<div class="plot-wrap" id="plot-wrap">
  <div id="plot"></div>
  <div class="playhead" id="plot-playhead"></div>
</div>

<script>__PLOTLY_JS__</script>
<script>
(function () {
  const SEGMENTS = __SEGMENTS__;
  const FIGURE = __FIGURE__;
  const XAXIS = "__XAXIS__";
  const PX_PER_SEC = 40;
  const FOLLOW_PAUSE_MS = 3000;
  const PITCH = {C:0,"C#":1,Db:1,D:2,"D#":3,Eb:3,E:4,F:5,"F#":6,Gb:6,G:7,"G#":8,Ab:8,A:9,"A#":10,Bb:10,B:11};

  const audio = document.getElementById("audio");
  const prevEl = document.getElementById("prev");
  const nowEl = document.getElementById("now");
  const nowTimeEl = document.getElementById("now-time");
  const nextEl = document.getElementById("next");
  const stripWrap = document.getElementById("strip-wrap");
  const strip = document.getElementById("strip");
  const stripHead = document.getElementById("strip-playhead");
  const gd = document.getElementById("plot");
  const plotHead = document.getElementById("plot-playhead");

  const duration = SEGMENTS.length ? SEGMENTS[SEGMENTS.length - 1].end : 0;
  const fmt = (t) => { const m = Math.floor(t / 60); const s = (t % 60).toFixed(1).padStart(4, "0"); return m + ":" + s; };

  function rootColor(label) {
    const root = label.split(":")[0].split("/")[0];
    if (!(root in PITCH)) return "#888";
    return "hsl(" + (PITCH[root] * 30) + ", 60%, 45%)";
  }

  // ---- chord timeline strip ------------------------------------------------
  strip.style.width = Math.max(duration * PX_PER_SEC, stripWrap.clientWidth) + "px";
  const segEls = SEGMENTS.map((seg, i) => {
    const el = document.createElement("div");
    el.className = "seg";
    el.style.left = (seg.start * PX_PER_SEC) + "px";
    el.style.width = Math.max(1, (seg.end - seg.start) * PX_PER_SEC - 1) + "px";
    el.style.background = rootColor(seg.label);
    el.textContent = seg.label;
    el.title = seg.label + "  (" + fmt(seg.start) + " – " + fmt(seg.end) + ")";
    el.addEventListener("click", () => { audio.currentTime = seg.start + 1e-3; });
    strip.insertBefore(el, stripHead);
    return el;
  });

  function findSegment(t) {
    let lo = 0, hi = SEGMENTS.length - 1, ans = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (SEGMENTS[mid].start <= t) { ans = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    return ans;
  }

  let currentIdx = -2;
  function updateChord(idx) {
    if (idx === currentIdx) return;
    if (currentIdx >= 0 && segEls[currentIdx]) segEls[currentIdx].classList.remove("active");
    currentIdx = idx;
    const seg = SEGMENTS[idx];
    prevEl.textContent = idx > 0 ? SEGMENTS[idx - 1].label : "–";
    nowEl.textContent = seg ? seg.label : "–";
    nowTimeEl.textContent = seg ? fmt(seg.start) + " – " + fmt(seg.end) : "";
    nextEl.textContent = idx >= 0 && idx + 1 < SEGMENTS.length ? SEGMENTS[idx + 1].label : "–";
    if (seg) segEls[idx].classList.add("active");
  }

  // ---- heatmaps --------------------------------------------------------------
  let plotReady = false;
  let ignoreRelayout = false;
  let userInteractedAt = 0;
  const fg = getComputedStyle(document.documentElement).getPropertyValue("--fg").trim() || "#31333f";
  FIGURE.layout = FIGURE.layout || {};
  FIGURE.layout.font = Object.assign({}, FIGURE.layout.font, { color: fg });
  (FIGURE.layout.annotations || []).forEach((a) => { a.font = Object.assign({}, a.font, { color: fg }); });

  Plotly.newPlot(gd, FIGURE.data, FIGURE.layout, { responsive: true, displaylogo: false, scrollZoom: true })
    .then(() => {
      plotReady = true;
      gd.on("plotly_click", (ev) => {
        if (ev && ev.points && ev.points.length) audio.currentTime = Number(ev.points[0].x);
      });
      gd.on("plotly_relayout", (ev) => {
        if (ignoreRelayout) { ignoreRelayout = false; return; }
        if (ev && Object.keys(ev).some((k) => k.startsWith("xaxis"))) userInteractedAt = performance.now();
      });
    });

  function updatePlotPlayhead(t) {
    if (!plotReady || !gd._fullLayout) return;
    const xa = gd._fullLayout[XAXIS];
    const size = gd._fullLayout._size;
    if (!xa || !size) return;
    const [r0, r1] = xa.range.map(Number);

    if (!audio.paused && performance.now() - userInteractedAt > FOLLOW_PAUSE_MS && (t < r0 || t > r1)) {
      const width = r1 - r0;
      const start = Math.max(0, t - 0.05 * width);
      ignoreRelayout = true;
      Plotly.relayout(gd, { [XAXIS + ".range"]: [start, start + width] });
      return;
    }

    if (t < r0 || t > r1) { plotHead.style.display = "none"; return; }
    const x = xa._offset + ((t - r0) / (r1 - r0)) * xa._length;
    plotHead.style.display = "block";
    plotHead.style.left = (x - 1) + "px";
    plotHead.style.top = size.t + "px";
    plotHead.style.bottom = "auto";
    plotHead.style.height = size.h + "px";
  }

  // ---- sync loop -------------------------------------------------------------
  function tick() {
    const t = audio.currentTime || 0;
    updateChord(findSegment(t));

    const sx = t * PX_PER_SEC;
    stripHead.style.left = (sx - 1) + "px";
    if (!audio.paused) {
      const view0 = stripWrap.scrollLeft, view1 = view0 + stripWrap.clientWidth;
      if (sx < view0 || sx > view1 - 40) stripWrap.scrollLeft = sx - 40;
    }

    updatePlotPlayhead(t);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();
</script>
"""


def build_player_html(audio_b64: str, mime: str, segments: list[dict], fig) -> str:
    n_rows = sum(1 for trace in fig.data if trace.type == "heatmap")
    bottom_axis = "xaxis" if n_rows <= 1 else f"xaxis{n_rows}"
    replacements = {
        "__MIME__": mime,
        "__AUDIO__": audio_b64,
        "__SEGMENTS__": _script_json(segments),
        "__FIGURE__": _script_json(fig.to_json()),
        "__XAXIS__": bottom_axis,
        "__PLOTLY_JS__": _plotly_js().replace("</script", "<\\/script"),
    }
    html = _PLAYER_TEMPLATE
    # Plotly JS last: it is huge and must not be scanned for the other placeholders.
    for key, value in replacements.items():
        if key != "__PLOTLY_JS__":
            html = html.replace(key, value)
    return html.replace("__PLOTLY_JS__", replacements["__PLOTLY_JS__"])


def render_live_player(audio_b64: str, mime: str, segments: list[dict], fig) -> None:
    import streamlit as st

    height = int(fig.layout.height or 600) + 200
    html = build_player_html(audio_b64, mime, segments, fig)
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:  # Streamlit releases before st.iframe existed
        import streamlit.components.v1 as components

        components.html(html, height=height, scrolling=False)
