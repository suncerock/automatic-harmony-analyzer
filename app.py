from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import streamlit as st

from acr.inference import _build_model, _resolve_chord_ckpt_path, _resolve_mpe_ckpt_path, run_inference

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "tmp_demo"
UPLOAD_DIR.mkdir(exist_ok=True)

AUDIO_TYPES = ["wav", "flac", "mp3", "ogg", "aiff", "aif"]
SYMBOLIC_TYPES = ["mid", "midi", "xml", "musicxml"]


@st.cache_resource
def get_model(backbone_type: str, model_type: str | None, input_kind: str):
    use_mpe = input_kind == "audio"
    chord_ckpt_path = _resolve_chord_ckpt_path(backbone_type, model_type, use_mpe=use_mpe)
    mpe_ckpt_path = _resolve_mpe_ckpt_path() if use_mpe else Path("unused")
    return _build_model(
        backbone_type=backbone_type,
        chord_ckpt_path=chord_ckpt_path,
        mpe_ckpt_path=mpe_ckpt_path,
        use_mpe=use_mpe,
    )


def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
    unique_name = f"{uuid.uuid4().hex}.{suffix}"
    target_path = UPLOAD_DIR / unique_name
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target_path


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


def _probability_to_color(value: float, palette: tuple[int, int, int]) -> str:
    intensity = float(np.clip(value, 0.0, 1.0))
    r, g, b = palette
    alpha = 0.12 + 0.88 * intensity
    return f"rgba({r}, {g}, {b}, {alpha})"



def render_soft_output(npz_path: Path, input_kind: str) -> None:
    soft = dict(np.load(npz_path))
    available = {"root": soft.get("root"), "bass": soft.get("bass"), "pitch": soft.get("pitch")}
    required_keys = [key for key, arr in available.items() if arr is not None]
    fps = 10.0 if input_kind == "audio" else 20.0

    palettes = {
        "root": (72, 140, 255),
        "bass": (255, 166, 77),
        "pitch": (64, 196, 122),
    }
    labels = {"root": "Root", "bass": "Bass", "pitch": "Pitch class"}
    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    st.subheader("Soft output")
    for key in required_keys:
        matrix = _normalize_probability_matrix(available[key], key)
        if matrix.shape[0] != 12 or matrix.shape[1] == 0:
            continue

        time_steps = matrix.shape[1]
        if time_steps == 0:
            continue

        max_cols = 500
        stride = max(1, int(np.ceil(time_steps / max_cols)))
        sampled = matrix[:, ::stride]
        sampled_cols = sampled.shape[1]
        label_every = max(1, sampled_cols // 12)
        title = labels.get(key)
        cell_size = 12
        left_width = 52

        html = [
            "<div style='margin-top: 8px; margin-bottom: 8px; color: var(--text-color); font-weight: 600; font-size: 16px;'>",
            f"{title} probability" if title else "",
            "</div>",
            "<div style='overflow-x: auto; max-width: 100%; padding: 8px 0;'>",
            "<div style='min-width: 980px; display: flex; flex-direction: column; gap: 2px;'>",
            f"<div style='display: grid; grid-template-columns: {left_width}px repeat({sampled_cols}, {cell_size}px); gap: 2px; align-items: end; white-space: nowrap;' >",
        ]

        html.append(f"<div style='font-size: 11px; color: var(--text-color); text-align: center; white-space: nowrap;'></div>")
        for t_idx in range(sampled_cols):
            seconds = float((t_idx * stride) / fps)
            label = f"{seconds:.1f}" if t_idx % label_every == 0 or sampled_cols <= 12 else ""
            html.append(
                f"<div style='font-size: 11px; color: var(--text-color); text-align: center; min-width: {cell_size}px; height: 14px; line-height: 14px; white-space: nowrap; overflow: visible;'>{label}</div>"
            )
        html.append("</div>")

        for pitch_idx in reversed(range(12)):
            html.append(f"<div style='display: grid; grid-template-columns: {left_width}px repeat({sampled_cols}, {cell_size}px); gap: 2px; align-items: center; white-space: nowrap;' >")
            html.append(f"<div style='font-size: 11px; color: var(--text-color); text-align: right; padding-right: 6px; white-space: nowrap;'>{pitch_names[pitch_idx]}</div>")
            for t_idx in range(sampled_cols):
                value = float(sampled[pitch_idx, t_idx])
                color = _probability_to_color(value, palettes[key])
                html.append(
                    f"<span title='pitch={pitch_idx}, t={t_idx * stride} frames ({(t_idx * stride) / fps:.2f}s), p={value:.3f}' "
                    f"style='display:block; width:{cell_size}px; height:{cell_size}px; border-radius:2px; background:{color}; border: 1px solid rgba(0,0,0,0.09);'></span>"
                )
            html.append("</div>")

        html.extend([
            f"<div style='margin-top: 8px; margin-left: {left_width + 4}px; font-size: 11px; color: var(--text-color); text-align: left; white-space: nowrap;'>Time ({'s' if input_kind == 'audio' else 'measures'})</div>",
            "</div>",
            "</div>",
        ])
        st.markdown("".join(html), unsafe_allow_html=True)


st.set_page_config(page_title="Automatic Harmony Analyzer", page_icon="🎵", layout="wide")
st.title("Automatic Harmony Analyzer")
st.caption("Local demo for audio and symbolic chord recognition inference.")

input_kind = st.radio("Input type", ["audio", "symbolic"], horizontal=True)

if input_kind == "audio":
    file_types = AUDIO_TYPES
    backbone_type = st.selectbox("Backbone", ["crnn", "conformer"], index=0)
    model_type = st.selectbox("Audio checkpoint family", ["classical", "pop"], index=0)
else:
    file_types = SYMBOLIC_TYPES
    backbone_type = st.selectbox("Backbone", ["crnn", "conformer"], index=0)
    model_type = None

uploaded_file = st.file_uploader(
    "Upload a file",
    type=file_types,
    help="For audio: WAV/FLAC/MP3/OGG/AIFF; for symbolic: MIDI or MusicXML.",
)

if uploaded_file is not None:
    st.write(f"Selected file: {uploaded_file.name}")

    if st.button("Run inference", type="primary"):
        try:
            print("[demo] Running inference...")
            saved_path = save_uploaded_file(uploaded_file)
            output_stem = UPLOAD_DIR / f"demo_out_{uuid.uuid4().hex}"

            with st.status("Running inference...", expanded=True) as status:
                st.write("Processing the uploaded file and generating chord predictions.")
                model = get_model(backbone_type=backbone_type, model_type=model_type, input_kind=input_kind)

                if input_kind == "audio":
                    run_inference(
                        model=model,
                        output_path=output_stem,
                        audio_path=saved_path,
                        symbolic_path=None,
                        save_soft_output=True,
                        clip_seconds=None,
                    )
                else:
                    run_inference(
                        model=model,
                        output_path=output_stem,
                        audio_path=None,
                        symbolic_path=saved_path,
                        save_soft_output=True,
                    )

                status.update(label="Inference finished.", state="complete")

            lab_path = output_stem.with_suffix(".lab")
            soft_path = output_stem.with_suffix(".npz")
            if not lab_path.exists():
                raise FileNotFoundError(f"Expected output file was not created: {lab_path}")

            lines = lab_path.read_text(encoding="utf-8").strip().splitlines()
            st.success("Inference finished.")
            st.download_button(
                label="Download .lab output",
                data=lab_path.read_bytes(),
                file_name=lab_path.name,
                mime="text/plain",
            )

            render_soft_output(soft_path, input_kind=input_kind)

            st.subheader("Predicted chords")
            panel_html = (
                '<div style="max-height: 420px; overflow-y: auto; border: 1px solid var(--border-color); border-radius: 8px; '
                'padding: 12px; background: var(--secondary-background-color); color: var(--text-color); '
                'font-family: monospace; white-space: pre-wrap; scrollbar-color: var(--text-color) var(--background-color);">'
                + "\n".join(lines)
                + "</div>"
            )
            st.markdown(panel_html, unsafe_allow_html=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Inference failed: {exc}")

st.sidebar.header("How to use")
st.sidebar.markdown(
    """
    1. Choose whether your input is audio or symbolic.
    2. Upload a file.
    3. Select the backbone and checkpoint family.
    4. Click Run inference.
    5. Download the generated .lab output.
    """
)
st.sidebar.markdown("This app reuses the repo's inference runtime directly, so it is easy to demo on a local machine.")

#  python -m streamlit run app.py --server.headless true --server.port 8501