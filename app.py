from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from acr.inference import _build_model, _resolve_chord_ckpt_path, _resolve_mpe_ckpt_path, run_inference
from acr.player import build_soft_output_figure, encode_audio_for_playback, parse_lab, render_live_player
from acr.youtube import download_youtube_audio

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


@st.cache_data(show_spinner="Preparing audio for playback...")
def get_playback_audio(audio_path: str, mtime: float) -> tuple[str, str]:
    return encode_audio_for_playback(Path(audio_path))


def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
    unique_name = f"{uuid.uuid4().hex}.{suffix}"
    target_path = UPLOAD_DIR / unique_name
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target_path


def render_results(result: dict) -> None:
    lab_path: Path = result["lab_path"]
    soft_path: Path = result["soft_path"]
    input_kind: str = result["input_kind"]
    audio_path: Path | None = result.get("audio_path")

    st.success("Inference finished.")
    lab_col, npz_col, _ = st.columns([1, 1, 3])
    with lab_col:
        st.download_button(
            label="Download .lab output",
            data=lab_path.read_bytes(),
            file_name=lab_path.name,
            mime="text/plain",
            width="stretch",
        )
    if soft_path.exists():
        with npz_col:
            st.download_button(
                label="Download soft output (.npz)",
                data=soft_path.read_bytes(),
                file_name=soft_path.name,
                mime="application/octet-stream",
                width="stretch",
            )

    fig = build_soft_output_figure(soft_path, input_kind=input_kind) if soft_path.exists() else None

    if input_kind == "audio" and audio_path is not None and audio_path.exists() and fig is not None:
        st.subheader("Live player")
        st.caption("Press play to follow the predicted chords. Click the chord strip or the heatmaps to seek.")
        audio_b64, mime = get_playback_audio(str(audio_path), audio_path.stat().st_mtime)
        render_live_player(audio_b64, mime, parse_lab(lab_path), fig)
        return  # the player's chord cards and strip replace the raw .lab listing

    if fig is not None:
        st.subheader("Soft output")
        st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})

    st.subheader("Predicted chords")
    lines = lab_path.read_text(encoding="utf-8").strip().splitlines()
    panel_html = (
        '<div style="max-height: 420px; overflow-y: auto; border: 1px solid var(--border-color); border-radius: 8px; '
        'padding: 12px; background: var(--secondary-background-color); color: var(--text-color); '
        'font-family: monospace; white-space: pre-wrap; scrollbar-color: var(--text-color) var(--background-color);">'
        + "\n".join(lines)
        + "</div>"
    )
    st.markdown(panel_html, unsafe_allow_html=True)


st.set_page_config(page_title="Automatic Harmony Analyzer", page_icon="🎵", layout="wide")
st.title("Automatic Harmony Analyzer")
st.caption("Local demo for audio and symbolic chord recognition inference.")

input_kind = st.radio("Input type", ["audio", "symbolic"], horizontal=True)

if input_kind == "audio":
    file_types = AUDIO_TYPES
    audio_source = st.radio("Audio source", ["Upload", "YouTube"], horizontal=True)
    backbone_type = st.selectbox("Backbone", ["crnn", "conformer"], index=0)
    model_type = st.selectbox("Audio checkpoint family", ["classical", "pop"], index=0)
else:
    file_types = SYMBOLIC_TYPES
    backbone_type = st.selectbox("Backbone", ["crnn", "conformer"], index=0)
    model_type = None

youtube_url = ""
uploaded_file = None
if input_kind == "audio" and audio_source == "YouTube":
    youtube_url = st.text_input("YouTube video URL").strip()
else:
    uploaded_file = st.file_uploader(
        "Upload a file",
        type=file_types,
        help="For audio: WAV/FLAC/MP3/OGG/AIFF; for symbolic: MIDI or MusicXML.",
    )

if uploaded_file is not None or youtube_url:
    if uploaded_file is not None:
        st.write(f"Selected file: {uploaded_file.name}")

    if st.button("Run inference", type="primary"):
        st.session_state.pop("result", None)
        try:
            print("[demo] Running inference...")
            output_stem = UPLOAD_DIR / f"demo_out_{uuid.uuid4().hex}"

            with st.status("Running inference...", expanded=True) as status:
                if youtube_url:
                    st.write("Downloading or loading cached YouTube audio.")
                    saved_path = download_youtube_audio(youtube_url, UPLOAD_DIR / "youtube")
                else:
                    saved_path = save_uploaded_file(uploaded_file)
                st.write("Generating chord predictions.")
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
            if not lab_path.exists():
                raise FileNotFoundError(f"Expected output file was not created: {lab_path}")

            st.session_state["result"] = {
                "lab_path": lab_path,
                "soft_path": output_stem.with_suffix(".npz"),
                "input_kind": input_kind,
                "audio_path": saved_path if input_kind == "audio" else None,
            }
        except Exception as exc:  # noqa: BLE001
            st.error(f"Inference failed: {exc}")

if "result" in st.session_state:
    try:
        render_results(st.session_state["result"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not display results: {exc}")

st.sidebar.header("How to use")
st.sidebar.markdown(
    """
    1. Choose whether your input is audio or symbolic.
    2. Upload a file or enter a YouTube video URL.
    3. Select the backbone and checkpoint family.
    4. Click Run inference.
    5. For audio, press play in the Live player to follow the chords in real time.
    6. Download the generated .lab output.
    """
)
st.sidebar.markdown("This app reuses the repo's inference runtime directly, so it is easy to demo on a local machine.")

#  python -m streamlit run app.py --server.headless true --server.port 8501
