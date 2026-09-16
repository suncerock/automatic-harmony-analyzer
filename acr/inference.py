"""Inference runtime for audio chord recognition."""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.utils import ChordLabelConfig
from src.models.backbone.conformer import Conformer
from src.models.backbone.crnn import CRNN
from src.models.decoder import convert_harte_index_to_hchord, decode_rbp_sequence
from src.models.frontend.classic_pitch import HCQTEncoder
from src.models.model import ACRModel
from acr.hcqt import audio_to_hcqt
from acr.midi import SYMBOLIC_FPM, midi_to_symbolic_array, musicxml_to_symbolic_array

HCQT_SR = 22050
HCQT_HOP_LENGTH = 512
HCQT_MIN_NOTE = 24
HCQT_BINS_PER_OCTAVE = 36
HCQT_N_BINS = 216
HCQT_HARMONICS = [0.5, 1, 2, 3, 4, 5]
HCQT_SOURCE_FPS = HCQT_SR / HCQT_HOP_LENGTH
AUDIO_FPS = 10.0
DEFAULT_MPE_CKPT = _REPO_ROOT / "ckpt" / "pretrained_mpe.ckpt"
DECODE_PITCH_THRESHOLD = 0.4


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        raise ValueError("Audio signal is silent; cannot compute HCQT for inference.")
    return audio / peak


def _maybe_clip_audio(audio_path: Path, clip_seconds: Optional[float]) -> tuple[Path, Optional[Path]]:
    if clip_seconds is None or clip_seconds <= 0:
        return audio_path, None

    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
    clip_frames = int(min(len(audio), sr * clip_seconds))
    if clip_frames <= 0:
        raise ValueError(f"Clip length {clip_seconds} is invalid for audio file {audio_path}.")

    clipped = audio[:clip_frames]
    temp_file = Path(tempfile.NamedTemporaryFile(suffix=audio_path.suffix or ".wav", delete=False).name)
    sf.write(temp_file, clipped, sr)
    return temp_file, temp_file


def _audio_path_to_hcqt(audio_path: Path) -> torch.Tensor:
    audio, _ = librosa.load(str(audio_path), sr=HCQT_SR, mono=True)
    audio = _normalize_audio(audio)
    hcqt = audio_to_hcqt(
        audio,
        sr_audio=HCQT_SR,
        min_note=HCQT_MIN_NOTE,
        bins_per_octave=HCQT_BINS_PER_OCTAVE,
        n_bins=HCQT_N_BINS,
        harmonics=HCQT_HARMONICS,
        hop_length=HCQT_HOP_LENGTH,
    )
    return torch.from_numpy(hcqt).float().unsqueeze(0)


def _downsample_mpe_output(mpe_output: torch.Tensor, source_fps: float, target_fps: float) -> torch.Tensor:
    target_frames = round(mpe_output.shape[-1] * target_fps / source_fps)
    if target_frames <= 0:
        raise ValueError("Downsampled MPE output would have zero frames.")
    return torch.nn.functional.interpolate(
        mpe_output,
        size=(mpe_output.shape[2], target_frames),
        mode="bilinear",
        align_corners=False,
    )


def _coerce_symbolic_input(symbolic_path: Path) -> torch.Tensor:
    if not symbolic_path.exists():
        raise FileNotFoundError(f"Symbolic input file not found: {symbolic_path}.")

    suffix = symbolic_path.suffix.lower()
    if suffix in {".mid", ".midi"}:
        symbolic = midi_to_symbolic_array(symbolic_path)
    elif suffix in {".xml", ".musicxml"}:
        symbolic = musicxml_to_symbolic_array(symbolic_path)
    else:
        raise ValueError("Symbolic input must be a MIDI (.mid/.midi) or MusicXML (.xml/.musicxml) file; non-supported file types are not allowed.")

    tensor = torch.as_tensor(symbolic, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor


def _resolve_output_paths(output_path: Path) -> tuple[Path, Path]:
    base_path = output_path if output_path.suffix == "" else output_path.with_suffix("")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    return base_path.with_suffix(".npz"), base_path.with_suffix(".lab")


def _resolve_chord_ckpt_path(backbone_type: str, model_type: Optional[str], use_mpe: bool) -> Path:
    if use_mpe:
        if model_type is None:
            raise ValueError("--model-type is required for audio inference.")
        chord_ckpt_path = _REPO_ROOT / "ckpt" / f"audio_{backbone_type}_{model_type}.ckpt"
        if not chord_ckpt_path.exists():
            raise FileNotFoundError(f"Audio chord checkpoint not found: {chord_ckpt_path}.")
        return chord_ckpt_path

    symbolic_ckpt_path = _REPO_ROOT / "ckpt" / f"symbolic_{backbone_type}.ckpt"
    if not symbolic_ckpt_path.exists():
        raise FileNotFoundError(f"Symbolic chord checkpoint not found: {symbolic_ckpt_path}.")
    return symbolic_ckpt_path


def _resolve_mpe_ckpt_path() -> Path:
    mpe_ckpt_path = _REPO_ROOT / "ckpt" / "audio_mpe_pretrained.ckpt"
    if not mpe_ckpt_path.exists():
        raise FileNotFoundError(f"MPE checkpoint not found: {mpe_ckpt_path}.")
    return mpe_ckpt_path


def _build_model(backbone_type: str, chord_ckpt_path: Path, mpe_ckpt_path: Path, use_mpe: bool = True) -> ACRModel:
    class DummyLoss(nn.Module):
        def setup_loss(self, *args, **kwargs):
            pass

    if backbone_type == "crnn":
        backbone = CRNN(input_channels=1, freq_dim=72, num_channels=32, num_rnn_layers=4, dropout=0.2)
        label_config = ChordLabelConfig.mcfee_structured(harte_vocab="tetrads")
        pitch_informed_rbp = False
        rbp_informed_chord = True
    elif backbone_type == "conformer":
        backbone = Conformer(freq_dim=72, hidden_size=192, num_hidden_layers=4, intermediate_size=768)
        label_config = ChordLabelConfig.poltronieri_structured()
        pitch_informed_rbp = True
        rbp_informed_chord = False
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type!r}")

    model = ACRModel(
        backbone=backbone,
        loss_fn=DummyLoss(),
        optim_cfg={"optimizer": {}, "scheduler": {}},
        mpe_model=HCQTEncoder() if use_mpe else None,
        mpe_ckpt_path=str(mpe_ckpt_path) if use_mpe else None,
        backbone_ckpt_path=str(chord_ckpt_path),
        model_input_key="x_mpe_input",
        label_config=label_config,
        pitch_informed_rbp=pitch_informed_rbp,
        rbp_informed_chord=rbp_informed_chord,
    )
    model.eval()
    return model


def _group_frames_to_segments(
    chord_strings: List[str],
    fps: float,
    time_offset_frames: int = 0,
) -> Tuple[np.ndarray, List[str]]:
    assert len(chord_strings) > 0, "Input chord_strings must not be empty."

    intervals: List[List[float]] = []
    labels: List[str] = []
    start = time_offset_frames
    current = chord_strings[0]
    T = len(chord_strings)

    for t in range(1, T):
        if chord_strings[t] != current:
            intervals.append([(start + 0) / fps, (t + time_offset_frames) / fps])
            labels.append(current)
            start = t + time_offset_frames
            current = chord_strings[t]

    intervals.append([(start + 0) / fps, (T + time_offset_frames) / fps])
    labels.append(current)
    return np.array(intervals, dtype=np.float64), labels


def _decode_predictions_to_chord_strings(model: ACRModel, predictions: dict[str, np.ndarray]) -> List[str]:
    if "pitch" in predictions:
        predictions["pc"] = predictions.pop("pitch").T

    if "chord" in predictions:
        vocab = model.label_config.harte_vocab
        chord_indices = np.argmax(predictions["chord"], axis=0)
        return [
            convert_harte_index_to_hchord(int(idx), harte_vocab=vocab).export_string()
            for idx in chord_indices
        ]

    if "root" in predictions and "pc" in predictions and "bass" in predictions:
        return decode_rbp_sequence(
            root_logits=predictions["root"],
            pitch_logits=1 / (1 + np.exp(-predictions["pc"].T)),
            bass_logits=predictions["bass"],
            pitch_threshold=DECODE_PITCH_THRESHOLD,
        )

    raise ValueError("Predictions must contain either 'chord' or all of 'root', 'pc', and 'bass'.")


def _write_lab_file(lab_path: Path, chord_strings: List[str], fps: float, time_offset_frames: int = 0) -> None:
    intervals, labels = _group_frames_to_segments(chord_strings, fps, time_offset_frames=time_offset_frames)
    with open(lab_path, "w") as f:
        for (start, end), label in zip(intervals, labels):
            f.write(f"{start:.6f}\t{end:.6f}\t{label}\n")


def _infer_symbolic(model: ACRModel, symbolic_path: Path, output_path: Path, save_soft_output: bool = False) -> None:
    symbolic_input = _coerce_symbolic_input(symbolic_path)
    with torch.no_grad():
        embedding = model.backbone(symbolic_input)
        predictions = model._forward_heads(embedding)
        predictions = {k: v.squeeze(0).float().cpu().numpy() for k, v in predictions.items()}

    npz_path, lab_path = _resolve_output_paths(output_path)
    if save_soft_output:
        np.savez_compressed(npz_path, **predictions)

    frame_chord_strings = _decode_predictions_to_chord_strings(model, predictions)
    _write_lab_file(lab_path, frame_chord_strings, SYMBOLIC_FPM, time_offset_frames=int(SYMBOLIC_FPM))

    print(f"Symbolic input: {symbolic_path}")
    if save_soft_output:
        print(f"NPZ output   : {npz_path}")
    print(f"LAB output    : {lab_path}")
    print("Done.")


def _infer_audio(model: ACRModel, audio_path: Path, output_path: Path, save_soft_output: bool = False, clip_seconds: Optional[float] = None) -> None:
    if model.mpe_model is None:
        raise ValueError("Audio inference requires an MPE front-end model.")

    source_audio_path, temp_audio_path = _maybe_clip_audio(audio_path, clip_seconds)
    cleanup_needed = temp_audio_path is not None
    try:
        hcqt = _audio_path_to_hcqt(source_audio_path)
        hcqt = torch.log1p(hcqt * 1000)

        with torch.no_grad():
            mpe_output = model._run_mpe_stage(hcqt)
            chord_input = _downsample_mpe_output(mpe_output, source_fps=HCQT_SOURCE_FPS, target_fps=AUDIO_FPS)
            embedding = model.backbone(chord_input)
            predictions = model._forward_heads(embedding)
            predictions = {k: v.squeeze(0).float().cpu().numpy() for k, v in predictions.items()}

        npz_path, lab_path = _resolve_output_paths(output_path)
        if save_soft_output:
            np.savez_compressed(npz_path, **predictions)

        frame_chord_strings = _decode_predictions_to_chord_strings(model, predictions)
        _write_lab_file(lab_path, frame_chord_strings, AUDIO_FPS, time_offset_frames=0)

        print(f"Audio path    : {audio_path}")
        if clip_seconds is not None:
            print(f"Clip seconds  : {clip_seconds}")
        if save_soft_output:
            print(f"NPZ output   : {npz_path}")
        print(f"LAB output    : {lab_path}")
        print("Done.")
    finally:
        if cleanup_needed and temp_audio_path is not None:
            temp_audio_path.unlink(missing_ok=True)


def run_inference(
    model: ACRModel,
    output_path: Path,
    audio_path: Optional[Path] = None,
    symbolic_path: Optional[Path] = None,
    save_soft_output: bool = False,
    clip_seconds: Optional[float] = None,
) -> None:
    if audio_path is not None and symbolic_path is not None:
        raise ValueError("Specify either an audio input or a symbolic input, not both.")
    if audio_path is None and symbolic_path is None:
        raise ValueError("Either --audio-path or --symbolic-path must be provided.")

    if symbolic_path is not None:
        _infer_symbolic(model, symbolic_path, output_path, save_soft_output=save_soft_output)
        return

    _infer_audio(model, audio_path, output_path, save_soft_output=save_soft_output, clip_seconds=clip_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chord recognition inference on either an audio file or a symbolic matrix.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio-path", type=Path, help="Path to the input audio file.")
    input_group.add_argument("--symbolic-path", type=Path, help="Path to a symbolic input MIDI file (.mid/.midi) or MusicXML file (.xml/.musicxml).")
    parser.add_argument("--output-path", required=True, type=Path, help="Output file stem; the script writes both .npz and .lab files using this path.")
    parser.add_argument("--backbone-type", choices=["crnn", "conformer"], default="crnn", help="Chord-model backbone family to use (default: crnn).")
    parser.add_argument("--model-type", choices=["classical", "pop"], default=None, help="Model family to use for the audio checkpoint selection.")
    parser.add_argument("--soft-output", action="store_true", help="Save .npz soft outputs in addition to .lab decoded segments.")
    parser.add_argument("--clip-seconds", type=float, default=None, help="Optional: process only the first N seconds of the input audio for quick local testing.")
    args = parser.parse_args()

    use_mpe = args.audio_path is not None
    chord_ckpt_path = _resolve_chord_ckpt_path(args.backbone_type, args.model_type, use_mpe=use_mpe)
    mpe_ckpt_path = _resolve_mpe_ckpt_path() if use_mpe else Path("unused")

    print(f"Backbone     : {args.backbone_type}")
    if args.model_type is not None:
        print(f"Trained on   : {args.model_type}")
    if use_mpe:
        print("MPE ckpt     : auto-selected")
    else:
        print("MPE ckpt     : skipped (symbolic input)")

    model = _build_model(
        backbone_type=args.backbone_type,
        chord_ckpt_path=chord_ckpt_path,
        mpe_ckpt_path=mpe_ckpt_path,
        use_mpe=use_mpe,
    )

    run_inference(
        model=model,
        audio_path=args.audio_path,
        symbolic_path=args.symbolic_path,
        output_path=args.output_path,
        save_soft_output=args.soft_output,
        clip_seconds=args.clip_seconds,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
