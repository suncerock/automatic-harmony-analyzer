import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_a_parser = subparsers.add_parser(
        "infer-a",
        help="Run audio chord recognition inference on a single audio file.",
    )
    infer_a_parser.add_argument("--input-path", required=True, type=Path, help="Path to the input audio file.")
    infer_a_parser.add_argument("--output-path", required=True, type=Path, help="Output file stem for .lab and optional .npz outputs.")
    infer_a_parser.add_argument("--backbone-type", choices=["crnn", "conformer"], default="crnn", help="Chord model backbone family.")
    infer_a_parser.add_argument("--model-type", choices=["classical", "pop"], default=None, help="Checkpoint family to use.")
    infer_a_parser.add_argument("--soft-output", action="store_true", help="Save .npz soft outputs in addition to the .lab output.")
    infer_a_parser.add_argument("--clip-seconds", type=float, default=None, help="Optional: process only the first N seconds of the input audio.")

    infer_s_parser = subparsers.add_parser(
        "infer-s",
        help="Run symbolic chord recognition inference on a matrix-like input.",
    )
    infer_s_parser.add_argument("--input-path", required=True, type=Path, help="Path to a symbolic input MIDI file (.mid/.midi) or MusicXML file (.xml/.musicxml).")
    infer_s_parser.add_argument("--output-path", required=True, type=Path, help="Output file stem for .lab and optional .npz outputs.")
    infer_s_parser.add_argument("--backbone-type", choices=["crnn", "conformer"], default="crnn", help="Chord model backbone family.")
    infer_s_parser.add_argument("--model-type", choices=["classical", "pop"], default=None, help="Checkpoint family to use.")
    infer_s_parser.add_argument("--soft-output", action="store_true", help="Save .npz soft outputs in addition to the .lab output.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command not in {"infer-a", "infer-s"}:
        parser.error("Only 'infer-a' and 'infer-s' are supported.")

    if args.command == "infer-a" and args.model_type is None:
        parser.error("--model-type is required for audio inference.")

    from acr.inference import main as runtime_main

    runtime_argv = [
        "acr.inference",
        "--output-path",
        str(args.output_path),
        "--backbone-type",
        str(args.backbone_type),
    ]
    if args.model_type is not None:
        runtime_argv.extend(["--model-type", str(args.model_type)])
    if args.command == "infer-a":
        runtime_argv.extend(["--audio-path", str(args.input_path)])
        if args.clip_seconds is not None:
            runtime_argv.extend(["--clip-seconds", str(args.clip_seconds)])
    else:
        runtime_argv.extend(["--symbolic-path", str(args.input_path)])

    if args.soft_output:
        runtime_argv.append("--soft-output")

    old_argv = sys.argv[:]
    try:
        sys.argv = runtime_argv
        runtime_main()
    finally:
        sys.argv = old_argv
    return 0
