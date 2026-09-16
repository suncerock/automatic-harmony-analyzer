# Automatic Harmony Analyzer

This repository is a compact inference-only chord recognition runtime for audio and MIDI-based symbolic input.

It is built for one purpose: run a pretrained model and write chord predictions as `.lab` files, with optional soft outputs for debugging or downstream processing.

## Quick start

### 1) Install dependencies

```bash
conda env create -f environment.yaml
conda activate chord-recognition
```

### 2) Put the checkpoints in `ckpt/`

The runtime expects pretrained checkpoints in the repository-local `ckpt/` folder.

Use the model family you want:

- audio + `crnn` + `classical`
- audio + `crnn` + `pop`
- audio + `conformer` + `classical`
- audio + `conformer` + `pop`
- symbolic + `crnn`
- symbolic + `conformer`

The repository currently supports the following inferred runtime commands:

```bash
python -m acr infer-a --input-path /path/to/audio.wav --output-path /tmp/out --backbone-type crnn --model-type classical
python -m acr infer-s --input-path /path/to/song.mid --output-path /tmp/out --backbone-type crnn
python -m acr infer-s --input-path /path/to/song.musicxml --output-path /tmp/out --backbone-type crnn
```

You can also use the CLI entrypoint if installed as a package:

```bash
acr infer-a --input-path /path/to/audio.wav --output-path /tmp/out --backbone-type crnn --model-type classical
acr infer-s --input-path /path/to/song.mid --output-path /tmp/out --backbone-type crnn
acr infer-s --input-path /path/to/song.musicxml --output-path /tmp/out --backbone-type crnn
```

## Supported input types

### Audio inference

```bash
python -m acr infer-a \
  --input-path /path/to/audio.wav \
  --output-path /tmp/chords \
  --backbone-type crnn \
  --model-type classical \
  --soft-output
```

This writes:

- `/tmp/chords.lab`
- `/tmp/chords.npz` if `--soft-output` is set

### Symbolic inference (MIDI / MusicXML)

```bash
python -m acr infer-s \
  --input-path /path/to/song.mid \
  --output-path /tmp/chords \
  --backbone-type crnn

python -m acr infer-s \
  --input-path /path/to/song.musicxml \
  --output-path /tmp/chords \
  --backbone-type crnn
```

This accepts MIDI files (`.mid` or `.midi`) and MusicXML files (`.xml` or `.musicxml`). Any other symbolic file type is deliberately rejected.

The command writes:

- `/tmp/chords.lab`
- `/tmp/chords.npz` if `--soft-output` is set

## CLI options

### `infer-a`

- `--input-path`: path to an audio file
- `--output-path`: output stem used for `.lab` and optional `.npz`
- `--backbone-type`: `crnn` or `conformer`
- `--model-type`: `classical` or `pop` for audio checkpoints
- `--soft-output`: save per-frame logits in `.npz`
- `--clip-seconds`: option to process only the first N seconds of the audio

### `infer-s`

- `--input-path`: path to a symbolic file (`.mid`, `.midi`, `.xml`, or `.musicxml`)
- `--output-path`: output stem used for `.lab` and optional `.npz`
- `--backbone-type`: `crnn` or `conformer`
- `--soft-output`: save per-frame logits in `.npz`

## Output format

The runtime produces a `.lab` file in the usual segment format:

```text
0.000000	2.500000	C
2.500000	5.000000	G
```

This is:

- start time in seconds
- end time in seconds
- chord label

If `--soft-output` is enabled, the `.npz` file stores frame-wise predictions from the model for downstream use.

## Notes

- This project is focused on inference only; training code and dataset pipelines are intentionally not included.
- The symbolic path is intentionally strict: it accepts MIDI and MusicXML only, because the model input format is fixed and deterministic.
- For audio inference, the HCQT frontend is computed directly from the raw waveform at runtime.
- Symbolic MusicXML inputs are converted to the same score-grid representation as MIDI before inference, so they share the same runtime contract.

## License

Code: MIT License

Pretrained weights: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — for research and academic use only, not for commercial use.
