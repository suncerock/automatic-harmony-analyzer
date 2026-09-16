"""MIDI/MusicXML-to-symbolic conversion helpers used by the inference runtime."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np


SYMBOLIC_FPM = 20.0


def _score_measure_frame(measure_index: int, offset_within_measure: float, measure_length_quarters: float) -> float:
    if measure_length_quarters <= 0:
        raise ValueError("A measure with non-positive quarter length was encountered in the symbolic score input.")
    return measure_index * SYMBOLIC_FPM + (offset_within_measure / measure_length_quarters) * SYMBOLIC_FPM


def _note_events_to_binary_mpe_array(
    note_events: List[tuple[float, float, int]],
    times: np.ndarray,
    min_note: int,
    n_pitch_bins: int,
) -> np.ndarray:
    pitch_bins = np.arange(min_note, min_note + n_pitch_bins)
    mpe_array = np.zeros((n_pitch_bins, len(times)), dtype=np.float32)

    for start_frame, end_frame, pitch in note_events:
        start_idx = int(np.clip(int(round(start_frame)), 0, len(times) - 1))
        end_idx = int(np.clip(int(round(end_frame)), 0, len(times) - 1))
        pitch_idx = int(np.clip(int(np.argmin(np.abs(pitch - pitch_bins))), 0, n_pitch_bins - 1))
        if end_idx > start_idx:
            mpe_array[pitch_idx, start_idx:end_idx] = 1.0
        elif end_idx == start_idx and start_idx < len(times):
            mpe_array[pitch_idx, start_idx] = 1.0

    return mpe_array


def _score_note_events(score) -> List[tuple[float, float, int]]:
    note_events: List[tuple[float, float, int]] = []
    for part in score.parts:
        measures = part.getElementsByClass("Measure")
        for measure_index, measure in enumerate(measures):
            measure_length_quarters = float(measure.duration.quarterLength)
            if measure_length_quarters <= 0:
                raise ValueError(f"Measure {measure_index + 1} has a non-positive quarter length in the symbolic score input.")
            for element in measure.recurse().notes:
                if getattr(element, "isRest", False):
                    continue
                start_offset_quarters = float(element.offset)
                duration_quarter = float(element.duration.quarterLength)
                if duration_quarter <= 0:
                    continue
                for pitch in getattr(element, "pitches", [element]):
                    start_frame = _score_measure_frame(measure_index, start_offset_quarters, measure_length_quarters)
                    end_frame = _score_measure_frame(
                        measure_index,
                        start_offset_quarters + duration_quarter,
                        measure_length_quarters,
                    )
                    note_events.append((start_frame, end_frame, int(pitch.midi)))
    return note_events


def _symbolic_array_from_note_events(
    note_events: List[tuple[float, float, int]],
    num_measures: int | None = None,
) -> np.ndarray:
    if not note_events:
        return np.zeros((72, 1), dtype=np.float32)

    if num_measures is None:
        max_end_frame = max(end for _, end, _ in note_events)
        total_frames = max(1, int(math.ceil(max_end_frame)) + 1)
    else:
        total_frames = max(1, int(num_measures * SYMBOLIC_FPM))

    times = np.arange(0, total_frames, dtype=np.float64)
    return _note_events_to_binary_mpe_array(note_events, times, min_note=24, n_pitch_bins=72)


def midi_to_symbolic_array(midi_path: Path) -> np.ndarray:
    from music21 import converter

    score = converter.parse(str(midi_path))
    num_measures = len(score.parts[0].getElementsByClass("Measure")) if score.parts else 0
    return _symbolic_array_from_note_events(_score_note_events(score), num_measures=num_measures or None)


def musicxml_to_symbolic_array(musicxml_path: Path) -> np.ndarray:
    from music21 import converter

    score = converter.parse(str(musicxml_path))
    num_measures = len(score.parts[0].getElementsByClass("Measure")) if score.parts else 0
    return _symbolic_array_from_note_events(_score_note_events(score), num_measures=num_measures or None)
