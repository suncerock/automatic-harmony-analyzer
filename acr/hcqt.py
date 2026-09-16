"""HCQT feature extraction helpers used by the inference runtime."""

from __future__ import annotations

from typing import List

import librosa
import numpy as np


def audio_to_hcqt(
    audio,
    sr_audio,
    min_note=21,
    bins_per_octave=36,
    n_bins=252,
    harmonics=(0.5, 1, 2, 3, 4, 5),
    hop_length=512,
):
    """Compute a Harmonic CQT for the given mono audio array."""
    fmin = float(librosa.midi_to_hz(min_note))
    tuning_est = librosa.estimate_tuning(y=audio)
    fmin_tuned = fmin * 2 ** (tuning_est / bins_per_octave)

    hcqt_list = []
    min_time_frames = float("inf")
    for h in harmonics:
        A_m = np.abs(
            librosa.cqt(
                y=audio,
                sr=sr_audio,
                fmin=h * fmin_tuned,
                hop_length=hop_length,
                bins_per_octave=bins_per_octave,
                n_bins=n_bins,
                tuning=0.0,
            )
        )
        hcqt_list.append(A_m)
        min_time_frames = min(min_time_frames, A_m.shape[1])

    hcqt_list = [A[:, :min_time_frames] for A in hcqt_list]
    return np.stack(hcqt_list, axis=0)
