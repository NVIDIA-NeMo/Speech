# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from nemo.collections.speechlm2.parts.ctc_timestamp_utils import _banded_ctc_search


def search_streams(log_probs, labels, state_lengths, blank_id, columns, speaker_probs, weight, frame_lengths=None):
    """Adapt padded test fixtures to independent banded searches with blank at column zero."""
    paths, scores = [], []
    order = [blank_id] + [i for i in range(log_probs.shape[-1]) if i != blank_id]
    for row, length in enumerate(state_lengths.tolist()):
        logp = log_probs if log_probs.ndim == 2 else log_probs[row]
        frames = logp.shape[0] if frame_lengths is None else int(frame_lengths[row])
        target = np.array([order.index(i) for i in labels[row, :length].tolist()])
        prior = None
        if speaker_probs is not None:
            activity = speaker_probs if speaker_probs.ndim == 2 else speaker_probs[row]
            column = int(columns[row][columns[row] >= 0][0])
            prior = weight * activity[:frames, column].clamp_min(1e-6).log().numpy()
        path, score, _ = _banded_ctc_search(logp[:frames, order].numpy(), target, np.arange(frames), prior)
        paths.append(torch.from_numpy(path))
        scores.append(score)
    return paths, scores
