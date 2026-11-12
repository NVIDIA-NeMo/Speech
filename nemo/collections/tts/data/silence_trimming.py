# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-source waveform preprocessing for single-utterance Lhotse TTS examples."""

from dataclasses import replace
from io import BytesIO

import soundfile as sf
from lhotse import CutSet, Recording

from nemo.collections.tts.parts.utils.silence_trimming import silence_trim_config, trim_long_silences


def trim_tts_silence(cuts: CutSet, load_cached_codes: bool) -> CutSet:
    """Trim tagged target recordings before collation and online code extraction.

    Untagged cuts are returned untouched. Dialogue timelines are intentionally
    unsupported: independently trimming their channels would desynchronize turns.
    Cached codes on the target timeline are rejected when they could be consumed
    alongside changed audio. Context recordings and cached context codes are preserved.
    """
    result = []
    for cut in cuts:
        config = silence_trim_config((cut.custom or {}).get("silence_trim"))
        if config is None:
            result.append(cut)
            continue
        if str(getattr(cut, "task", "tts")).lower() != "tts" or len(cut.supervisions) != 1:
            raise ValueError(f"{cut.id}: silence_trim requires a single-supervision TTS cut")
        if load_cached_codes and any(cut.has_custom(key) for key in ("target_codes", "source_codes")):
            raise ValueError(f"{cut.id}: silence_trim requires online codes; disable load_cached_codes_if_available")
        if not cut.has_custom("target_audio"):
            raise ValueError(f"{cut.id}: silence_trim requires target_audio")

        # Copy metadata before edits: SHAR examples may be repeated by the sampler.
        custom = dict(cut.custom)
        target = cut.target_audio
        audio = target.load_audio().squeeze(0)
        trimmed = trim_long_silences(audio, target.sampling_rate, config)
        if trimmed.removed:
            target = _memory_recording(trimmed.audio, target.sampling_rate, target.id)
            custom["target_audio"] = target
            supervision = cut.supervisions[0]
            start = trimmed.map_time(supervision.start, target.sampling_rate)
            end = trimmed.map_time(supervision.end, target.sampling_rate)
            alignment = None
            if supervision.alignment is not None:
                alignment = {
                    key: [
                        item._replace(
                            start=trimmed.map_time(item.start, target.sampling_rate),
                            duration=trimmed.map_time(item.end, target.sampling_rate)
                            - trimmed.map_time(item.start, target.sampling_rate),
                        )
                        for item in items
                    ]
                    for key, items in supervision.alignment.items()
                }
            cut = replace(
                cut,
                start=0.0,
                duration=target.duration,
                recording=target,
                custom=custom,
                supervisions=[
                    replace(
                        supervision, recording_id=target.id, start=start, duration=end - start, alignment=alignment
                    )
                ],
            )
        # Never expose pre-trim arrays on the shortened target timeline.
        for key in ("target_codes", "source_codes"):
            custom.pop(key, None)
        result.append(replace(cut, custom=custom))
    return CutSet.from_cuts(result)


def _memory_recording(audio, sample_rate: int, recording_id: str) -> Recording:
    # FLOAT WAV is lossless for float32 and avoids PCM16 quantization on every training read.
    buffer = BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="FLOAT")
    return Recording.from_bytes(buffer.getvalue(), recording_id=recording_id)
