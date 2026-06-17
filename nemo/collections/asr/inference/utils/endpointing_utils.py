# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def millisecond_to_frames(millisecond: int, ms_per_timestep: int) -> int:
    """
    Convert milliseconds to frames
    Args:
        millisecond (int): milliseconds to convert
        ms_per_timestep (int): milliseconds per timestep
    Returns:
        int: number of frames
    """
    return (millisecond + ms_per_timestep - 1) // ms_per_timestep


def validate_eou_thresholds(stop_history_eou_ms: int, stop_history_eou_end_ms: int | None) -> None:
    """
    Validate that the end-of-buffer EoU threshold is not weaker than the regular one.

    An end-of-buffer EoU (trailing silence, no following word observed yet) must require at least as
    much silence as a mid-buffer EoU; otherwise it would cut words at the buffer edge more eagerly
    than mid-buffer, defeating its purpose.
    Args:
        stop_history_eou_ms (int): regular (mid-buffer) silence threshold in ms
        stop_history_eou_end_ms (int | None): end-of-buffer silence threshold in ms
    Raises:
        ValueError: if both thresholds are enabled (> 0) and the end threshold is smaller.
    """
    if (
        stop_history_eou_ms is not None
        and stop_history_eou_ms > 0
        and stop_history_eou_end_ms is not None
        and stop_history_eou_end_ms > 0
        and stop_history_eou_end_ms < stop_history_eou_ms
    ):
        raise ValueError(
            f"endpointing.stop_history_eou_end ({stop_history_eou_end_ms} ms) must be >= "
            f"endpointing.stop_history_eou ({stop_history_eou_ms} ms)."
        )


def derive_eou_buffer_size(
    silence_threshold_ms: int,
    tokens_per_frame: int,
    ms_per_timestep: int,
) -> int:
    """
    Derive the EoU label-buffer size (in frames) from the largest silence threshold and the
    per-chunk token count.

    The buffer is refilled once per chunk and slides by `tokens_per_frame` each step, so it must hold
    the longest qualifying silence run we need to detect (`silence_threshold_ms`, i.e. the end-of-buffer
    threshold, which is the larger of the regular and end thresholds) plus a full chunk -- otherwise a
    single update could evict the run before it is evaluated, and the token right after the run (word
    start / late punctuation) would not be visible for the absorb and validity checks.
    Args:
        silence_threshold_ms (int): largest silence threshold to detect (the end-of-buffer threshold), in ms
        tokens_per_frame (int): number of output frames produced per streaming chunk
        ms_per_timestep (int): milliseconds per output frame
    Returns:
        int: EoU label-buffer size in frames
    """
    silence_frames = millisecond_to_frames(silence_threshold_ms, ms_per_timestep)
    return silence_frames + tokens_per_frame


def get_custom_stop_history_eou(
    stop_history_eou: int | None, default_stop_history_eou: int, ms_per_timestep: int
) -> int:
    """
    Get the custom stop history of EOU
    Args:
        stop_history_eou (int): stop history of EOU
        default_stop_history_eou (int): default stop history of EOU
        ms_per_timestep (int): milliseconds per timestep
    Returns:
        int: custom stop history of EOU
    """
    if stop_history_eou is None:
        return default_stop_history_eou
    if stop_history_eou > 0:
        return millisecond_to_frames(stop_history_eou, ms_per_timestep)
    return 0 if stop_history_eou == 0 else -1
