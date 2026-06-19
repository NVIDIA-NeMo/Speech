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


from nemo.collections.asr.inference.streaming.state.state import StreamingState


class CacheAwareStreamingState(StreamingState):
    """
    State of the cache aware CTC/RNNT streaming pipelines
    """

    def __init__(self):
        """
        Initialize the CacheAwareStreamingState
        """
        super().__init__()
        self._additional_params_reset()

    def reset(self) -> None:
        """
        Reset the state
        """
        super().reset()
        self._additional_params_reset()

    def _additional_params_reset(self) -> None:
        """
        Reset non-inherited parameters
        """
        # label_buffer will be used to detect EoU
        self.label_buffer = []
        self.label_buffer_size = 0
        self.offset = 0

        # Number of label-frames shifted into the buffer so far. The rightmost buffer slot
        # (index label_buffer_size - 1) maps to the global frame `label_buffer_global_end - 1`.
        self.label_buffer_global_end = 0
        # Global frame index from which to start searching for the next EoU. Advanced past a
        # detected EoU so that the same silence is not re-detected as the buffer slides.
        self.eou_search_start_global = 0

        # Tokens that survive past a detected EoU (the start of the next utterance). They are
        # temporarily stashed here while the finalized portion is decoded, then restored.
        self._carryover_tokens = []
        self._carryover_timesteps = []
        self._carryover_confidences = []

    def set_offset(self, offset: int) -> None:
        """
        Set the offset
        Args:
            offset: (int) offset
        """
        self.offset = offset

    def setup_label_buffer(self, label_buffer_size: int, blank_id: int) -> None:
        """
        Set up the label buffer
        Args:
            label_buffer_size: (int) size of the label buffer
            blank_id: (int) blank id
        """
        self.label_buffer_size = label_buffer_size
        self.label_buffer = [blank_id] * self.label_buffer_size

    def update_label_buffer(self, labels: list[int]) -> None:
        """
        Update the label buffer with the labels of the current chunk and advance the global frame
        counter that maps buffer positions to global timesteps.
        Args:
            labels: (list[int]) list of labels
        """
        shift = len(labels)
        if shift == 0:
            return
        self.label_buffer_global_end += shift
        if shift >= len(self.label_buffer):
            self.label_buffer[:] = labels[-len(self.label_buffer) :]
            return
        self.label_buffer[:-shift] = self.label_buffer[shift:].copy()
        self.label_buffer[-shift:] = labels.copy()

    def get_label_buffer(self) -> list[int]:
        """
        Get the current label buffer
        Returns:
            list[int]: current state of the label buffer
        """
        return self.label_buffer.copy()

    def buffer_local_to_global(self, local_idx: int) -> int:
        """
        Convert a label-buffer-local index to a global frame index.
        Args:
            local_idx: (int) index within the label buffer
        Returns:
            int: corresponding global frame index
        """
        return self.label_buffer_global_end - self.label_buffer_size + local_idx

    def get_local_search_start(self) -> int:
        """
        Convert the stored global EoU search start into a label-buffer-local index, clamped to the
        current buffer window.
        Returns:
            int: buffer-local index to start searching for the next EoU
        """
        left_edge = self.label_buffer_global_end - self.label_buffer_size
        return max(0, self.eou_search_start_global - left_edge)

    def set_eou_search_start(self, global_frame: int) -> None:
        """
        Set the global frame index from which to start searching for the next EoU.
        Args:
            global_frame: (int) global frame index
        """
        self.eou_search_start_global = global_frame

    def prepare_finalize(self, resume_global_frame: int, is_token_start_of_word=None) -> None:
        """
        Split the accumulated tokens at `resume_global_frame`. Tokens before it (including absorbed
        punctuation/language tokens) stay in the state to be finalized; tokens at or after it are the
        next utterance and are stashed as carryover to be restored after the finalized portion is
        decoded and cleaned up.

        A new utterance must begin at a word start. Late tokens (sentence punctuation, language tokens,
        or a word's continuation sub-tokens) often share the resume frame with the next word -- the dense
        EoU label buffer keeps only one label per frame, so the detector cannot see them, and the strict
        timestamp split would push them into the carryover (causing punct-leading segments or mid-word
        splits). So any leading non-word-start tokens at the head of the carryover are moved back into the
        finalized (previous) utterance.
        Args:
            resume_global_frame: (int) global frame index marking the start of the next utterance
            is_token_start_of_word: (Callable[[int], bool] | None) predicate identifying word-start
                tokens; leading non-word-start carryover tokens are moved back into the finalized portion.
        """
        k = 0
        n = len(self.timesteps)
        while k < n and self.timesteps[k] < resume_global_frame:
            k += 1
        # The next utterance must start at a word boundary: keep leading punctuation / language tokens /
        # word-continuation sub-tokens with the finalized utterance.
        if is_token_start_of_word is not None:
            while k < n and not is_token_start_of_word(self.tokens[k]):
                k += 1

        self._carryover_tokens = self.tokens[k:]
        self._carryover_timesteps = self.timesteps[k:]
        self._carryover_confidences = self.confidences[k:]

        self.tokens = self.tokens[:k]
        self.timesteps = self.timesteps[:k]
        self.confidences = self.confidences[:k]

    def restore_carryover(self) -> None:
        """
        Restore the tokens that survived past the EoU as the start of the next utterance. Must be
        called after `cleanup_after_eou` (which clears the finalized tokens).
        """
        self.tokens = self._carryover_tokens
        self.timesteps = self._carryover_timesteps
        self.confidences = self._carryover_confidences
        self._carryover_tokens = []
        self._carryover_timesteps = []
        self._carryover_confidences = []

    def update_state(self, completed_output: dict, eou_detected: bool) -> None:
        """
        Update the state with the completed output
        Args:
            completed_output: (dict) completed output
            eou_detected: (bool) is EoU detected
        """

        if len(completed_output) == 0 or len(completed_output["tokens"]) == 0:
            return

        timesteps = completed_output["timesteps"]
        for i, t in enumerate(timesteps):
            timesteps[i] = t + self.global_offset

        # we will not perform overlap aware merging of the tokens for CacheAware Models
        # It is too error-prone to do this in the streaming mode -> skip=0
        self._update_state(completed_output, skip=0)
        self.eou_detected_before = eou_detected
