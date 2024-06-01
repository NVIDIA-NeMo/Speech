# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import os
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.collections.asr.parts.utils.audio_utils import get_samples
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import LengthsType, MelSpectrogramType, NeuralType

# Minimum number of tokens required to assign a LCS merge step, otherwise ignore and
# select all i-1 and ith buffer tokens to merge.
MIN_MERGE_SUBSEQUENCE_LEN = 1


def print_alignment(alignment):
    """
    Print an alignment matrix of the shape (m + 1, n + 1)

    Args:
        alignment: An integer alignment matrix of shape (m + 1, n + 1)
    """
    m = len(alignment)
    if m > 0:
        n = len(alignment[0])
        for i in range(m):
            for j in range(n):
                if j == 0:
                    print(f"{i:4d} |", end=" ")
                print(f"{alignment[i][j]}", end=" ")
            print()


def write_lcs_alignment_to_pickle(alignment, filepath, extras=None):
    """
    Writes out the LCS alignment to a file, along with any extras provided.

    Args:
        alignment: An alignment matrix of shape [m + 1, n + 1]
        filepath: str filepath
        extras: Optional dictionary of items to preserve.
    """
    if extras is None:
        extras = {}

    extras['alignment'] = alignment
    torch.save(extras, filepath)


def longest_common_subsequence_merge(X, Y, filepath=None):
    """
    Longest Common Subsequence merge algorithm for aligning two consecutive buffers.

    Base alignment construction algorithm is Longest Common Subsequence (reffered to as LCS hear after)

    LCS Merge algorithm looks at two chunks i-1 and i, determins the aligned overlap at the
    end of i-1 and beginning of ith chunk, and then clips the subsegment of the ith chunk.

    Assumption is that the two chunks are consecutive chunks, and there exists at least small overlap acoustically.

    It is a sub-word token merge algorithm, operating on the abstract notion of integer ids representing the subword ids.
    It is independent of text or character encoding.

    Since the algorithm is merge based, and depends on consecutive buffers, the very first buffer is processes using
    the "middle tokens" algorithm.

    It requires a delay of some number of tokens such that:
        lcs_delay = math.floor(((total_buffer_in_secs - chunk_len_in_sec)) / model_stride_in_secs)

    Total cost of the model is O(m_{i-1} * n_{i}) where (m, n) represents the number of subword ids of the buffer.

    Args:
        X: The subset of the previous chunk i-1, sliced such X = X[-(lcs_delay * max_steps_per_timestep):]
            Therefore there can be at most lcs_delay * max_steps_per_timestep symbols for X, preserving computation.
        Y: The entire current chunk i.
        filepath: Optional filepath to save the LCS alignment matrix for later introspection.

    Returns:
        A tuple containing -
            - i: Start index of alignment along the i-1 chunk.
            - j: Start index of alignment along the ith chunk.
            - slice_len: number of tokens to slice off from the ith chunk.
        The LCS alignment matrix itself (shape m + 1, n + 1)
    """
    # LCSuff is the table with zero
    # value initially in each cell
    m = len(X)
    n = len(Y)
    LCSuff = [[0 for k in range(n + 1)] for l in range(m + 1)]

    # To store the length of
    # longest common substring
    result = 0
    result_idx = [0, 0, 0]  # Contains (i, j, slice_len)

    # Following steps to build
    # LCSuff[m+1][n+1] in bottom up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                LCSuff[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1

                if result <= LCSuff[i][j]:
                    result = LCSuff[i][j]  # max(result, LCSuff[i][j])
                    result_idx = [i, j, result]

            else:
                LCSuff[i][j] = 0

    # Check if perfect alignment was found or not
    # Perfect alignment is found if :
    # Longest common subsequence extends to the final row of of the old buffer
    # This means that there exists a diagonal LCS backtracking to the beginning of the new buffer
    i, j = result_idx[0:2]
    is_complete_merge = i == m

    # Perfect alignment was found, slice eagerly
    if is_complete_merge:
        length = result_idx[-1]

        # In case the LCS was incomplete - missing a few tokens at the beginning
        # Perform backtrack to find the origin point of the slice (j) and how many tokens should be sliced
        while length >= 0 and i > 0 and j > 0:
            # Alignment exists at the required diagonal
            if LCSuff[i - 1][j - 1] > 0:
                length -= 1
                i, j = i - 1, j - 1

            else:
                # End of longest alignment
                i, j, length = i - 1, j - 1, length - 1
                break

    else:
        # Expand hypothesis to catch partial mismatch

        # There are 3 steps for partial mismatch in alignment
        # 1) Backward search for leftmost LCS
        # 2) Greedy expansion of leftmost LCS to the right
        # 3) Backtrack final leftmost expanded LCS to find origin point of slice

        # (1) Backward search for Leftmost LCS
        # This is required for cases where multiple common subsequences exist
        # We only need to select the leftmost one - since that corresponds
        # to the last potential subsequence that matched with the new buffer.
        # If we just chose the LCS (and not the leftmost LCS), then we can potentially
        # slice off major sections of text which are repeated between two overlapping buffers.

        # backward linear search for leftmost j with longest subsequence
        max_j = 0
        max_j_idx = n

        i_partial = m  # Starting index of i for partial merge
        j_partial = -1  # Index holder of j for partial merge
        j_skip = 0  # Number of tokens that were skipped along the diagonal
        slice_count = 0  # Number of tokens that should be sliced

        # Select leftmost LCS
        for i_idx in range(m, -1, -1):  # start from last timestep of old buffer
            for j_idx in range(0, n + 1):  # start from first token from new buffer
                # Select the longest LCSuff, while minimizing the index of j (token index for new buffer)
                if LCSuff[i_idx][j_idx] > max_j and j_idx <= max_j_idx:
                    max_j = LCSuff[i_idx][j_idx]
                    max_j_idx = j_idx

                    # Update the starting indices of the partial merge
                    i_partial = i_idx
                    j_partial = j_idx

        # EARLY EXIT (if max subsequence length <= MIN merge length)
        # Important case where there is long silence
        # The end of one buffer will have many blank tokens, the beginning of new buffer may have many blank tokens
        # As such, LCS will potentially be from the region of actual tokens.
        # This can be detected as the max length of the suffix in LCS
        # If this max length of the leftmost suffix is less than some margin, avoid slicing all together.
        if max_j <= MIN_MERGE_SUBSEQUENCE_LEN:
            # If the number of partiial tokens to be deleted are less than the minimum,
            # dont delete any tokens at all.

            i = i_partial
            j = 0
            result_idx[-1] = 0

        else:
            # Some valid long partial alignment was found
            # (2) Expand this alignment along the diagonal *downwards* towards the end of the old buffer
            # such that i_partial = m + 1.
            # This is a common case where due to LSTM state or reduced buffer size, the alignment breaks
            # in the middle but there are common subsequences between old and new buffers towards the end
            # We can expand the current leftmost LCS in a diagonal manner downwards to include such potential
            # merge regions.

            # Expand current partial subsequence with co-located tokens
            i_temp = i_partial + 1  # diagonal next i
            j_temp = j_partial + 1  # diagonal next j

            j_exp = 0  # number of tokens to expand along the diagonal
            j_skip = 0  # how many diagonals didnt have the token. Incremented by 1 for every row i

            for i_idx in range(i_temp, m + 1):  # walk from i_partial + 1 => m + 1
                j_any_skip = 0  # If the diagonal element at this location is not found, set to 1
                # j_any_skip expands the search space one place to the right
                # This allows 1 diagonal misalignment per timestep i (and expands the search for the next timestep)

                # walk along the diagonal corresponding to i_idx, plus allowing diagonal skips to occur
                # diagonal elements may not be aligned due to ASR model predicting
                # incorrect token in between correct tokens
                for j_idx in range(j_temp, j_temp + j_skip + 1):
                    if j_idx < n + 1:
                        if LCSuff[i_idx][j_idx] == 0:
                            j_any_skip = 1
                        else:
                            j_exp = 1 + j_skip + j_any_skip

                # If the diagonal element existed, dont expand the search space,
                # otherwise expand the search space 1 token to the right
                j_skip += j_any_skip

                # Move one step to the right for the next diagonal j corresponding to i
                j_temp += 1

            # reset j_skip, augment j_partial with expansions
            j_skip = 0
            j_partial += j_exp

            # (3) Given new leftmost j_partial with expansions, backtrack the partial alignments
            # counting how many diagonal skips occured to compute slice length
            # as well as starting point of slice.

            # Partial backward trace to find start of slice
            while i_partial > 0 and j_partial > 0:
                if LCSuff[i_partial][j_partial] == 0:
                    # diagonal skip occured, move j to left 1 extra time
                    j_partial -= 1
                    j_skip += 1

                if j_partial > 0:
                    # If there are more steps to be taken to the left, slice off the current j
                    # Then loop for next (i, j) diagonal to the upper left
                    slice_count += 1
                    i_partial -= 1
                    j_partial -= 1

            # Recompute total slice length as slice count along diagonal
            # plus the number of diagonal skips
            i = max(0, i_partial)
            j = max(0, j_partial)
            result_idx[-1] = slice_count + j_skip

    # Set the value of i and j
    result_idx[0] = i
    result_idx[1] = j

    if filepath is not None:
        extras = {
            "is_complete_merge": is_complete_merge,
            "X": X,
            "Y": Y,
            "slice_idx": result_idx,
        }
        write_lcs_alignment_to_pickle(LCSuff, filepath=filepath, extras=extras)
        print("Wrote alignemnt to :", filepath)

    return result_idx, LCSuff


def lcs_alignment_merge_buffer(buffer, data, delay, model, max_steps_per_timestep: int = 5, filepath: str = None):
    """
    Merges the new text from the current frame with the previous text contained in the buffer.

    The alignment is based on a Longest Common Subsequence algorithm, with some additional heuristics leveraging
    the notion that the chunk size is >= the context window. In case this assumptio is violated, the results of the merge
    will be incorrect (or at least obtain worse WER overall).
    """
    # If delay timesteps is 0, that means no future context was used. Simply concatenate the buffer with new data.
    if delay < 1:
        buffer += data
        return buffer

    # If buffer is empty, simply concatenate the buffer and data.
    if len(buffer) == 0:
        buffer += data
        return buffer

    # Prepare a subset of the buffer that will be LCS Merged with new data
    search_size = int(delay * max_steps_per_timestep)
    buffer_slice = buffer[-search_size:]

    # Perform LCS Merge
    lcs_idx, lcs_alignment = longest_common_subsequence_merge(buffer_slice, data, filepath=filepath)

    # Slice off new data
    # i, j, slice_len = lcs_idx
    slice_idx = lcs_idx[1] + lcs_idx[-1]  # slice = j + slice_len
    data = data[slice_idx:]

    # Concat data to buffer
    buffer += data
    return buffer


def inplace_buffer_merge(buffer, data, timesteps, model):
    """
    Merges the new text from the current frame with the previous text contained in the buffer.

    The alignment is based on a Longest Common Subsequence algorithm, with some additional heuristics leveraging
    the notion that the chunk size is >= the context window. In case this assumptio is violated, the results of the merge
    will be incorrect (or at least obtain worse WER overall).
    """
    # If delay timesteps is 0, that means no future context was used. Simply concatenate the buffer with new data.
    if timesteps < 1:
        buffer += data
        return buffer

    # If buffer is empty, simply concatenate the buffer and data.
    if len(buffer) == 0:
        buffer += data
        return buffer

    # Concat data to buffer
    buffer += data
    return buffer


class StreamingFeatureBufferer:
    """
    Class to append each feature frame to a buffer and return an array of buffers.
    This class is designed to perform a real-life streaming decoding where only a single chunk
    is provided at each step of a streaming pipeline.
    """

    def __init__(self, asr_model, chunk_size, buffer_size):
        '''
        Args:
            asr_model:
                Reference to the asr model instance for which the feature needs to be created
            chunk_size (float):
                Duration of the new chunk of audio
            buffer_size (float):
                Size of the total audio in seconds maintained in the buffer
        '''

        self.NORM_CONSTANT = 1e-5
        if hasattr(asr_model.preprocessor, 'log') and asr_model.preprocessor.log:
            self.ZERO_LEVEL_SPEC_DB_VAL = -16.635  # Log-Melspectrogram value for zero signal
        else:
            self.ZERO_LEVEL_SPEC_DB_VAL = 0.0
        self.asr_model = asr_model
        self.sr = asr_model.cfg.sample_rate
        self.model_normalize_type = asr_model.cfg.preprocessor.normalize
        self.chunk_size = chunk_size
        timestep_duration = asr_model.cfg.preprocessor.window_stride

        self.n_chunk_look_back = int(timestep_duration * self.sr)
        self.n_chunk_samples = int(chunk_size * self.sr)
        self.buffer_size = buffer_size
        total_buffer_len = int(buffer_size / timestep_duration)
        self.n_feat = asr_model.cfg.preprocessor.features
        self.sample_buffer = torch.zeros(int(self.buffer_size * self.sr))
        self.buffer = torch.ones([self.n_feat, total_buffer_len], dtype=torch.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        self.feature_chunk_len = int(chunk_size / timestep_duration)
        self.feature_buffer_len = total_buffer_len

        self.reset()
        cfg = copy.deepcopy(asr_model.cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)

        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        self.raw_preprocessor = EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        self.raw_preprocessor.to(asr_model.device)

    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer = torch.ones(self.buffer.shape, dtype=torch.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        self.frame_buffers = []
        self.sample_buffer = torch.zeros(int(self.buffer_size * self.sr))
        self.feature_buffer = (
            torch.ones([self.n_feat, self.feature_buffer_len], dtype=torch.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        )

    def _add_chunk_to_buffer(self, chunk):
        """
        Add time-series audio signal to `sample_buffer`

        Args:
            chunk (Tensor):
                Tensor filled with time-series audio signal
        """
        self.sample_buffer[: -self.n_chunk_samples] = self.sample_buffer[self.n_chunk_samples :].clone()
        self.sample_buffer[-self.n_chunk_samples :] = chunk.clone()

    def _update_feature_buffer(self, feat_chunk):
        """
        Add an extracted feature to `feature_buffer`
        """
        self.feature_buffer[:, : -self.feature_chunk_len] = self.feature_buffer[:, self.feature_chunk_len :].clone()
        self.feature_buffer[:, -self.feature_chunk_len :] = feat_chunk.clone()

    def get_raw_feature_buffer(self):
        return self.feature_buffer

    def get_normalized_feature_buffer(self):
        normalized_buffer, _, _ = normalize_batch(
            x=self.feature_buffer.unsqueeze(0),
            seq_len=torch.tensor([len(self.feature_buffer)]),
            normalize_type=self.model_normalize_type,
        )
        return normalized_buffer.squeeze(0)

    def _convert_buffer_to_features(self):
        """
        Extract features from the time-series audio buffer `sample_buffer`.
        """
        # samples for conversion to features.
        # Add look_back to have context for the first feature
        samples = self.sample_buffer[: -(self.n_chunk_samples + self.n_chunk_look_back)]
        device = self.asr_model.device
        audio_signal = samples.unsqueeze_(0).to(device)
        audio_signal_len = torch.Tensor([samples.shape[1]]).to(device)
        features, features_len = self.raw_preprocessor(
            input_signal=audio_signal,
            length=audio_signal_len,
        )
        features = features.squeeze()
        self._update_feature_buffer(features[:, -self.feature_chunk_len :])

    def update_feature_buffer(self, chunk):
        """
        Update time-series signal `chunk` to the buffer then generate features out of the
        signal in the audio buffer.

        Args:
            chunk (Tensor):
                Tensor filled with time-series audio signal
        """
        if len(chunk) > self.n_chunk_samples:
            raise ValueError(f"chunk should be of length {self.n_chunk_samples} or less")
        if len(chunk) < self.n_chunk_samples:
            temp_chunk = torch.zeros(self.n_chunk_samples, dtype=torch.float32)
            temp_chunk[: chunk.shape[0]] = chunk
            chunk = temp_chunk
        self._add_chunk_to_buffer(chunk)
        self._convert_buffer_to_features()


class AudioFeatureIterator(IterableDataset):
    def __init__(self, samples, frame_len, preprocessor, device, pad_to_frame_len=True):
        self._samples = samples
        self._frame_len = frame_len
        self._start = 0
        self.output = True
        self.count = 0
        self.pad_to_frame_len = pad_to_frame_len
        timestep_duration = preprocessor._cfg['window_stride']
        self._feature_frame_len = frame_len / timestep_duration
        audio_signal = torch.from_numpy(self._samples).unsqueeze_(0).to(device)
        audio_signal_len = torch.Tensor([self._samples.shape[0]]).to(device)
        self._features, self._features_len = preprocessor(
            input_signal=audio_signal,
            length=audio_signal_len,
        )
        self._features = self._features.squeeze()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        last = int(self._start + self._feature_frame_len)
        if last <= self._features_len[0]:
            frame = self._features[:, self._start : last].cpu()
            self._start = last
        else:
            if not self.pad_to_frame_len:
                frame = self._features[:, self._start : self._features_len[0]].cpu()
            else:
                frame = np.zeros([self._features.shape[0], int(self._feature_frame_len)], dtype='float32')
                segment = self._features[:, self._start : self._features_len[0]].cpu()
                frame[:, : segment.shape[1]] = segment
            self.output = False
        self.count += 1
        return frame


def speech_collate_fn(batch):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    _, audio_lengths = zip(*batch)
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()

    audio_signal = []
    for sig, sig_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None

    return audio_signal, audio_lengths


# simple data layer to pass buffered frames of audio samples
class AudioBuffersDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            "processed_signal": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self):
        super().__init__()

    def __iter__(self):
        return self

    def __next__(self):
        if self._buf_count == len(self.signal):
            raise StopIteration
        self._buf_count += 1
        return (
            torch.as_tensor(self.signal[self._buf_count - 1], dtype=torch.float32),
            torch.as_tensor(self.signal[self._buf_count - 1].shape[1], dtype=torch.int64),
        )

    def set_signal(self, signals):
        self.signal = signals
        self.signal_shape = self.signal[0].shape
        self._buf_count = 0

    def __len__(self):
        return 1


class FeatureFrameBufferer:
    """
    Class to append each feature frame to a buffer and return
    an array of buffers.
    """

    def __init__(self, asr_model, frame_len=1.6, batch_size=4, total_buffer=4.0, pad_to_buffer_len=True):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        if hasattr(asr_model.preprocessor, 'log') and asr_model.preprocessor.log:
            self.ZERO_LEVEL_SPEC_DB_VAL = -16.635  # Log-Melspectrogram value for zero signal
        else:
            self.ZERO_LEVEL_SPEC_DB_VAL = 0.0
        self.asr_model = asr_model
        self.sr = asr_model._cfg.sample_rate
        self.frame_len = frame_len
        timestep_duration = asr_model._cfg.preprocessor.window_stride
        self.n_frame_len = int(frame_len / timestep_duration)

        total_buffer_len = int(total_buffer / timestep_duration)
        self.n_feat = asr_model._cfg.preprocessor.features
        self.buffer = np.ones([self.n_feat, total_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        self.pad_to_buffer_len = pad_to_buffer_len
        self.batch_size = batch_size

        self.signal_end = False
        self.frame_reader = None
        self.feature_buffer_len = total_buffer_len

        self.feature_buffer = (
            np.ones([self.n_feat, self.feature_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        )
        self.frame_buffers = []
        self.buffered_features_size = 0
        self.reset()
        self.buffered_len = 0

    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer = np.ones(shape=self.buffer.shape, dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        self.prev_char = ''
        self.unmerged = []
        self.frame_buffers = []
        self.buffered_len = 0
        self.feature_buffer = (
            np.ones([self.n_feat, self.feature_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        )

    def get_batch_frames(self):
        if self.signal_end:
            return []
        batch_frames = []
        for frame in self.frame_reader:
            batch_frames.append(np.copy(frame))
            if len(batch_frames) == self.batch_size:
                return batch_frames
        self.signal_end = True

        return batch_frames

    def get_frame_buffers(self, frames):
        # Build buffers for each frame
        self.frame_buffers = []
        for frame in frames:
            curr_frame_len = frame.shape[1]
            self.buffered_len += curr_frame_len
            if curr_frame_len < self.feature_buffer_len and not self.pad_to_buffer_len:
                self.frame_buffers.append(np.copy(frame))
                continue
            self.buffer[:, :-curr_frame_len] = self.buffer[:, curr_frame_len:]
            self.buffer[:, -self.n_frame_len :] = frame
            self.frame_buffers.append(np.copy(self.buffer))
        return self.frame_buffers

    def set_frame_reader(self, frame_reader):
        self.frame_reader = frame_reader
        self.signal_end = False

    def _update_feature_buffer(self, feat_frame):
        curr_frame_len = feat_frame.shape[1]
        if curr_frame_len < self.feature_buffer_len and not self.pad_to_buffer_len:
            self.feature_buffer = np.copy(feat_frame)  # assume that only the last frame is less than the buffer length
        else:
            self.feature_buffer[:, : -feat_frame.shape[1]] = self.feature_buffer[:, feat_frame.shape[1] :]
            self.feature_buffer[:, -feat_frame.shape[1] :] = feat_frame
        self.buffered_features_size += feat_frame.shape[1]

    def get_norm_consts_per_frame(self, batch_frames):
        norm_consts = []
        for i, frame in enumerate(batch_frames):
            self._update_feature_buffer(frame)
            mean_from_buffer = np.mean(self.feature_buffer, axis=1)
            stdev_from_buffer = np.std(self.feature_buffer, axis=1)
            norm_consts.append((mean_from_buffer.reshape(self.n_feat, 1), stdev_from_buffer.reshape(self.n_feat, 1)))
        return norm_consts

    def normalize_frame_buffers(self, frame_buffers, norm_consts):
        CONSTANT = 1e-5
        for i, frame_buffer in enumerate(frame_buffers):
            frame_buffers[i] = (frame_buffer - norm_consts[i][0]) / (norm_consts[i][1] + CONSTANT)

    def get_buffers_batch(self):
        batch_frames = self.get_batch_frames()

        while len(batch_frames) > 0:

            frame_buffers = self.get_frame_buffers(batch_frames)
            norm_consts = self.get_norm_consts_per_frame(batch_frames)
            if len(frame_buffers) == 0:
                continue
            self.normalize_frame_buffers(frame_buffers, norm_consts)
            return frame_buffers
        return []


# class for streaming frame-based ASR
# 1) use reset() method to reset FrameASR's state
# 2) call transcribe(frame) to do ASR on
#    contiguous signal's frames
class FrameBatchASR:
    """
    class for streaming frame-based ASR use reset() method to reset FrameASR's
    state call transcribe(frame) to do ASR on contiguous signal's frames
    """

    def __init__(
        self,
        asr_model,
        frame_len=1.6,
        total_buffer=4.0,
        batch_size=4,
        pad_to_buffer_len=True,
    ):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        self.frame_bufferer = FeatureFrameBufferer(
            asr_model=asr_model,
            frame_len=frame_len,
            batch_size=batch_size,
            total_buffer=total_buffer,
            pad_to_buffer_len=pad_to_buffer_len,
        )

        self.asr_model = asr_model
        self.decoder = getattr(asr_model, "decoder", None)

        self.batch_size = batch_size
        self.all_logits = []
        self.all_preds = []

        self.unmerged = []

        if self.decoder is None:
            self.blank_id = len(asr_model.tokenizer.vocabulary)
        elif hasattr(asr_model.decoder, "vocabulary"):
            self.blank_id = len(asr_model.decoder.vocabulary)
        else:
            self.blank_id = len(asr_model.joint.vocabulary)
        self.tokenizer = asr_model.tokenizer
        self.toks_unmerged = []
        self.frame_buffers = []
        self.reset()
        cfg = copy.deepcopy(asr_model._cfg)
        self.cfg = cfg
        self.frame_len = frame_len
        OmegaConf.set_struct(cfg.preprocessor, False)

        # some changes for streaming scenario
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        self.raw_preprocessor = EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        self.raw_preprocessor.to(asr_model.device)
        self.preprocessor = self.raw_preprocessor

    def reset(self):
        """
        Reset frame_history and decoder's state
        """
        self.prev_char = ''
        self.unmerged = []
        self.data_layer = AudioBuffersDataLayer()
        self.data_loader = DataLoader(self.data_layer, batch_size=self.batch_size, collate_fn=speech_collate_fn)
        self.all_logits = []
        self.all_preds = []
        self.toks_unmerged = []
        self.frame_buffers = []
        self.frame_bufferer.reset()

    def read_audio_file(self, audio_filepath: str, delay, model_stride_in_secs):
        samples = get_samples(audio_filepath)
        samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
        frame_reader = AudioFeatureIterator(samples, self.frame_len, self.raw_preprocessor, self.asr_model.device)
        self.set_frame_reader(frame_reader)

    def set_frame_reader(self, frame_reader):
        self.frame_bufferer.set_frame_reader(frame_reader)

    @torch.no_grad()
    def infer_logits(self, keep_logits=False):
        frame_buffers = self.frame_bufferer.get_buffers_batch()

        while len(frame_buffers) > 0:
            self.frame_buffers += frame_buffers[:]
            self.data_layer.set_signal(frame_buffers[:])
            self._get_batch_preds(keep_logits)
            frame_buffers = self.frame_bufferer.get_buffers_batch()

    @torch.no_grad()
    def _get_batch_preds(self, keep_logits=False):
        device = self.asr_model.device
        for batch in iter(self.data_loader):

            feat_signal, feat_signal_len = batch
            feat_signal, feat_signal_len = feat_signal.to(device), feat_signal_len.to(device)
            forward_outs = self.asr_model(processed_signal=feat_signal, processed_signal_length=feat_signal_len)

            if len(forward_outs) == 2:  # hybrid ctc rnnt model
                encoded, encoded_len = forward_outs
                log_probs = self.asr_model.ctc_decoder(encoder_output=encoded)
                predictions = log_probs.argmax(dim=-1, keepdim=False)
            else:
                log_probs, encoded_len, predictions = forward_outs

            preds = torch.unbind(predictions)
            for pred in preds:
                self.all_preds.append(pred.cpu().numpy())
            if keep_logits:
                log_probs = torch.unbind(log_probs)
                for log_prob in log_probs:
                    self.all_logits.append(log_prob.cpu())
            else:
                del log_probs
            del encoded_len
            del predictions

    def transcribe(self, tokens_per_chunk: int, delay: int, keep_logits: bool = False):
        self.infer_logits(keep_logits)
        self.unmerged = []
        for pred in self.all_preds:
            decoded = pred.tolist()
            self.unmerged += decoded[len(decoded) - 1 - delay : len(decoded) - 1 - delay + tokens_per_chunk]
        hypothesis = self.greedy_merge(self.unmerged)
        if not keep_logits:
            return hypothesis

        all_logits = []
        for log_prob in self.all_logits:
            T = log_prob.shape[0]
            log_prob = log_prob[T - 1 - delay : T - 1 - delay + tokens_per_chunk, :]
            all_logits.append(log_prob)
        all_logits = torch.concat(all_logits, 0)
        return hypothesis, all_logits

    def greedy_merge(self, preds):
        decoded_prediction = []
        previous = self.blank_id
        for p in preds:
            if (p != previous or previous == self.blank_id) and p != self.blank_id:
                decoded_prediction.append(p)
            previous = p
        hypothesis = self.tokenizer.ids_to_text(decoded_prediction)
        return hypothesis


class BatchedFeatureFrameBufferer(FeatureFrameBufferer):
    """
    Batched variant of FeatureFrameBufferer where batch dimension is the independent audio samples.
    """

    def __init__(self, asr_model, frame_len=1.6, batch_size=4, total_buffer=4.0):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        super().__init__(asr_model, frame_len=frame_len, batch_size=batch_size, total_buffer=total_buffer)

        # OVERRIDES OF BASE CLASS
        timestep_duration = asr_model._cfg.preprocessor.window_stride
        total_buffer_len = int(total_buffer / timestep_duration)
        self.buffer = (
            np.ones([batch_size, self.n_feat, total_buffer_len], dtype=np.float32) * self.ZERO_LEVEL_SPEC_DB_VAL
        )

        # Preserve list of buffers and indices, one for every sample
        self.all_frame_reader = [None for _ in range(self.batch_size)]
        self.signal_end = [False for _ in range(self.batch_size)]
        self.signal_end_index = [None for _ in range(self.batch_size)]
        self.buffer_number = 0  # preserve number of buffers returned since reset.

        self.reset()
        del self.buffered_len
        del self.buffered_features_size

    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        super().reset()
        self.feature_buffer = (
            np.ones([self.batch_size, self.n_feat, self.feature_buffer_len], dtype=np.float32)
            * self.ZERO_LEVEL_SPEC_DB_VAL
        )
        self.all_frame_reader = [None for _ in range(self.batch_size)]
        self.signal_end = [False for _ in range(self.batch_size)]
        self.signal_end_index = [None for _ in range(self.batch_size)]
        self.buffer_number = 0

    def get_batch_frames(self):
        # Exit if all buffers of all samples have been processed
        if all(self.signal_end):
            return []

        # Otherwise sequentially process frames of each sample one by one.
        batch_frames = []
        for idx, frame_reader in enumerate(self.all_frame_reader):
            try:
                frame = next(frame_reader)
                frame = np.copy(frame)

                batch_frames.append(frame)
            except StopIteration:
                # If this sample has finished all of its buffers
                # Set its signal_end flag, and assign it the id of which buffer index
                # did it finish the sample (if not previously set)
                # This will let the alignment module know which sample in the batch finished
                # at which index.
                batch_frames.append(None)
                self.signal_end[idx] = True

                if self.signal_end_index[idx] is None:
                    self.signal_end_index[idx] = self.buffer_number

        self.buffer_number += 1
        return batch_frames

    def get_frame_buffers(self, frames):
        # Build buffers for each frame
        self.frame_buffers = []
        # Loop over all buffers of all samples
        for idx in range(self.batch_size):
            frame = frames[idx]
            # If the sample has a buffer, then process it as usual
            if frame is not None:
                self.buffer[idx, :, : -self.n_frame_len] = self.buffer[idx, :, self.n_frame_len :]
                self.buffer[idx, :, -self.n_frame_len :] = frame
                # self.buffered_len += frame.shape[1]
                # WRAP the buffer at index idx into a outer list
                self.frame_buffers.append([np.copy(self.buffer[idx])])
            else:
                # If the buffer does not exist, the sample has finished processing
                # set the entire buffer for that sample to 0
                self.buffer[idx, :, :] *= 0.0
                self.frame_buffers.append([np.copy(self.buffer[idx])])

        return self.frame_buffers

    def set_frame_reader(self, frame_reader, idx):
        self.all_frame_reader[idx] = frame_reader
        self.signal_end[idx] = False
        self.signal_end_index[idx] = None

    def _update_feature_buffer(self, feat_frame, idx):
        # Update the feature buffer for given sample, or reset if the sample has finished processing
        if feat_frame is not None:
            self.feature_buffer[idx, :, : -feat_frame.shape[1]] = self.feature_buffer[idx, :, feat_frame.shape[1] :]
            self.feature_buffer[idx, :, -feat_frame.shape[1] :] = feat_frame
            # self.buffered_features_size += feat_frame.shape[1]
        else:
            self.feature_buffer[idx, :, :] *= 0.0

    def get_norm_consts_per_frame(self, batch_frames):
        for idx, frame in enumerate(batch_frames):
            self._update_feature_buffer(frame, idx)

        mean_from_buffer = np.mean(self.feature_buffer, axis=2, keepdims=True)  # [B, self.n_feat, 1]
        stdev_from_buffer = np.std(self.feature_buffer, axis=2, keepdims=True)  # [B, self.n_feat, 1]

        return (mean_from_buffer, stdev_from_buffer)

    def normalize_frame_buffers(self, frame_buffers, norm_consts):
        CONSTANT = 1e-8
        for i in range(len(frame_buffers)):
            frame_buffers[i] = (frame_buffers[i] - norm_consts[0][i]) / (norm_consts[1][i] + CONSTANT)

    def get_buffers_batch(self):
        batch_frames = self.get_batch_frames()

        while len(batch_frames) > 0:
            # while there exists at least one sample that has not been processed yet
            frame_buffers = self.get_frame_buffers(batch_frames)
            norm_consts = self.get_norm_consts_per_frame(batch_frames)

            self.normalize_frame_buffers(frame_buffers, norm_consts)
            return frame_buffers
        return []


class BatchedFrameASRRNNT(FrameBatchASR):
    """
    Batched implementation of FrameBatchASR for RNNT models, where the batch dimension is independent audio samples.
    """

    def __init__(
        self,
        asr_model,
        frame_len=1.6,
        total_buffer=4.0,
        batch_size=32,
        max_steps_per_timestep: int = 5,
        stateful_decoding: bool = False,
    ):
        '''
        Args:
            asr_model: An RNNT model.
            frame_len: frame's duration, seconds.
            total_buffer: duration of total audio chunk size, in seconds.
            batch_size: Number of independent audio samples to process at each step.
            max_steps_per_timestep: Maximum number of tokens (u) to process per acoustic timestep (t).
            stateful_decoding: Boolean whether to enable stateful decoding for preservation of state across buffers.
        '''
        super().__init__(asr_model, frame_len=frame_len, total_buffer=total_buffer, batch_size=batch_size)

        # OVERRIDES OF THE BASE CLASS
        self.max_steps_per_timestep = max_steps_per_timestep
        self.stateful_decoding = stateful_decoding

        self.all_alignments = [[] for _ in range(self.batch_size)]
        self.all_preds = [[] for _ in range(self.batch_size)]
        self.all_timestamps = [[] for _ in range(self.batch_size)]
        self.previous_hypotheses = None
        self.batch_index_map = {
            idx: idx for idx in range(self.batch_size)
        }  # pointer from global batch id : local sub-batch id

        try:
            self.eos_id = self.asr_model.tokenizer.eos_id
        except Exception:
            self.eos_id = -1

        print("Performing Stateful decoding :", self.stateful_decoding)

        # OVERRIDES
        self.frame_bufferer = BatchedFeatureFrameBufferer(
            asr_model=asr_model, frame_len=frame_len, batch_size=batch_size, total_buffer=total_buffer
        )

        self.reset()

    def reset(self):
        """
        Reset frame_history and decoder's state
        """
        super().reset()

        self.all_alignments = [[] for _ in range(self.batch_size)]
        self.all_preds = [[] for _ in range(self.batch_size)]
        self.all_timestamps = [[] for _ in range(self.batch_size)]
        self.previous_hypotheses = None
        self.batch_index_map = {idx: idx for idx in range(self.batch_size)}

        self.data_layer = [AudioBuffersDataLayer() for _ in range(self.batch_size)]
        self.data_loader = [
            DataLoader(self.data_layer[idx], batch_size=1, collate_fn=speech_collate_fn)
            for idx in range(self.batch_size)
        ]

    def read_audio_file(self, audio_filepath: list, delay, model_stride_in_secs):
        assert len(audio_filepath) == self.batch_size

        # Read in a batch of audio files, one by one
        for idx in range(self.batch_size):
            samples = get_samples(audio_filepath[idx])
            samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
            frame_reader = AudioFeatureIterator(samples, self.frame_len, self.raw_preprocessor, self.asr_model.device)
            self.set_frame_reader(frame_reader, idx)

    def set_frame_reader(self, frame_reader, idx):
        self.frame_bufferer.set_frame_reader(frame_reader, idx)

    @torch.no_grad()
    def infer_logits(self):
        frame_buffers = self.frame_bufferer.get_buffers_batch()

        while len(frame_buffers) > 0:
            # While at least 1 sample has a buffer left to process
            self.frame_buffers += frame_buffers[:]

            for idx, buffer in enumerate(frame_buffers):
                self.data_layer[idx].set_signal(buffer[:])

            self._get_batch_preds()
            frame_buffers = self.frame_bufferer.get_buffers_batch()

    @torch.no_grad()
    def _get_batch_preds(self):
        """
        Perform dynamic batch size decoding of frame buffers of all samples.

        Steps:
            -   Load all data loaders of every sample
            -   For all samples, determine if signal has finished.
                -   If so, skip calculation of mel-specs.
                -   If not, compute mel spec and length
            -   Perform Encoder forward over this sub-batch of samples. Maintain the indices of samples that were processed.
            -   If performing stateful decoding, prior to decoder forward, remove the states of samples that were not processed.
            -   Perform Decoder + Joint forward for samples that were processed.
            -   For all output RNNT alignment matrix of the joint do:
                -   If signal has ended previously (this was last buffer of padding), skip alignment
                -   Otherwise, recalculate global index of this sample from the sub-batch index, and preserve alignment.
            -   Same for preds
            -   Update indices of sub-batch with global index map.
            - Redo steps until all samples were processed (sub-batch size == 0).
        """
        device = self.asr_model.device

        data_iters = [iter(data_loader) for data_loader in self.data_loader]

        feat_signals = []
        feat_signal_lens = []

        new_batch_keys = []
        # while not all(self.frame_bufferer.signal_end):
        for idx in range(self.batch_size):
            if self.frame_bufferer.signal_end[idx]:
                continue

            batch = next(data_iters[idx])
            feat_signal, feat_signal_len = batch
            feat_signal, feat_signal_len = feat_signal.to(device), feat_signal_len.to(device)

            feat_signals.append(feat_signal)
            feat_signal_lens.append(feat_signal_len)

            # preserve batch indeices
            new_batch_keys.append(idx)

        if len(feat_signals) == 0:
            return

        feat_signal = torch.cat(feat_signals, 0)
        feat_signal_len = torch.cat(feat_signal_lens, 0)

        del feat_signals, feat_signal_lens

        encoded, encoded_len = self.asr_model(processed_signal=feat_signal, processed_signal_length=feat_signal_len)

        # filter out partial hypotheses from older batch subset
        if self.stateful_decoding and self.previous_hypotheses is not None:
            new_prev_hypothesis = []
            for new_batch_idx, global_index_key in enumerate(new_batch_keys):
                old_pos = self.batch_index_map[global_index_key]
                new_prev_hypothesis.append(self.previous_hypotheses[old_pos])
            self.previous_hypotheses = new_prev_hypothesis

        best_hyp, _ = self.asr_model.decoding.rnnt_decoder_predictions_tensor(
            encoded, encoded_len, return_hypotheses=True, partial_hypotheses=self.previous_hypotheses
        )

        if self.stateful_decoding:
            # preserve last state from hypothesis of new batch indices
            self.previous_hypotheses = best_hyp

        for idx, hyp in enumerate(best_hyp):
            global_index_key = new_batch_keys[idx]  # get index of this sample in the global batch

            has_signal_ended = self.frame_bufferer.signal_end[global_index_key]
            if not has_signal_ended:
                self.all_alignments[global_index_key].append(hyp.alignments)

        preds = [hyp.y_sequence for hyp in best_hyp]
        for idx, pred in enumerate(preds):
            global_index_key = new_batch_keys[idx]  # get index of this sample in the global batch

            has_signal_ended = self.frame_bufferer.signal_end[global_index_key]
            if not has_signal_ended:
                self.all_preds[global_index_key].append(pred.cpu().numpy())

        timestamps = [hyp.timestep for hyp in best_hyp]
        for idx, timestep in enumerate(timestamps):
            global_index_key = new_batch_keys[idx]  # get index of this sample in the global batch

            has_signal_ended = self.frame_bufferer.signal_end[global_index_key]
            if not has_signal_ended:
                self.all_timestamps[global_index_key].append(timestep)

        if self.stateful_decoding:
            # State resetting is being done on sub-batch only, global index information is not being updated
            reset_states = self.asr_model.decoder.initialize_state(encoded)

            for idx, pred in enumerate(preds):
                if len(pred) > 0 and pred[-1] == self.eos_id:
                    # reset states :
                    self.previous_hypotheses[idx].y_sequence = self.previous_hypotheses[idx].y_sequence[:-1]
                    self.previous_hypotheses[idx].dec_state = self.asr_model.decoder.batch_select_state(
                        reset_states, idx
                    )

        # Position map update
        if len(new_batch_keys) != len(self.batch_index_map):
            for new_batch_idx, global_index_key in enumerate(new_batch_keys):
                self.batch_index_map[global_index_key] = new_batch_idx  # let index point from global pos -> local pos

        del encoded, encoded_len
        del best_hyp, pred

    def transcribe(
        self,
        tokens_per_chunk: int,
        delay: int,
    ):
        """
        Performs "middle token" alignment prediction using the buffered audio chunk.
        """
        self.infer_logits()

        self.unmerged = [[] for _ in range(self.batch_size)]
        for idx, alignments in enumerate(self.all_alignments):

            signal_end_idx = self.frame_bufferer.signal_end_index[idx]
            if signal_end_idx is None:
                raise ValueError("Signal did not end")

            for a_idx, alignment in enumerate(alignments):
                if delay == len(alignment):  # chunk size = buffer size
                    offset = 0
                else:  # all other cases
                    offset = 1

                alignment = alignment[
                    len(alignment) - offset - delay : len(alignment) - offset - delay + tokens_per_chunk
                ]

                ids, toks = self._alignment_decoder(alignment, self.asr_model.tokenizer, self.blank_id)

                if len(ids) > 0 and a_idx < signal_end_idx:
                    self.unmerged[idx] = inplace_buffer_merge(
                        self.unmerged[idx],
                        ids,
                        delay,
                        model=self.asr_model,
                    )

        output = []
        for idx in range(self.batch_size):
            output.append(self.greedy_merge(self.unmerged[idx]))
        return output

    def _alignment_decoder(self, alignments, tokenizer, blank_id):
        s = []
        ids = []

        for t in range(len(alignments)):
            for u in range(len(alignments[t])):
                _, token_id = alignments[t][u]  # (logprob, token_id)
                token_id = int(token_id)
                if token_id != blank_id:
                    token = tokenizer.ids_to_tokens([token_id])[0]
                    s.append(token)
                    ids.append(token_id)

                else:
                    # blank token
                    pass

        return ids, s

    def greedy_merge(self, preds):
        decoded_prediction = [p for p in preds]
        hypothesis = self.asr_model.tokenizer.ids_to_text(decoded_prediction)
        return hypothesis


class LongestCommonSubsequenceBatchedFrameASRRNNT(BatchedFrameASRRNNT):
    """
    Implements a token alignment algorithm for text alignment instead of middle token alignment.

    For more detail, read the docstring of longest_common_subsequence_merge().
    """

    def __init__(
        self,
        asr_model,
        frame_len=1.6,
        total_buffer=4.0,
        batch_size=4,
        max_steps_per_timestep: int = 5,
        stateful_decoding: bool = False,
        alignment_basepath: str = None,
    ):
        '''
        Args:
            asr_model: An RNNT model.
            frame_len: frame's duration, seconds.
            total_buffer: duration of total audio chunk size, in seconds.
            batch_size: Number of independent audio samples to process at each step.
            max_steps_per_timestep: Maximum number of tokens (u) to process per acoustic timestep (t).
            stateful_decoding: Boolean whether to enable stateful decoding for preservation of state across buffers.
            alignment_basepath: Str path to a directory where alignments from LCS will be preserved for later analysis.
        '''
        super().__init__(asr_model, frame_len, total_buffer, batch_size, max_steps_per_timestep, stateful_decoding)
        self.sample_offset = 0
        self.lcs_delay = -1

        self.alignment_basepath = alignment_basepath

    def transcribe(
        self,
        tokens_per_chunk: int,
        delay: int,
    ):
        if self.lcs_delay < 0:
            raise ValueError(
                "Please set LCS Delay valus as `(buffer_duration - chunk_duration) / model_stride_in_secs`"
            )

        self.infer_logits()

        self.unmerged = [[] for _ in range(self.batch_size)]
        for idx, alignments in enumerate(self.all_alignments):

            signal_end_idx = self.frame_bufferer.signal_end_index[idx]
            if signal_end_idx is None:
                raise ValueError("Signal did not end")

            for a_idx, alignment in enumerate(alignments):

                # Middle token first chunk
                if a_idx == 0:
                    # len(alignment) - 1 - delay + tokens_per_chunk
                    alignment = alignment[len(alignment) - 1 - delay :]
                    ids, toks = self._alignment_decoder(alignment, self.asr_model.tokenizer, self.blank_id)

                    if len(ids) > 0:
                        self.unmerged[idx] = inplace_buffer_merge(
                            self.unmerged[idx],
                            ids,
                            delay,
                            model=self.asr_model,
                        )

                else:
                    ids, toks = self._alignment_decoder(alignment, self.asr_model.tokenizer, self.blank_id)
                    if len(ids) > 0 and a_idx < signal_end_idx:

                        if self.alignment_basepath is not None:
                            basepath = self.alignment_basepath
                            sample_offset = self.sample_offset + idx
                            alignment_offset = a_idx
                            path = os.path.join(basepath, str(sample_offset))

                            os.makedirs(path, exist_ok=True)
                            path = os.path.join(path, "alignment_" + str(alignment_offset) + '.pt')

                            filepath = path
                        else:
                            filepath = None

                        self.unmerged[idx] = lcs_alignment_merge_buffer(
                            self.unmerged[idx],
                            ids,
                            self.lcs_delay,
                            model=self.asr_model,
                            max_steps_per_timestep=self.max_steps_per_timestep,
                            filepath=filepath,
                        )

        output = []
        for idx in range(self.batch_size):
            output.append(self.greedy_merge(self.unmerged[idx]))
        return output


class CacheAwareStreamingAudioBuffer:
    """
    A buffer to be used for cache-aware streaming. It can load a single or multiple audio files/processed signals, split them in chunks and return one on one.
    It can be used to simulate streaming audio or audios.
    """

    def __init__(self, model, online_normalization=None, pad_and_drop_preencoded=False):
        '''
        Args:
            model: An ASR model.
            online_normalization (bool): whether to perform online normalization per chunk or normalize the whole audio before chunking
            pad_and_drop_preencoded (bool): if true pad first audio chunk and always drop preencoded
        '''
        self.model = model
        self.buffer = None
        self.buffer_idx = 0
        self.streams_length = None
        self.step = 0
        self.pad_and_drop_preencoded = pad_and_drop_preencoded

        self.online_normalization = online_normalization
        if not isinstance(model.encoder, StreamingEncoder):
            raise ValueError(
                "The model's encoder is not inherited from StreamingEncoder, and likely not to support streaming!"
            )
        if model.encoder.streaming_cfg is None:
            model.encoder.setup_streaming_params()
        self.streaming_cfg = model.encoder.streaming_cfg

        self.input_features = model.encoder._feat_in

        self.preprocessor = self.extract_preprocessor()

        if hasattr(model.encoder, "pre_encode") and hasattr(model.encoder.pre_encode, "get_sampling_frames"):
            self.sampling_frames = model.encoder.pre_encode.get_sampling_frames()
        else:
            self.sampling_frames = None

    def __iter__(self):
        while True:
            if self.buffer_idx >= self.buffer.size(-1):
                return

            if self.buffer_idx == 0 and isinstance(self.streaming_cfg.chunk_size, list):
                if self.pad_and_drop_preencoded:
                    chunk_size = self.streaming_cfg.chunk_size[1]
                else:
                    chunk_size = self.streaming_cfg.chunk_size[0]
            else:
                chunk_size = (
                    self.streaming_cfg.chunk_size[1]
                    if isinstance(self.streaming_cfg.chunk_size, list)
                    else self.streaming_cfg.chunk_size
                )

            if self.buffer_idx == 0 and isinstance(self.streaming_cfg.shift_size, list):
                if self.pad_and_drop_preencoded:
                    shift_size = self.streaming_cfg.shift_size[1]
                else:
                    shift_size = self.streaming_cfg.shift_size[0]
            else:
                shift_size = (
                    self.streaming_cfg.shift_size[1]
                    if isinstance(self.streaming_cfg.shift_size, list)
                    else self.streaming_cfg.shift_size
                )

            audio_chunk = self.buffer[:, :, self.buffer_idx : self.buffer_idx + chunk_size]

            if self.sampling_frames is not None:
                # checking to make sure the audio chunk has enough frames to produce at least one output after downsampling
                if self.buffer_idx == 0 and isinstance(self.sampling_frames, list):
                    cur_sampling_frames = self.sampling_frames[0]
                else:
                    cur_sampling_frames = (
                        self.sampling_frames[1] if isinstance(self.sampling_frames, list) else self.sampling_frames
                    )
                if audio_chunk.size(-1) < cur_sampling_frames:
                    return

            # Adding the cache needed for the pre-encoder part of the model to the chunk
            # if there is not enough frames to be used as the pre-encoding cache, zeros would be added
            zeros_pads = None
            if self.buffer_idx == 0 and isinstance(self.streaming_cfg.pre_encode_cache_size, list):
                if self.pad_and_drop_preencoded:
                    cache_pre_encode_num_frames = self.streaming_cfg.pre_encode_cache_size[1]
                else:
                    cache_pre_encode_num_frames = self.streaming_cfg.pre_encode_cache_size[0]
                cache_pre_encode = torch.zeros(
                    (audio_chunk.size(0), self.input_features, cache_pre_encode_num_frames),
                    device=audio_chunk.device,
                    dtype=audio_chunk.dtype,
                )
            else:
                if isinstance(self.streaming_cfg.pre_encode_cache_size, list):
                    pre_encode_cache_size = self.streaming_cfg.pre_encode_cache_size[1]
                else:
                    pre_encode_cache_size = self.streaming_cfg.pre_encode_cache_size

                start_pre_encode_cache = self.buffer_idx - pre_encode_cache_size
                if start_pre_encode_cache < 0:
                    start_pre_encode_cache = 0
                cache_pre_encode = self.buffer[:, :, start_pre_encode_cache : self.buffer_idx]
                if cache_pre_encode.size(-1) < pre_encode_cache_size:
                    zeros_pads = torch.zeros(
                        (
                            audio_chunk.size(0),
                            audio_chunk.size(-2),
                            pre_encode_cache_size - cache_pre_encode.size(-1),
                        ),
                        device=audio_chunk.device,
                        dtype=audio_chunk.dtype,
                    )

            added_len = cache_pre_encode.size(-1)
            audio_chunk = torch.cat((cache_pre_encode, audio_chunk), dim=-1)

            if self.online_normalization:
                audio_chunk, x_mean, x_std = normalize_batch(
                    x=audio_chunk,
                    seq_len=torch.tensor([audio_chunk.size(-1)] * audio_chunk.size(0)),
                    normalize_type=self.model_normalize_type,
                )

            if zeros_pads is not None:
                # TODO: check here when zero_pads is not None and added_len is already non-zero
                audio_chunk = torch.cat((zeros_pads, audio_chunk), dim=-1)
                added_len += zeros_pads.size(-1)

            max_chunk_lengths = self.streams_length - self.buffer_idx
            max_chunk_lengths = max_chunk_lengths + added_len
            chunk_lengths = torch.clamp(max_chunk_lengths, min=0, max=audio_chunk.size(-1))

            self.buffer_idx += shift_size
            self.step += 1
            yield audio_chunk, chunk_lengths

    def is_buffer_empty(self):
        if self.buffer_idx >= self.buffer.size(-1):
            return True
        else:
            return False

    def __len__(self):
        return len(self.buffer)

    def reset_buffer(self):
        self.buffer = None
        self.buffer_idx = 0
        self.streams_length = None
        self.step = 0

    def reset_buffer_pointer(self):
        self.buffer_idx = 0
        self.step = 0

    def extract_preprocessor(self):
        cfg = copy.deepcopy(self.model._cfg)
        self.model_normalize_type = cfg.preprocessor.normalize
        OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        if self.online_normalization:
            cfg.preprocessor.normalize = "None"

        preprocessor = self.model.from_config_dict(cfg.preprocessor)
        return preprocessor.to(self.get_model_device())

    def append_audio_file(self, audio_filepath, stream_id=-1):
        audio = get_samples(audio_filepath)
        processed_signal, processed_signal_length, stream_id = self.append_audio(audio, stream_id)
        return processed_signal, processed_signal_length, stream_id

    def append_audio(self, audio, stream_id=-1):
        processed_signal, processed_signal_length = self.preprocess_audio(audio)
        processed_signal, processed_signal_length, stream_id = self.append_processed_signal(
            processed_signal, stream_id
        )
        return processed_signal, processed_signal_length, stream_id

    def append_processed_signal(self, processed_signal, stream_id=-1):
        processed_signal_length = torch.tensor(processed_signal.size(-1), device=processed_signal.device)
        if stream_id >= 0 and (self.streams_length is not None and stream_id >= len(self.streams_length)):
            raise ValueError("Not valid stream_id!")
        if self.buffer is None:
            if stream_id >= 0:
                raise ValueError("stream_id can not be specified when there is no stream.")
            self.buffer = processed_signal
            self.streams_length = torch.tensor([processed_signal_length], device=processed_signal.device)
        else:
            if self.buffer.size(1) != processed_signal.size(1):
                raise ValueError("Buffer and the processed signal have different dimensions!")
            if stream_id < 0:
                self.buffer = torch.nn.functional.pad(self.buffer, pad=(0, 0, 0, 0, 0, 1))
                self.streams_length = torch.cat(
                    (self.streams_length, torch.tensor([0], device=self.streams_length.device)), dim=-1
                )
                stream_id = len(self.streams_length) - 1
            needed_len = self.streams_length[stream_id] + processed_signal_length
            if needed_len > self.buffer.size(-1):
                self.buffer = torch.nn.functional.pad(self.buffer, pad=(0, needed_len - self.buffer.size(-1)))

            self.buffer[
                stream_id, :, self.streams_length[stream_id] : self.streams_length[stream_id] + processed_signal_length
            ] = processed_signal
            self.streams_length[stream_id] = self.streams_length[stream_id] + processed_signal.size(-1)

        if self.online_normalization:
            processed_signal, x_mean, x_std = normalize_batch(
                x=processed_signal,
                seq_len=torch.tensor([processed_signal_length]),
                normalize_type=self.model_normalize_type,
            )
        return processed_signal, processed_signal_length, stream_id

    def get_model_device(self):
        return self.model.device

    def preprocess_audio(self, audio, device=None):
        if device is None:
            device = self.get_model_device()
        audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(device)
        audio_signal_len = torch.Tensor([audio.shape[0]]).to(device)
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=audio_signal, length=audio_signal_len
        )
        return processed_signal, processed_signal_length

    def get_all_audios(self):
        processed_signal = self.buffer
        if self.online_normalization:
            processed_signal, x_mean, x_std = normalize_batch(
                x=processed_signal,
                seq_len=torch.tensor(self.streams_length),
                normalize_type=self.model_normalize_type,
            )
        return processed_signal, self.streams_length


class FrameBatchMultiTaskAED(FrameBatchASR):
    def __init__(self, asr_model, frame_len=4, total_buffer=4, batch_size=4):
        super().__init__(asr_model, frame_len, total_buffer, batch_size, pad_to_buffer_len=False)

    def get_input_tokens(self, sample: dict):
        if self.asr_model.prompt_format == "canary":
            missing_keys = [k for k in ("source_lang", "target_lang", "taskname", "pnc") if k not in sample]
            if missing_keys:
                raise RuntimeError(
                    f"We found sample that is missing the following keys: {missing_keys}"
                    f"Please ensure that every utterance in the input manifests contains these keys. Sample: {sample}"
                )
            tokens = self.asr_model.prompt.encode_dialog(
                turns=[
                    {
                        "role": "user",
                        "slots": {
                            **sample,
                            self.asr_model.prompt.PROMPT_LANGUAGE_SLOT: "spl_tokens",
                        },
                    }
                ]
            )["context_ids"]
        else:
            raise ValueError(f"Unknown prompt format: {self.asr_model.prompt_format}")
        return torch.tensor(tokens, dtype=torch.long, device=self.asr_model.device).unsqueeze(0)  # [1, T]

    def read_audio_file(self, audio_filepath: str, delay, model_stride_in_secs, meta_data):
        self.input_tokens = self.get_input_tokens(meta_data)
        samples = get_samples(audio_filepath)
        samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
        frame_reader = AudioFeatureIterator(
            samples, self.frame_len, self.raw_preprocessor, self.asr_model.device, pad_to_frame_len=False
        )
        self.set_frame_reader(frame_reader)

    @torch.no_grad()
    def _get_batch_preds(self, keep_logits=False):
        device = self.asr_model.device
        for batch in iter(self.data_loader):
            feat_signal, feat_signal_len = batch
            feat_signal, feat_signal_len = feat_signal.to(device), feat_signal_len.to(device)
            tokens = self.input_tokens.to(device).repeat(feat_signal.size(0), 1)
            tokens_len = torch.tensor([tokens.size(1)] * tokens.size(0), device=device).long()

            batch_input = (feat_signal, feat_signal_len, None, None, tokens, tokens_len)
            predictions = self.asr_model.predict_step(batch_input, has_processed_signal=True)
            self.all_preds.extend(predictions)
            del predictions

    def transcribe(
        self, tokens_per_chunk: Optional[int] = None, delay: Optional[int] = None, keep_logits: bool = False
    ):
        """
        unsued params are for keeping the same signature as the parent class
        """
        self.infer_logits(keep_logits)

        hypothesis = " ".join(self.all_preds)
        if not keep_logits:
            return hypothesis

        print("keep_logits=True is not supported for MultiTaskAEDFrameBatchInfer. Returning empty logits.")
        return hypothesis, []


class FrameBatchChunkedRNNT(FrameBatchASR):
    def __init__(self, asr_model, frame_len=4, total_buffer=4, batch_size=4):
        super().__init__(asr_model, frame_len, total_buffer, batch_size, pad_to_buffer_len=False)

    def read_audio_file(self, audio_filepath: str, delay, model_stride_in_secs):
        samples = get_samples(audio_filepath)
        samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
        frame_reader = AudioFeatureIterator(
            samples, self.frame_len, self.raw_preprocessor, self.asr_model.device, pad_to_frame_len=False
        )
        self.set_frame_reader(frame_reader)

    @torch.no_grad()
    def _get_batch_preds(self, keep_logits=False):
        device = self.asr_model.device
        for batch in iter(self.data_loader):
            feat_signal, feat_signal_len = batch
            feat_signal, feat_signal_len = feat_signal.to(device), feat_signal_len.to(device)

            encoded, encoded_len = self.asr_model(
                processed_signal=feat_signal, processed_signal_length=feat_signal_len
            )

            best_hyp_text, all_hyp_text = self.asr_model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
            )
            self.all_preds.extend(best_hyp_text)
            del best_hyp_text
            del all_hyp_text
            del encoded
            del encoded_len

    def transcribe(
        self, tokens_per_chunk: Optional[int] = None, delay: Optional[int] = None, keep_logits: bool = False
    ):
        """
        unsued params are for keeping the same signature as the parent class
        """
        self.infer_logits(keep_logits)

        hypothesis = " ".join(self.all_preds)
        if not keep_logits:
            return hypothesis

        print("keep_logits=True is not supported for FrameBatchChunkedRNNT. Returning empty logits.")
        return hypothesis, []


class FrameBatchChunkedCTC(FrameBatchASR):
    def __init__(self, asr_model, frame_len=4, total_buffer=4, batch_size=4):
        super().__init__(asr_model, frame_len, total_buffer, batch_size, pad_to_buffer_len=False)

    def read_audio_file(self, audio_filepath: str, delay, model_stride_in_secs):
        samples = get_samples(audio_filepath)
        samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
        frame_reader = AudioFeatureIterator(
            samples, self.frame_len, self.raw_preprocessor, self.asr_model.device, pad_to_frame_len=False
        )
        self.set_frame_reader(frame_reader)

    @torch.no_grad()
    def _get_batch_preds(self, keep_logits=False):
        device = self.asr_model.device
        for batch in iter(self.data_loader):
            feat_signal, feat_signal_len = batch
            feat_signal, feat_signal_len = feat_signal.to(device), feat_signal_len.to(device)

            results = self.asr_model(processed_signal=feat_signal, processed_signal_length=feat_signal_len)
            if len(results) == 2:  # hybrid model
                encoded, encoded_len = results
                log_probs = self.asr_model.ctc_decoder(encoder_output=encoded)
                transcribed_texts, _ = self.asr_model.ctc_decoding.ctc_decoder_predictions_tensor(
                    decoder_outputs=log_probs,
                    decoder_lengths=encoded_len,
                    return_hypotheses=False,
                )
            else:
                log_probs, encoded_len, predictions = results
                transcribed_texts, _ = self.asr_model.decoding.ctc_decoder_predictions_tensor(
                    decoder_outputs=log_probs,
                    decoder_lengths=encoded_len,
                    return_hypotheses=False,
                )

            self.all_preds.extend(transcribed_texts)
            del log_probs
            del encoded_len
            del predictions

    def transcribe(
        self, tokens_per_chunk: Optional[int] = None, delay: Optional[int] = None, keep_logits: bool = False
    ):
        """
        unsued params are for keeping the same signature as the parent class
        """
        self.infer_logits(keep_logits)

        hypothesis = " ".join(self.all_preds)
        if not keep_logits:
            return hypothesis

        print("keep_logits=True is not supported for FrameBatchChunkedCTC. Returning empty logits.")
        return hypothesis, []
