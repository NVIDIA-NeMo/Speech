# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
import time
from copy import deepcopy
from typing import Dict

import torch
from omegaconf import DictConfig

from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.offline_clustering import get_scale_interpolated_embs, split_input_data
from nemo.collections.asr.parts.utils.online_clustering import OnlineSpeakerClustering
from nemo.collections.asr.parts.utils.speaker_utils import (
    OnlineSegmentor,
    audio_rttm_map,
    generate_cluster_labels,
    get_embs_and_timestamps,
)
from nemo.utils import logging, model_utils

__all__ = ['OnlineClusteringDiarizer']


def timeit(method):
    """
    Monitor elapsed time of the corresponding function displaying the method name.

    Args:
        method: function that is being measured

    Return:
        `timed` function for measuring the elapsed time
    """

    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        if 'log_time' in kwargs:
            name = kwargs.get('log_name', method.__name__.upper())
            kwargs['log_time'][name] = int((te - ts) * 1000)
        else:
            logging.info('%2.2fms %r' % ((te - ts) * 1000, method.__name__))
        return result

    return timed


class OnlineClusteringDiarizer(ClusteringDiarizer):
    """
    A class that enables online (streaming) clustering based diarization.

    - The instance created from `OnlineClusteringDiarizer` sets aside a certain amount of memory
      to provide the upcoming inference with history information

    - There are two major modules involved: `OnlineSegmentor` and `OnlineSpeakerClustering`.
        OnlineSegmentor: Take the VAD-timestamps and generate segments for each scale
        OnlineSpeakerClustering: Update the entire speaker labels of the given online session
                                 while updating the speaker labels of the streaming inputs.

    - The overall diarization process is done by calling `diarize_step` function.
      `diarize_step` function goes through the following steps:
        (1) Segmentation (`OnlineSegmentor` class)
        (2) Embedding extraction (`_extract_online_embeddings` function call)
        (3) Online speaker counting and speaker clustering (`OnlineClusteringDiarizer` class)
        (4) Label generation (`generate_cluster_labels` function call)
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = model_utils.convert_model_config_to_dict_config(cfg)
        self._cfg_diarizer = self.cfg.diarizer
        self.base_scale_index = max(self.multiscale_args_dict['scale_dict'].keys())

        self.uniq_id = self._cfg_diarizer.get('uniq_id', None)
        self.decimals = self._cfg_diarizer.get('decimals', 2)
        self.AUDIO_RTTM_MAP = audio_rttm_map(self.cfg.diarizer.manifest_filepath)
        self.sample_rate = self.cfg.sample_rate
        torch.manual_seed(0)

        self._out_dir = self._cfg_diarizer.out_dir
        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)

        if torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device("cuda")
        else:
            self.cuda = False
            self.device = torch.device("cpu")

        self.reset()

        # Set speaker embedding model in eval mode
        self._speaker_model.eval()

    def _init_online_clustering_module(self, clustering_params):
        """
        Initialize online speaker clustering module

        Attributes:
            online_clus (OnlineSpeakerClustering):
                Online clustering diarizer class instance
            history_n (int):
                History buffer size for saving history of speaker label inference
                Total number of embedding vectors saved in the buffer that is kept till the end of the session
            current_n (int):
                Current buffer (FIFO queue) size for calculating the speaker label inference
                Total number of embedding vectors saved in the FIFO queue for clustering inference
        """
        self.online_clus = OnlineSpeakerClustering(
            max_num_speakers=clustering_params.max_num_speakers,
            max_rp_threshold=clustering_params.max_rp_threshold,
            sparse_search_volume=clustering_params.sparse_search_volume,
            history_buffer_size=clustering_params.history_buffer_size,
            current_buffer_size=clustering_params.current_buffer_size,
            cuda=self.cuda,
        )
        self.history_n = clustering_params.history_buffer_size
        self.current_n = clustering_params.current_buffer_size

        self.max_num_speakers = self.online_clus.max_num_speakers

    def _init_online_segmentor_module(self, sample_rate):
        """
        Initialize an online segmentor module

        Attributes:
            online_segmentor (OnlineSegmentor):
                online segmentation module that generates short speech segments from the VAD input
        """
        self.online_segmentor = OnlineSegmentor(sample_rate)

    def _init_memory_buffer(self):
        """
        Variables are kept in memory for future updates

        Attributes:
            memory_margin (int):
                The number of embeddings saved in the memory buffer.
                This memory margin is dependent on the base scale length: margin = (buffer_length)/(base scale shift)
                memory margin is automatically calculated to have minimal memory usage
            memory_segment_ranges (dict):
                The segment range information kept in the memory buffer
            memory_segment_indexes (dict):
                The segment indexes kept in the memory buffer
            memory_cluster_labels (Tensor):
                The cluster labels inferred in the previous diarization steps
        """
        self.memory_margin = 0
        self.memory_segment_ranges = {key: [] for key in self.multiscale_args_dict['scale_dict'].keys()}
        self.memory_segment_indexes = {key: [] for key in self.multiscale_args_dict['scale_dict'].keys()}
        self.memory_cluster_labels = torch.tensor([])

    def _init_temporal_major_voting_module(self, clustering_params):
        """
        Variables needed for taking majority votes for speaker labels

        Attributes:
            use_temporal_label_major_vote (bool):
                Boolean for whether to use temporal majority voting
            temporal_label_major_vote_buffer_size (int):
                buffer size for majority voting
            base_scale_label_dict (dict):
                Dictionary containing multiple speaker labels for major voting
                Speaker labels from multiple steps are saved for each segment index.
        """
        self.use_temporal_label_major_vote = clustering_params.get('use_temporal_label_major_vote', False)
        self.temporal_label_major_vote_buffer_size = clustering_params.get('temporal_label_major_vote_buffer_size', 1)
        self.base_scale_label_dict = {}

    def _init_segment_variables(self):
        """
        Initialize segment variables for each scale.
        Note that we have `uniq_id` variable in case where multiple sessions are handled.
        """
        self.emb_vectors = {}
        self.time_stamps = {}
        self.segment_range_ts = {}
        self.segment_raw_audio = {}
        self.segment_indexes = {}

        for scale_idx in self.multiscale_args_dict['scale_dict'].keys():
            self.multiscale_embeddings_and_timestamps[scale_idx] = [None, None]
            self.emb_vectors[scale_idx] = torch.tensor([])
            self.time_stamps[scale_idx] = []
            self.segment_range_ts[scale_idx] = []
            self.segment_raw_audio[scale_idx] = []
            self.segment_indexes[scale_idx] = []

    def _init_buffer_frame_timestamps(self):
        """
        Timing variables transferred from OnlineDiarWithASR class.
        Buffer is window region where input signal is kept for ASR.
        Frame is window region where the actual inference ASR decoded results are updated

        Example:
            buffer_len = 5.0
            frame_len = 1.0

            |___Buffer___[___________]____________|
            |____________[   Frame   ]____________|

            | <- buffer_start
            |____________| <- frame_start
            |_____________________________________| <- buffer_end

            buffer_start = 12.0
            buffer_end = 17.0
            frame_start = 14.0

        These timestamps and index variables are updated by OnlineDiarWithASR.

        Attributes:
            frame_index (int):
                Integer index of frame window
            frame_start (float):
                The start of the frame window
            buffer_start (float):
                The start of the buffer window
            buffer_end (float):
                The end of the buffer
        """
        self.frame_index = 0
        self.frame_start = 0.0
        self.buffer_start = 0.0
        self.buffer_end = 0.0

    def _transfer_timestamps_to_segmentor(self):
        """
        Pass the timing information from streaming ASR buffers.
        """
        self.online_segmentor.frame_start = self.frame_start
        self.online_segmentor.buffer_start = self.buffer_start
        self.online_segmentor.buffer_end = self.buffer_end

    def reset(self):
        """
        Reset all the necessary variables and initialize classes.

        Attributes:
            n_embed_seg_len (int):
                Number of segments needed for 1 second of input time-series signal
        """
        self.n_embed_seg_len = int(
            self.sample_rate * self.multiscale_args_dict['scale_dict'][self.base_scale_index][0]
        )
        self._init_segment_variables()
        self._init_online_clustering_module(self._cfg_diarizer.clustering.parameters)
        self._init_online_segmentor_module(self.cfg.sample_rate)
        self._init_memory_buffer()
        self._init_temporal_major_voting_module(self._cfg_diarizer.clustering.parameters)
        self._init_buffer_frame_timestamps()

    def _clear_memory(self, scale_idx: int):
        """
        Calculate how many segments should be removed from memory (`memory_margin`) and
        save the necessary information.
        `keep_range` determines how many segments and their corresponding embedding, raw audio,
        timestamps in the memory of the online diarizer instance.

        Args:
            scale_idx (int):
                Scale index in integer type
        """
        base_scale_shift = self.multiscale_args_dict['scale_dict'][self.base_scale_index][1]
        self.memory_margin = int((self.buffer_end - self.buffer_start) / base_scale_shift)

        scale_buffer_size = int(
            len(set(self.scale_mapping_dict[scale_idx].tolist()))
            / len(set(self.scale_mapping_dict[self.base_scale_index].tolist()))
            * (self.history_n + self.current_n)
        )
        keep_range = scale_buffer_size + self.memory_margin
        self.emb_vectors[scale_idx] = self.emb_vectors[scale_idx][-keep_range:]
        self.segment_raw_audio[scale_idx] = self.segment_raw_audio[scale_idx][-keep_range:]
        self.segment_range_ts[scale_idx] = self.segment_range_ts[scale_idx][-keep_range:]
        self.segment_indexes[scale_idx] = self.segment_indexes[scale_idx][-keep_range:]

    @timeit
    def _temporal_label_major_vote(self) -> torch.Tensor:
        """
        Take a majority voting for every segment on temporal steps. This feature significantly reduces the error coming
        from unstable speaker counting in the beginning of sessions.

        Returns:
            maj_vote_labels (list):
                List containing the major-voted speaker labels on temporal domain
        """
        maj_vote_labels = []
        for seg_idx in self.memory_segment_indexes[self.base_scale_index]:
            if seg_idx not in self.base_scale_label_dict:
                self.base_scale_label_dict[seg_idx] = [self.memory_cluster_labels[seg_idx]]
            else:
                while len(self.base_scale_label_dict[seg_idx]) > self.temporal_label_major_vote_buffer_size:
                    self.base_scale_label_dict[seg_idx].pop(0)
                self.base_scale_label_dict[seg_idx].append(self.memory_cluster_labels[seg_idx])

            maj_vote_labels.append(torch.mode(torch.tensor(self.base_scale_label_dict[seg_idx]))[0].item())
        return maj_vote_labels

    def save_history_data(self, scale_idx: int, total_cluster_labels: torch.Tensor, is_online: bool) -> torch.Tensor:
        """
        Save the temporary input to the class memory buffer.

        - Clustering is done for (hist_N + curr_N) number of embeddings.
        - Thus, we need to remove the clustering results on the embedding memory.
        - If self.diar.history_buffer_seg_end is not None, that indicates streaming diarization system
          is starting to save embeddings to its memory. Thus, the new incoming clustering label should be separated.
        - If `is_online = True`, old embeddings outside the window are removed to save GPU memory.

        Args:
            scale_idx (int):
                Scale index in integer
            total_cluster_labels (Tensor):
                The speaker labels from the beginning of the session to the current position
            is_online (bool)
                Boolean variable that indicates whether the system is currently in online mode or not

        Returns:
            cluster_label_hyp (Tensor):
                Majority voted speaker labels over multiple inferences
        """
        total_cluster_labels = total_cluster_labels.tolist()

        if not is_online:
            self.memory_segment_ranges[scale_idx] = deepcopy(self.segment_range_ts[scale_idx])
            self.memory_segment_indexes[scale_idx] = deepcopy(self.segment_indexes[scale_idx])
            if scale_idx == self.base_scale_index:
                self.memory_cluster_labels = deepcopy(total_cluster_labels)

        # Only if there are newly obtained embeddings, update ranges and embeddings.
        elif self.segment_indexes[scale_idx][-1] > self.memory_segment_indexes[scale_idx][-1]:
            # Get the global index of the first segment we want to keep in the buffer
            global_stt_idx = max(max(self.memory_segment_indexes[scale_idx]) - self.memory_margin, 0)

            # Convert global index global_stt_idx to buffer index buffer_stt_idx
            segment_indexes_mat = torch.tensor(self.segment_indexes[scale_idx])
            buffer_stt_idx = torch.where(segment_indexes_mat == global_stt_idx)[0][0]
            self.memory_segment_ranges[scale_idx][global_stt_idx:] = deepcopy(
                self.segment_range_ts[scale_idx][buffer_stt_idx:]
            )
            self.memory_segment_indexes[scale_idx][global_stt_idx:] = deepcopy(
                self.segment_indexes[scale_idx][buffer_stt_idx:]
            )
            if scale_idx == self.base_scale_index:
                self.memory_cluster_labels[global_stt_idx:] = deepcopy(total_cluster_labels[global_stt_idx:])
                if len(self.memory_cluster_labels) != len(self.memory_segment_ranges[scale_idx]):
                    raise ValueError(
                        "self.memory_cluster_labels and self.memory_segment_ranges should always have the same length, "
                        f"but they have {len(self.memory_cluster_labels)} and {len(self.memory_segment_ranges[scale_idx])}."
                    )

            # Remove unnecessary old values
            self._clear_memory(scale_idx)

        if not (
            len(self.emb_vectors[scale_idx])
            == len(self.segment_raw_audio[scale_idx])
            == len(self.segment_indexes[scale_idx])
            == len(self.segment_range_ts[scale_idx])
        ):
            raise ValueError(
                "self.emb_vectors, self.segment_raw_audio, self.segment_indexes, and self.segment_range_ts "
                "should always have the same length, "
                f"but they have {len(self.emb_vectors[scale_idx])}, {len(self.segment_raw_audio[scale_idx])}, "
                f"{len(self.segment_indexes[scale_idx])}, and {len(self.segment_range_ts[scale_idx])}, respectively."
            )

        if self.use_temporal_label_major_vote:
            cluster_label_hyp = self._temporal_label_major_vote()
        else:
            cluster_label_hyp = self.memory_cluster_labels
        return cluster_label_hyp

    @timeit
    @torch.no_grad()
    def _run_embedding_extractor(self, audio_signal: torch.Tensor) -> torch.Tensor:
        """
        Call `forward` function of the speaker embedding model.

        Args:
            audio_signal (Tensor):
                Torch tensor containing time-series signal

        Returns:
            Speaker embedding vectors for the given time-series input `audio_signal`.
        """
        audio_signal = torch.stack(audio_signal).float().to(self.device)
        audio_signal_lens = torch.tensor([self.n_embed_seg_len for k in range(audio_signal.shape[0])]).to(self.device)
        _, torch_embs = self._speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_lens)
        return torch_embs

    @timeit
    def _extract_online_embeddings(
        self, audio_signal: torch.Tensor, segment_ranges: torch.Tensor, embeddings
    ) -> torch.Tensor:
        """
        Incrementally extract speaker embeddings based on `audio_signal` and `segment_ranges` variables.
        Unlike offline speaker diarization, speaker embedding and subsegment ranges are not saved to disk.
        Measures the mismatch between `segment_ranges` and `embeddings` then extract the necessary amount of
        speaker embeddings.

        Args:
            audio_signal (Tensor):
                Torch tensor containing time-series audio signal
            embeddings (Tensor):
                Previously existing Torch tensor containing speaker embedding vector
            segment_ranges(Tensor):
                Torch tensor containing the start and end of each segment

        Returns:
            embeddings (Tensor):
                Concatenated speaker embedding vectors that match segment range information in `segment_ranges`.
        """
        stt_idx = 0 if embeddings is None else embeddings.shape[0]
        end_idx = len(segment_ranges)

        if end_idx > stt_idx:
            torch_embs = self._run_embedding_extractor(audio_signal[stt_idx:end_idx])
            if embeddings is None or embeddings.shape[0] == 0:
                embeddings = torch_embs
            else:
                embeddings = torch.vstack((embeddings[:stt_idx, :], torch_embs))

        elif end_idx < stt_idx:
            embeddings = embeddings[: len(segment_ranges)]

        if len(segment_ranges) != embeddings.shape[0]:
            raise ValueError("Segment ranges and embeddings shapes do not match.")
        return embeddings

    @timeit
    def _perform_online_clustering(
        self, uniq_embs_and_timestamps: Dict[str, torch.Tensor], cuda=False,
    ) -> torch.Tensor:
        """
        Launch online clustering for `uniq_embs_and_timestamps` input variable.

        Args:
            uniq_embs_and_timestamps (dict):
                Dictionary containing embeddings, timestamps and multiscale weights.
                If uniq_embs_and_timestamps contains only one scale, single scale diarization
                is performed.
            cuda (bool):
                Boolean indicator for cuda usages
        """
        device = torch.device("cuda") if cuda else torch.device("cpu")

        # Get base-scale (the highest index) information from uniq_embs_and_timestamps.
        embeddings_in_scales, timestamps_in_scales = split_input_data(
            embeddings_in_scales=uniq_embs_and_timestamps['embeddings'],
            timestamps_in_scales=uniq_embs_and_timestamps['timestamps'],
            multiscale_segment_counts=uniq_embs_and_timestamps['multiscale_segment_counts'],
        )

        curr_emb, self.scale_mapping_dict = get_scale_interpolated_embs(
            multiscale_weights=uniq_embs_and_timestamps['multiscale_weights'],
            embeddings_in_scales=embeddings_in_scales,
            timestamps_in_scales=timestamps_in_scales,
            device=device,
        )

        base_segment_indexes = torch.tensor(self.segment_indexes[self.base_scale_index]).to(curr_emb.device)
        merged_clus_labels = self.online_clus.forward_infer(
            curr_emb=curr_emb, base_segment_indexes=base_segment_indexes, frame_index=self.frame_index, cuda=cuda,
        )
        # Update history data
        for scale_idx, (window, shift) in self.multiscale_args_dict['scale_dict'].items():
            cluster_label_hyp = self.save_history_data(scale_idx, merged_clus_labels, self.online_clus.is_online)

        return cluster_label_hyp

    def _get_interim_output(self) -> torch.Tensor:
        """
        In case buffer is not filled or there is no speech activity in the input, generate temporary output.

        Returns:
            diar_hyp (Tensor): Speaker labels based on the previously saved segments and speaker labels
        """
        if len(self.memory_cluster_labels) == 0 or self.buffer_start < 0:
            diar_hyp, _ = generate_cluster_labels([[0.0, self.total_buffer_in_secs]], [0])
        else:
            diar_hyp, _ = generate_cluster_labels(
                self.memory_segment_ranges[self.base_scale_index], self.memory_cluster_labels
            )
        return diar_hyp

    @timeit
    def diarize_step(self, audio_buffer: torch.Tensor, vad_timestamps: torch.Tensor) -> torch.Tensor:
        """
        A function for a unit diarization step. Each diarization step goes through the following steps:

        1. Segmentation:
            Using `OnlineSegmentor` class, call `run_online_segmentation` method to get the segments.
        2. Embedding Extraction:
            Extract multiscale embeddings from the extracted speech segments.
        3. Online Clustering & Counting
            Perform online speaker clustering by using `OnlineSpeakerClustering` class.
        4. Generate speaker labels:
            Generate start and end timestamps of speaker labels based on the diarization results.

        c.f.) Also see method `diarize` in `ClusteringDiarizer` class.

        Args:
            audio_buffer (Tensor):
                Tensor variable containing the time series signal at the current frame
                Dimensions: (Number of audio time-series samples) x 1
            vad_timestamps (Tensor):
                List containing VAD timestamps.
                Dimensions: (Number of segments) x 2
                Example:
                    >>> vad_timestamps = torch.Tensor([[0.05, 2.52], [3.12, 6.85]])

        Returns:
            diar_hyp (Tensor):
                Speaker label hypothesis from the start of the session to the current position
        """
        self._transfer_timestamps_to_segmentor()

        # In case buffer is not filled or there is no speech activity in the input
        if self.buffer_start < 0 or len(vad_timestamps) == 0:
            return self._get_interim_output()

        # Segmentation: (c.f. see `diarize` function in ClusteringDiarizer class)
        for scale_idx, (window, shift) in self.multiscale_args_dict['scale_dict'].items():

            # Step 1: Get subsegments for embedding extraction.
            audio_sigs, segment_ranges, range_inds = self.online_segmentor.run_online_segmentation(
                audio_buffer=audio_buffer,
                vad_timestamps=vad_timestamps,
                segment_raw_audio=self.segment_raw_audio[scale_idx],
                segment_range_ts=self.segment_range_ts[scale_idx],
                segment_indexes=self.segment_indexes[scale_idx],
                window=window,
                shift=shift,
            )
            self.segment_raw_audio[scale_idx] = audio_sigs
            self.segment_range_ts[scale_idx] = segment_ranges
            self.segment_indexes[scale_idx] = range_inds

            # Step 2-1: Extract speaker embeddings from the extracted subsegment timestamps.
            embeddings = self._extract_online_embeddings(
                audio_signal=self.segment_raw_audio[scale_idx],
                segment_ranges=self.segment_range_ts[scale_idx],
                embeddings=self.emb_vectors[scale_idx],
            )

            # Step 2-2:Save the embeddings and segmentation timestamps in memory
            self.emb_vectors[scale_idx] = embeddings

            self.multiscale_embeddings_and_timestamps[scale_idx] = [
                {self.uniq_id: embeddings},
                {self.uniq_id: segment_ranges},
            ]

        embs_and_timestamps = get_embs_and_timestamps(
            self.multiscale_embeddings_and_timestamps, self.multiscale_args_dict
        )

        # Step 3 - Clustering: Perform an online version of clustering algorithm
        cluster_label_hyp = self._perform_online_clustering(embs_and_timestamps[self.uniq_id], cuda=self.cuda,)

        # Step 4: Generate RTTM style diarization labels from segment ranges and cluster labels
        diar_hyp, _ = generate_cluster_labels(self.memory_segment_ranges[self.base_scale_index], cluster_label_hyp)
        return diar_hyp
