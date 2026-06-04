# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, List, Tuple
import torch
from tqdm import tqdm
from nemo.collections.asr.parts.utils.offline_clustering import (
    SpeakerClustering,
    get_scale_interpolated_embs,
    getCosAffinityMatrix,
    split_input_data,
)
from nemo.collections.asr.parts.utils.online_clustering import get_merge_quantity, run_reducer


class LongFormSpeakerClustering(torch.nn.Module):
    def __init__(self, cuda: bool = False):
        """
        Initializes a speaker clustering class tailored for long-form audio, leveraging methods from the `SpeakerClustering` class.
        The clustering algorithm for long-form content is executed via the `forward_infer` function (not shown here). Input embedding
        vectors are divided into chunks, each of size `embeddings_per_chunk`. Within every chunk, the clustering algorithm aims
        to identify `chunk_cluster_count` distinct clusters. The resulting clustering labels are then expanded to match the original
        length of the input embeddings.

        NOTE: torch.jit.script currently does not support inherited methods with a `super()` call.

        Args:
            cuda (bool):
                Flag indicating whether CUDA is available for computation.
        """
        super().__init__()
        self.speaker_clustering = SpeakerClustering(cuda=cuda)
        self.embeddings_in_scales: List[torch.Tensor] = [torch.tensor([0])]
        self.timestamps_in_scales: List[torch.Tensor] = [torch.tensor([0])]
        self.cuda = cuda
        self.device = torch.device("cuda") if self.cuda else torch.device("cpu")

    def check_input(self, embeddings_per_chunk: int, chunk_cluster_count: int, max_num_speakers: int) -> None:
        """
        Checks the validity of the input parameters.

        Args:
            embeddings_per_chunk (int):
                The size of the windows in which the algorithm aims to identify `chunk_cluster_count` clusters.
            chunk_cluster_count (int):
                The target number of clusters to identify within each window.
            max_num_speakers (int):
                The maximum number of speakers to be detected in the audio.
        """
        if chunk_cluster_count is None or embeddings_per_chunk is None:
            raise ValueError(
                f"chunk_cluster_count ({chunk_cluster_count}) and embeddings_per_chunk ({embeddings_per_chunk}) should be set."
            )
        elif (
            all(v is not None for v in [chunk_cluster_count, embeddings_per_chunk])
            and chunk_cluster_count >= embeddings_per_chunk
        ):
            raise ValueError(
                f"chunk_cluster_count ({chunk_cluster_count}) should be smaller than embeddings_per_chunk ({embeddings_per_chunk})."
            )

        if chunk_cluster_count <= max_num_speakers:
            raise ValueError(
                f"chunk_cluster_count ({chunk_cluster_count}) should be larger than max_num_speakers ({max_num_speakers})."
            )

    def unpack_labels(
        self,
        Y_aggr: torch.Tensor,
        window_range_list: List[List[int]],
        absolute_merge_mapping: List[List[torch.Tensor]],
        org_len: int,
    ) -> torch.LongTensor:
        """
        Unpack the labels from the aggregated labels to the original labels.

        Args:
            Y_aggr (Tensor):
                Aggregated label vector from the merged segments.
            window_range_list (List[List[int]]):
                List of window ranges for each of the merged segments.
            absolute_merge_mapping (List[List[torch.Tensor]]):
                List of absolute mappings for each of the merged segments. Each list element contains two tensors:
                    - The first tensor represents the absolute index of the bypassed segment (segments that remain unchanged).
                    - The second tensor represents the absolute index of the merged segment (segments that have had their indexes changed).
            org_len (int):
                Original length of the labels. In most cases, this is a fairly large number (on the order of 10^5).

        Returns:
            Y_unpack (Tensor):
                Unpacked labels derived from the aggregated labels.
        """
        Y_unpack = torch.zeros((org_len,)).long().to(Y_aggr.device)
        for win_rng, abs_mapping in zip(window_range_list, absolute_merge_mapping):
            inferred_merged_embs = Y_aggr[win_rng[0] : win_rng[1]]
            if len(abs_mapping[1]) > 0:
                Y_unpack[abs_mapping[1]] = inferred_merged_embs[-1].clone()  # Merged
                if len(abs_mapping[0]) > 0:
                    Y_unpack[abs_mapping[0]] = inferred_merged_embs[:-1].clone()  # Bypass
            else:
                if len(abs_mapping[0]) > 0:
                    Y_unpack[abs_mapping[0]] = inferred_merged_embs.clone()
        return Y_unpack

    def split_embs_to_windows(
        self,
        index: int,
        emb: torch.Tensor,
        embeddings_per_chunk: int,
    ) -> Tuple[torch.Tensor, int]:
        """
        Splits the embedding tensor into smaller window-sized tensors based on a given index.

        Args:
            index (int): The index of the desired window. This determines the starting point
                         of the window using the formula:
                         start = embeddings_per_chunk * index
            emb (Tensor): The embedding tensor which needs to be split.
            embeddings_per_chunk (int):
                The size of the windows in which the algorithm aims to identify `chunk_cluster_count` clusters.

        Returns:
            emb_part (Tensor):
                The window-sized tensor, which is a portion of the `emb`.
            offset_index (int):
                The starting position of the window in the `emb` tensor.
        """
        if embeddings_per_chunk * (index + 1) > emb.shape[0]:
            emb_part = emb[-1 * embeddings_per_chunk :]
            offset_index = emb.shape[0] - embeddings_per_chunk
        else:
            emb_part = emb[embeddings_per_chunk * index : embeddings_per_chunk * (index + 1)]
            offset_index = embeddings_per_chunk * index
        return emb_part, offset_index

    def forward(self, param_dict: Dict[str, torch.Tensor]) -> torch.LongTensor:
        """
        A function wrapper designed for performing inference using an exported script format.

        Note:
            A dictionary is used to facilitate inference with the exported jit model in the Triton server.
            This is done using an easy-to-understand naming convention.
            See https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#special-conventions-for-pytorch-backend

        Args:
            param_dict (dict):
                Dictionary containing the arguments for speaker clustering.
                See `forward_infer` function for the argument information.

        Returns:
            (LongTensor): Speaker labels for the segments in the given input embeddings.
        """
        embeddings_in_scales = param_dict['embeddings']
        timestamps_in_scales = param_dict['timestamps']
        multiscale_segment_counts = param_dict['multiscale_segment_counts']
        multiscale_weights = param_dict['multiscale_weights']
        oracle_num_speakers = int(param_dict['oracle_num_speakers'].item())
        max_num_speakers = int(param_dict['max_num_speakers'].item())
        enhanced_count_thres = int(param_dict['enhanced_count_thres'].item())
        sparse_search_volume = int(param_dict['sparse_search_volume'].item())
        max_rp_threshold = float(param_dict['max_rp_threshold'].item())
        fixed_thres = float(param_dict['fixed_thres'].item())
        return self.forward_infer(
            embeddings_in_scales=embeddings_in_scales,
            timestamps_in_scales=timestamps_in_scales,
            multiscale_segment_counts=multiscale_segment_counts,
            multiscale_weights=multiscale_weights,
            oracle_num_speakers=oracle_num_speakers,
            max_rp_threshold=max_rp_threshold,
            max_num_speakers=max_num_speakers,
            enhanced_count_thres=enhanced_count_thres,
            sparse_search_volume=sparse_search_volume,
            fixed_thres=fixed_thres,
        )

    def get_div_ceil_count(self, numer: int, denomin: int) -> int:
        """
        Calculates the ceiling of the division of two integers.

        Args:
            numer (int): Numerator, the number of segments or clusters, for example.
            denomin (int): Denominator, the number of speakers or clusters, for example.

        Returns:
            (int): The ceiling of the division of the two integers (number of chunks).
        """
        return int(torch.ceil(torch.tensor(numer / denomin)).item())

    def long_forward_infer(
        self,
        embeddings_in_scales: torch.Tensor,
        timestamps_in_scales: torch.Tensor,
        multiscale_segment_counts: torch.LongTensor,
        multiscale_weights: torch.Tensor,
        oracle_num_speakers: int,
        max_rp_threshold: float,
        max_num_speakers: int,
        sparse_search_volume: int,
        fixed_thres: float,
        chunk_cluster_count: int,
        embeddings_per_chunk: int,
    ) -> torch.LongTensor:
        """
        This is forward function for long-form speaker clustering.
        Please refer to `SpeakerClustering` class for the original argument information.
        
        In the `LongFormSpeakerClustering` process:
            Step-1: Input embeddings are divided into smaller windows of size `embeddings_per_chunk`.
            Step-2: Each window undergoes overclustering, resulting in `chunk_cluster_count` fine-grained clusters.
            Step-3: These fine-grained clusters are merged to form the aggregated clustering labels `Y_aggr`.
            Step-4: The `unpack_labels` function is then employed to map the aggregated labels `Y_aggr` back to the 
            original labels for all `org_len` input embeddings: `Y_unpack`.
        
        Args:
            embeddings_in_scales (Tensor):
                List containing concatenated Torch tensor embeddings across multiple scales.
                The length of the list is equal to the number of scales.
                Each tensor has dimensions of (Number of base segments) x (Embedding Dimension).
            timestamps_in_scales (Tensor):
                List containing concatenated Torch tensor timestamps across multiple scales.
                The length of the list is equal to the number of scales.
                Each tensor has dimensions of (Total number of segments across all scales) x 2.
                Example:
                    >>> timestamps_in_scales[0] = \
                        torch.Tensor([[0.4, 1.4], [0.9, 1.9], [1.4, 2.4], ... [121.2, 122.2]])
            multiscale_segment_counts (LongTensor):
                A Torch tensor containing the number of segments for each scale.
                The tensor has dimensions of (Number of scales).
                Example:
                    >>> multiscale_segment_counts = torch.LongTensor([31, 52, 84, 105, 120])
            multiscale_weights (Tensor):
                Multi-scale weights used when merging affinity scores.
                Example:
                    >>> multiscale_weights = torch.tensor([1.4, 1.3, 1.2, 1.1, 1.0])
            oracle_num_speakers (int):
                The number of speakers in a session as given by the reference transcript.
            max_num_speakers (int):
                The upper bound for the number of speakers in each session.
            max_rp_threshold (float):
                Limits the range of parameter search.
                The clustering performance can vary based on this range.
                The default value is 0.15.
            enhanced_count_thres (int):
                For shorter audio recordings, the clustering algorithm might not accumulate enough speaker profiles for each cluster.
                Thus, the function `getEnhancedSpeakerCount` uses anchor embeddings (dummy representations) to mitigate the effects of cluster sparsity.
                A value of 80 is recommended for `enhanced_count_thres`.
            sparse_search_volume (int):
                The number of p_values considered during NME analysis.
                The default is 30. Lower values speed up the NME-analysis but might lead to poorer parameter estimations. Values below 20 are not recommended.
            fixed_thres (float):
                If a `fixed_thres` value is provided, the NME-analysis process will be skipped.
                This value should be optimized on a development set for best results.
                By default, it is set to -1.0, and the function performs NME-analysis to estimate the threshold.
            kmeans_random_trials (int):
                The number of random trials for initializing k-means clustering. More trials can result in more stable clustering. The default is 1.
            chunk_cluster_count (int):
                The target number of clusters to identify within each chunk.
            embeddings_per_chunk (int):
                The size of the chunks in which the algorithm aims to identify `chunk_cluster_count` clusters.

        Returns:
            Y_unpack (LongTensor):
                Speaker labels for the segments in the provided input embeddings.
        """
        self.check_input(embeddings_per_chunk, chunk_cluster_count, max_num_speakers)

        self.embeddings_in_scales, self.timestamps_in_scales = split_input_data(
            embeddings_in_scales, timestamps_in_scales, multiscale_segment_counts
        )
        emb, _ = get_scale_interpolated_embs(
            multiscale_weights, self.embeddings_in_scales, self.timestamps_in_scales, self.device
        )
        offset_index: int = 0
        window_offset: int = 0
        total_emb: List[torch.Tensor] = []
        window_range_list: List[List[int]] = []
        absolute_merge_mapping: List[List[torch.Tensor]] = []
        total_window_count = self.get_div_ceil_count(numer=emb.shape[0], denomin=embeddings_per_chunk)

        if not torch.jit.is_scripting():
            pbar = tqdm(range(total_window_count), desc="Clustering Sub-Windows", leave=True, unit="window")
        else:
            pbar = range(total_window_count)

        for win_index in pbar:
            # Step-1: Split the embeddings into smaller chunks
            emb_part, offset_index = self.split_embs_to_windows(
                index=win_index, emb=emb, embeddings_per_chunk=embeddings_per_chunk
            )

            # Step-2: Perform overclustering on the chunks to identify `chunk_cluster_count` clusters
            if emb_part.shape[0] == 1:
                Y_part = torch.zeros((1,), dtype=torch.int64)
            else:
                mat = getCosAffinityMatrix(emb_part)
                overcluster_count = min(chunk_cluster_count, mat.shape[0])
                Y_part = self.speaker_clustering.forward_unit_infer(
                    mat=mat,
                    oracle_num_speakers=overcluster_count,
                    max_rp_threshold=max_rp_threshold,
                    max_num_speakers=chunk_cluster_count,
                    sparse_search_volume=sparse_search_volume,
                )

            # Step-3: Merge the clusters to form the aggregated clustering labels `Y_aggr`
            num_to_be_merged = int(min(embeddings_per_chunk, emb_part.shape[0]) - chunk_cluster_count)
            min_count_per_cluster = self.get_div_ceil_count(
                numer=chunk_cluster_count, denomin=len(torch.unique(Y_part))
            )

            # We want only one embedding vector for each cluster, so we calculate the number of embedding vectors to be removed
            class_target_vol = get_merge_quantity(
                num_to_be_removed=num_to_be_merged,
                pre_clus_labels=Y_part,
                min_count_per_cluster=min_count_per_cluster,
            )
            if not torch.jit.is_scripting():
                pbar.update(1)

            # `class_target_vol` is a list of cluster-indices from overclustering
            for spk_idx, merge_quantity in enumerate(list(class_target_vol)):
                merged_embs, merged_clus_labels, index_mapping = run_reducer(
                    pre_embs=emb_part,
                    target_spk_idx=spk_idx,
                    merge_quantity=merge_quantity,
                    pre_clus_labels=Y_part,
                )
                total_emb.append(merged_embs)
                absolute_index_mapping = [x + offset_index for x in index_mapping]
                absolute_merge_mapping.append(absolute_index_mapping)
                window_range_list.append([window_offset, window_offset + merged_embs.shape[0]])
                window_offset += merged_embs.shape[0]

        if not torch.jit.is_scripting():
            pbar.close()

        # Concatenate the reduced embeddings then perform high-level clustering
        reduced_embs = torch.cat(total_emb)
        reduced_mat = getCosAffinityMatrix(reduced_embs)

        # Step-4: Map the aggregated labels `Y_aggr` back to the original labels for all `org_len` input embeddings: `Y_unpack`
        Y_aggr = self.speaker_clustering.forward_unit_infer(
            mat=reduced_mat,
            oracle_num_speakers=oracle_num_speakers,
            max_rp_threshold=max_rp_threshold,
            max_num_speakers=max_num_speakers,
            sparse_search_volume=sparse_search_volume,
            fixed_thres=fixed_thres,
        )
        if reduced_embs.shape[0] != Y_aggr.shape[0]:
            raise ValueError(
                f"The number of embeddings ({reduced_embs.shape[0]}) and the number of clustered labels ({Y_aggr.shape[0]}) do not match."
            )

        # Reassign the labels to the original embeddings
        Y_unpack = self.unpack_labels(
            Y_aggr=Y_aggr,
            window_range_list=window_range_list,
            absolute_merge_mapping=absolute_merge_mapping,
            org_len=emb.shape[0],
        )
        if Y_unpack.shape[0] != emb.shape[0]:
            raise ValueError(
                f"The number of raw input embeddings ({emb.shape[0]}) and the number of clustered labels ({Y_unpack.shape[0]}) do not match."
            )
        return Y_unpack

    def forward_infer(
        self,
        embeddings_in_scales: torch.Tensor,
        timestamps_in_scales: torch.Tensor,
        multiscale_segment_counts: torch.LongTensor,
        multiscale_weights: torch.Tensor,
        oracle_num_speakers: int = -1,
        max_rp_threshold: float = 0.15,
        max_num_speakers: int = 8,
        enhanced_count_thres: int = 80,
        sparse_search_volume: int = 30,
        fixed_thres: float = -1.0,
        chunk_cluster_count: int = 50,
        embeddings_per_chunk: int = 10000,
    ) -> torch.LongTensor:
        """
        This function is a wrapper designed for toggling between long-form and short-form speaker clustering.
        The details of short-form clustering is in `SpeakerClustering` class.
        NOTE: `torch.jit.script` currently does not support `**kwargs` in the function signature therefore,
        we need to use a wrapper function to handle the arguments.
        """
        if embeddings_per_chunk is not None and torch.max(multiscale_segment_counts) > embeddings_per_chunk:
            return self.long_forward_infer(
                embeddings_in_scales=embeddings_in_scales,
                timestamps_in_scales=timestamps_in_scales,
                multiscale_segment_counts=multiscale_segment_counts,
                multiscale_weights=multiscale_weights,
                oracle_num_speakers=oracle_num_speakers,
                max_rp_threshold=max_rp_threshold,
                max_num_speakers=max_num_speakers,
                sparse_search_volume=sparse_search_volume,
                fixed_thres=fixed_thres,
                chunk_cluster_count=chunk_cluster_count,
                embeddings_per_chunk=embeddings_per_chunk,
            )
        else:
            cluster_labels = self.speaker_clustering.forward_infer(
                embeddings_in_scales=embeddings_in_scales,
                timestamps_in_scales=timestamps_in_scales,
                multiscale_segment_counts=multiscale_segment_counts,
                multiscale_weights=multiscale_weights,
                oracle_num_speakers=oracle_num_speakers,
                max_rp_threshold=max_rp_threshold,
                max_num_speakers=max_num_speakers,
                enhanced_count_thres=enhanced_count_thres,
                sparse_search_volume=sparse_search_volume,
                fixed_thres=fixed_thres,
            )
            self.timestamps_in_scales = self.speaker_clustering.timestamps_in_scales
            return cluster_labels
