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

# Copyright (c) 2007-2020 The scikit-learn developers.

# BSD 3-Clause License

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# NME-SC clustering is based on the implementation from the paper
# https://arxiv.org/pdf/2003.02405.pdf and the implementation from
# https://github.com/tango4j/Auto-Tuning-Spectral-Clustering.

from typing import List, Set, Tuple

import numpy as np
import torch

from nemo.collections.asr.parts.utils.offline_clustering import (
    NMESC,
    SpectralClustering,
    getAffinityGraphMat,
    getCosAffinityMatrix,
)
from nemo.collections.asr.parts.utils.optimization_utils import linear_sum_assignment


def get_lsa_speaker_mapping(
    U_set: torch.Tensor, cmm_P: torch.Tensor, cmm_Q: torch.Tensor, PandQ: torch.Tensor
) -> torch.Tensor:
    """
    Find a mapping that minimizes the matching cost between the label P and Q.
    One-hot encodding is employed to represent sequence and calculate the cost.

    Args:
        U_set (list):
            Whole set of the estimated speakers
        cmm_P (Tensor):
            Length-matched old sequence
        cmm_Q (Tensor):
            Length-matched new sequence
        PandQ (Tensor):
            Tensor containing the indices of the speakers that are in both old and new sequences

    Returns:
        mapping_array (np.array):
            Mapped labels that minimizes the cost
    """
    all_spks_labels = [[x] for x in range(len(U_set))]
    common_inds: List[int] = [int(x.item()) for x in PandQ]

    # Create tensors for one-hot encoding
    enc_P = torch.zeros((len(cmm_P), len(all_spks_labels))).to(cmm_P.device)
    enc_Q = torch.zeros((len(cmm_Q), len(all_spks_labels))).to(cmm_Q.device)

    # Create one-hot encoding
    enc_P[torch.arange(len(cmm_P)), cmm_P] = 1
    enc_Q[torch.arange(len(cmm_Q)), cmm_Q] = 1

    # Cost matrix from one-hot encoding vectors
    cost = -1 * torch.matmul(enc_P.T, enc_Q).T.to(PandQ.device)
    _, col_ind = linear_sum_assignment(cost)

    # If number of are speakers in each vector is not the same
    mapping_array = torch.arange(0, len(U_set)).to(PandQ.device)
    for x in range(col_ind.shape[0]):
        if x not in common_inds:
            mapping_array[x] = x
        else:
            mapping_array[x] = col_ind[x]
    return mapping_array


def get_minimal_indices(Y_new: torch.Tensor) -> torch.Tensor:
    """
    Force the unique indices of the labels to use the lowest numbers.

    Example:
        >>> Y_new = [3, 3, 3, 4, 4, 5]
        >>> get_minimal_indices(Y_new)
        Return:
            [0, 0, 0, 1, 1, 2]

    Args:
        Y_new (Tensor):
            Tensor containing cluster labels

    Returns:
        (Tensor): Newly mapped cluster labels that has minimized indicies
    """
    device = Y_new.device
    Y_new_enlisted = torch.unique(Y_new).sort()[0].to(torch.long).to(device)
    sequence = torch.arange(torch.max(Y_new_enlisted) + 1).to(device)
    sequence[Y_new_enlisted] = torch.arange(len(Y_new_enlisted)).to(device)
    return sequence[Y_new]


@torch.jit.script
def stitch_cluster_labels(Y_old: torch.Tensor, Y_new: torch.Tensor) -> torch.Tensor:
    """
    Run Hungarian (linear sum assignment) algorithm to find the best permutation mapping between
    the cumulated labels in history and the new clustering output labels.

    Args:
        Y_old (Tensor):
            Cumulated diarization labels. This will be concatenated with history embedding speaker label
            then compared with the predicted label Y_new.
        Y_new (Tensor):
            Contains predicted labels for reduced history embeddings concatenated with the predicted label.
            Permutation is not matched yet.

    Returns:
        mapping_array[Y] (Tensor):
            An output numpy array where the input Y_new is mapped with mapping_array.
    """
    Y_new = get_minimal_indices(Y_new)
    if len(Y_old) == 0:
        matched_output = Y_new
    else:
        P_raw, Q_raw = Y_old.to(Y_new.device), Y_new
        U_set = torch.unique(torch.cat([P_raw, Q_raw]))
        PQ = torch.cat([P_raw, Q_raw])
        a_cat_b, counts = torch.unique(PQ, return_counts=True)
        # Get a union set of old P and new Q labels
        PandQ = a_cat_b[torch.where(counts.gt(1))[0]]
        min_len = min(P_raw.shape[0], Q_raw.shape[0])
        P, Q = P_raw[:min_len], Q_raw[:min_len]

        if len(U_set) == 1:
            # When two speaker vectors are exactly the same: No need to encode.
            mapping_array = torch.tensor([0, 0]).to(Y_new.device)
        else:
            # Run Hungarian algorithm if there are more than one speaker in universal set U.
            mapping_array = get_lsa_speaker_mapping(U_set=U_set, cmm_P=P, cmm_Q=Q, PandQ=PandQ)
        matched_output = mapping_array[Y_new]
    matched_output = get_minimal_indices(matched_output)
    return matched_output


def calculate_removable_counts(removable_counts_mat: torch.Tensor, remain_count: int, num_clus: int) -> torch.Tensor:
    """
    Calculate removable counts based on the arguments and calculate how many counts should be
    removed from the each cluster. This function has `O(N)` (N = num_clus) time complexity to
    return the desired `removable_counts_mat`.

    Example:

        The original input to `get_merge_quantity` function:
        >>> pre_clus_labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        >>> num_to_be_removed = 3
        >>> min_count_per_cluster = 2

        Histogram: (`min_count_per_cluster`=2 is removed)
        0 |*****
        1 |***
        2 |*

        Inputs:
            >>> removable_counts_mat = [5, 3, 1]
            >>> remain_count = 6
            >>> num_clus = 3
        
        Interim results:
            >>> diff_counts 
            [1, 2, 2]
            >>> gradual_counts
            [3, 4, 2]
            >>> cumsum_counts
            [3, 7, 9]

        Return:
            >>> removable_counts_mat 
            [2, 1, 0]

    Args:
        removable_counts_mat (Tensor):
            Tensor containing how many vectors could be removed from each cluster
        remain_count (int):
            Integer value that indicates the number of vectors removed from the total set
        num_clus (int):
            Number of clusters in the given label sequence (cardinality of a label set)

    Returns:
        removable_counts_mat (Tensor):
            Tensor containing the number of vectors should be removed from each cluster
    """
    device = removable_counts_mat.device
    zero_padded_counts = torch.cat(
        [torch.tensor([0]).to(device), removable_counts_mat.sort()[0], torch.tensor([0]).to(device)], dim=0
    )
    removable_count_args = removable_counts_mat.sort(descending=True)[1]

    # Calculate the size difference between clusters
    diff_counts = (zero_padded_counts[1:] - zero_padded_counts[:-1])[:num_clus]
    gradual_counts = torch.arange(num_clus, 0, -1).to(device) * diff_counts
    cumsum_counts = torch.cumsum(gradual_counts, dim=0)
    remain_count_rem = remain_count

    # Find how many remaining counts we can use
    ind: int = 0
    for ind, num in enumerate(cumsum_counts):
        if remain_count < num:
            break

    # Subtract the common values step by step
    if ind > 0:
        for knd in range(ind):
            removable_counts_mat[removable_count_args[: num_clus - knd]] -= diff_counts[knd]
            remain_count_rem -= int(diff_counts[knd].item()) * (num_clus - knd)
    assert remain_count >= 0, "remain_count should never be negative."

    # Add remaining values
    num_labels = remain_count_rem // (num_clus - ind)
    rem_labels = remain_count_rem % (num_clus - ind)
    removable_counts_mat[removable_count_args[: (num_clus - ind)]] -= num_labels
    removable_counts_mat[removable_count_args[:rem_labels]] -= 1
    return removable_counts_mat


def get_merge_quantity(
    num_to_be_removed: int, pre_clus_labels: torch.Tensor, min_count_per_cluster: int,
) -> torch.Tensor:
    """
    Determine which embeddings we need to reduce or merge in history buffer.
    We want to merge or remove the embedding in the bigger cluster first.
    At the same time, we keep the minimum number of embedding per cluster
    with the variable named min_count_per_cluster.

    Constraint:
        - Each cluster should keep the number of vectors over `min_count_per_cluster`.
        - In total, `num_to_be_removed` of vectors should be removed from the total buffer.
        - While merging embeddings, minimize the gap between quantities between clusters.

    Example:
        >>> num_to_be_removed = 3
        >>> pre_clus_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        >>> min_count_per_cluster = 2
        >>> get_merge_quantity(num_to_be_removed, pre_clus_labels, min_count_per_cluster)
        Return:   
            torch.tensor([2, 1, 0]) 
        >>> # Sum should be equal to `num_to_be_removed` which is 3

    Args:
        num_to_be_removed: (int)
            the quantity of the newly obtained embedding from the new stream of input.
        pre_clus_labels: (Tensor)
            the speaker labels of (the history_embedding_buffer_emb) + (the new embeddings to be added)
        min_count_per_cluster: (int)
            Minimum vector quantity for each cluster

    Returns:
        removable_counts_mat: (Tensor)
            Tensor containing the number of vectors should be removed from each cluster
    """
    if num_to_be_removed > pre_clus_labels.shape[0] - 1:
        raise ValueError(f"num_to_be_removed: {num_to_be_removed} should be less than pre_clus_labels length - 1")
    remain_count = pre_clus_labels.shape[0] - num_to_be_removed
    spk_freq_count = torch.bincount(pre_clus_labels)
    num_clus = len(torch.unique(pre_clus_labels))
    if remain_count < min_count_per_cluster * num_clus:
        raise ValueError(f"The remaining embedding vectors should be more than { min_count_per_cluster * num_clus }")

    # Minimum vector counts should be excluded from the removable amount
    min_seg_count = torch.tensor([min_count_per_cluster] * len(spk_freq_count)).to(pre_clus_labels.device)
    min_seg_count_mat = torch.stack((min_seg_count, spk_freq_count)).min(0)[0]

    # Exclude minimum quantities from the removable count matrix
    remain_count -= int(torch.sum(min_seg_count_mat))
    removable_counts_mat = spk_freq_count - min_seg_count_mat

    # Calculate removable counts from `remain_count` variable
    removable_counts_mat = calculate_removable_counts(removable_counts_mat, remain_count, num_clus)
    if int(removable_counts_mat.sum()) != num_to_be_removed:
        raise ValueError("Sum of `removable_counts_mat` is not equal to `num_to_be_removed` variable.")
    if not torch.all(removable_counts_mat >= 0) or not torch.all(spk_freq_count - min_seg_count_mat >= 0):
        raise ValueError(
            f"Every value in `removable_counts_mat` should be always non-negative value but got {removable_counts_mat}"
        )
    return removable_counts_mat


def merge_vectors(
    selected_inds: torch.Tensor, emb_ndx: torch.Tensor, pre_cluster_labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge feature (embedding) vectors estimated to be the same cluster label.

    Args:
        selected_inds (Tensor):
            Selected indices for merging
        emb_ndx (Tensor):
            Feature (embedding) vectors
            Dimension: (original vector counts) x (feature dimension)
        pre_cluster_labels (Tensor):
            Original cluster labels before merging

    Returns:
        merged_vecs (Tensor):
            Merged feature vectors that are concatenated
            Dimension: (merged vector counts) x (feature dimension)
        merged_clus_labels (Tensor):
            Cluster labels for the merged feature vectors
            Dimension: (merged vector counts)
    """
    if emb_ndx.shape[0] != pre_cluster_labels.shape[0]:
        raise ValueError("pre_cluster_labels and emb_ndx have mismatch in dimension")
    avg_emb = torch.mean(emb_ndx[selected_inds, :], dim=0)
    merged_clus_labels = pre_cluster_labels[selected_inds]
    selected_inds_list: List[int] = selected_inds.tolist()
    bypass_inds_list: List[int] = []
    for k in range(emb_ndx.shape[0]):
        if k not in selected_inds_list:
            bypass_inds_list.append(k)
    bypass_inds = torch.tensor(bypass_inds_list)
    selected_inds = torch.tensor(selected_inds_list)
    merged_vecs = torch.vstack((emb_ndx[bypass_inds], avg_emb))
    merged_clus_labels = torch.hstack((pre_cluster_labels[bypass_inds], merged_clus_labels[0]))
    return merged_vecs, merged_clus_labels


def get_closest_embeddings(affinity_mat: torch.Tensor, n_closest: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the indices of the embedding vectors we want to merge.

    Example:
        >>> n_closest = 2
        >>> affinity_mat = [[1.0, 0.2, 0.8],
                            [0.2, 1.0, 0.4],
                            [0.8, 0.4, 1.0]]
        >>> affinity_mat.sum(0) 
        [2.0, 1.6, 2.2]

        # The closest two embedding vectors are at index 0 and 2.

    Args:
        affinity_mat: (Tensor)
            Symmetric affinity matrix of the given embedding vector set.
        n_closest (int):
            The amount of vector counts that are expected to be removed from the set
            Example:
                Input: 10 vectors in a set
                n_closest = 5
                (5+1) vectors are merged into 1 vector
                Output: 5 vectors in a set

    Returns:
        idx_aff_sum (torch.Tensor):
            Indices of the closest `n_closest` embedding vectors
        rest_inds (torch.Tensor):
            Indices of the complementary set of the indices in `idx_aff_sum`
    """
    comb_limit = int(affinity_mat.shape[0] - 1)
    if n_closest > comb_limit:
        raise ValueError(f"Got n_closest of {n_closest}: {n_closest} is bigger than comb_limit {comb_limit}")

    # Take summed values over one axis
    sum_cmat = affinity_mat.sum(0)

    # `n_closest + 1` will become 1 embedding vector after merging
    idx_aff_sum = torch.argsort(sum_cmat, descending=True)[: (n_closest + 1)]
    rest_inds = torch.argsort(sum_cmat, descending=True)[(n_closest + 1) :]
    return idx_aff_sum, rest_inds


def run_reducer(
    pre_embs: torch.Tensor, target_spk_idx: int, merge_quantity: int, pre_clus_labels: torch.Tensor,
):
    """
    Reduce the number of embedding vectors by merging the closest embedding vectors.
        - This merging algorithm is based on the assumption that the closest embeddings 
          are the most redundant embedding vectors.
        - The closest embedding vectors are chosen by selecting the highest top-N sum of 
          each column in a given affinity matrix.
        - If merge_quantity is N, we choose (N+1) vectors into 1 embedding vector. 
          Thus, we reduce N embeddings in the original embedding vector set.

    Example:
        >>> merge_quantity = 1 # We merge 1+1 = 2 embedding vectors
        >>> affinity_mat = [[1.0, 0.2, 0.8],
                            [0.2, 1.0, 0.4],
                            [0.8, 0.4, 1.0]]
        >>> affinity_mat.sum(0) 
        [2.0, 1.6, 2.2]

        The first and the third embedding vectors are merged into one embedding vector.
        >>> index_mapping # (bypassed indices, merged indices)
        ([1], [0, 2]) 

    Args:
        pre_embs (Tensor):
            Potential Embedding vectors to be merged
        affinity_mat (Tensor):
            The affinity matrix of the `pre_embs`
        target_spk_idx (int):
            The targeted speaker index for merging
        merge_quantity (int):
            The count of embeddings to be reduced
        pre_clus_labels (list)
            The original cluster (speaker) index

    Returns:
        merged_embs (torch.Tensor):    
            The merged embedding vectors.
        merged_clus_labels (torch.Tensor):
            The cluster (speaker) indices for the merged embedding vectors.
        index_mapping (Tuple[torch.Tensor, torch.Tensor]):
            A tuple containing the indices of the original embeddings that were not merged (`bypassed indices`)
            and the indices of the new merged embeddings (`merged indices`).
    """
    if pre_embs.shape[0] != pre_clus_labels.shape[0]:
        raise ValueError("Dimension mismatch between `pre_embs` and `pre_clus_labels`.")

    target_emb_index = torch.where(pre_clus_labels == target_spk_idx)[0]
    org_size = target_emb_index.shape[0]
    if merge_quantity > 0:
        if merge_quantity > (target_emb_index.shape[0] - 1):
            raise ValueError(
                f"merge_quantity {merge_quantity} should not be larger than target_emb_index length: {target_emb_index.shape[0]-1}"
            )
        total_affinity_mat = getCosAffinityMatrix(pre_embs)

        # Get the lower triangle of the affinity_mat array
        affinity_mat = total_affinity_mat[:, target_emb_index][target_emb_index, :]
        if affinity_mat.shape[0] != target_emb_index.shape[0]:
            raise ValueError(
                "Dimension mismatch between targeted speaker affinity `affinity_mat` and targeted speaker index `target_emb_index`."
            )
        # Get the indices of the closest embedding vectors
        selected_inds, rest_inds = get_closest_embeddings(affinity_mat, merge_quantity)
        spk_cluster_labels, selected_embs = pre_clus_labels[target_emb_index], pre_embs[target_emb_index]

        # Note that we need to return the indices of speaker-specific indices from `target_emb_index`.
        index_mapping = (target_emb_index[rest_inds.sort()[0]], target_emb_index[selected_inds])

        # Merge the embeddings targeted by the 2-dim indices `index_2d`
        merged_embs, merged_clus_labels = merge_vectors(selected_inds, selected_embs, spk_cluster_labels)

        if (org_size - merge_quantity) != merged_embs.shape[0]:
            raise ValueError(
                f"Reducer output {merged_embs.shape[0]} is not matched to the target quantity {org_size - merge_quantity}."
            )

    else:
        merged_embs = pre_embs[target_emb_index]
        merged_clus_labels = pre_clus_labels[target_emb_index]
        index_mapping = (target_emb_index, torch.arange(0))
    return merged_embs, merged_clus_labels, index_mapping


def get_first_arg_index(mat: torch.Tensor, label: int) -> int:
    """
    Get the index of the first element are specified by `index` variable.

    Args:
        mat (Tensor):
            Source matrix filled with indices
        label (int):
            Label which we want to find the first occuring index

    Returns:
        (int) The first index of the given label
    """
    return int(torch.where(mat == label)[0][0])


class OnlineSpeakerClustering(torch.nn.Module):
    """
    Online clustering method for speaker diarization based on cosine similarity.

    Regular Clustering Attributes:

        max_num_speakers (int):
            The upper bound for the number of speakers in each session
        max_rp_threshold (float):
            Limits the range of parameter search.
            Clustering performance can vary depending on this range.
            Default is 0.15.
        enhanced_count_thres (int):
            For the short audio recordings, clustering algorithm cannot
            accumulate enough amount of speaker profile for each cluster.
            Thus, function `getEnhancedSpeakerCount` employs anchor embeddings
            (dummy representations) to mitigate the effect of cluster sparsity.
            enhanced_count_thres = 40 is recommended.
        sparse_search_volume (int):
            Number of p_values we search during NME analysis.
            Default is 30. The lower the value, the faster NME-analysis becomes.
            Lower than 20 might cause a poor parameter estimation.
        fixed_thres (float):
            A fixed threshold for finding p-closest neighbors in affinity matrix for clustering.
            If fixed_thres value is provided, NME-analysis process will be skipped.
            This value should be optimized on a development set to obtain a quality result.
            Default is None and performs NME-analysis to estimate the threshold.
        min_samples_for_nmesc (int):
            The minimum number of samples required for NME clustering. This avoids
            zero p_neighbour_lists. If the input has fewer segments than min_samples,
            it is directed to the enhanced speaker counting mode.
        sparse_search (bool):
            Toggle sparse search mode. If True, limit the size of p_value_list to sparse_search_volume.
        cuda (bool):
            Use cuda for Eigen decomposition if cuda=True.

    Additional Online Processing Attributes:

        history_buffer_size (int):
            - This is a buffer where diarization history is saved in the form of averaged speaker embedding vector.
            - The values in [50, 200] range is recommended while the system requires bigger buffer size for
              sessions with larger number of speakers.
        current_buffer_size (int):
            - This is a buffer which process the most recent speaker embedding vector inputs.
              current-buffer is first-in-first-out (FIFO) queue where the embeddings accepted earlier
              get to merged and saved to history buffer.
            - In general, [50, 200] range is recommended and the performance can be sensitive on this buffer size.
        min_spk_counting_buffer_size (int):
            Integer number for speaker counting buffer. Number of speakers are estimated through a small buffer
            and the number is obtained by taking majority vote.
        min_frame_per_spk (int):
            Below this number, the system considers the whole input segments as a single speaker.
        p_update_freq (int):
            Frequency (interval) of updating p_value for NMESC algorithm.
        p_value_skip_frame_thres (int):
            After `frame_index` passes this number, `p_value` estimation is skipped for inference speed
        p_value_queue_size (int):
            `p_value` buffer for major voting
        use_temporal_label_major_vote (bool):
            Boolean that determines whether to use temporal majorvoting for the final speaker labels
        temporal_label_major_vote_buffer_size (int):
            Buffer size for major-voting the
        num_spk_stat (list):
            List of number of speakers for major voting. Number of speakers are estimated through 
            majority voting of `self.num_spk_stat` list.
        p_value_hist (list):
            List of p_values for major voting.
            To save the computation time, p_value is estimated every `p_update_freq` frames and
            saved to `self.p_value_hist`.

    Attributes for counters and buffers in streaming system:
        
        is_online (bool):
            - If self.is_online is False:
                FIFO queue does not push out any speaker embedding vector
            - If self.is_online is True:
                FIFO queue starts push out speaker embedding vectors and saving them into
                history buffer.
        max_embed_count (int):
            The maximum number of segments the streaming system has ever seen.
            This value keeps increasing as the system processes more and more segments.
        memory_margin (int):
            The margin that is added to keep the segmentation data in the streaming system
        minimum_segments_per_buffer (int):
            Maximum number of embedding vectors kept in history buffer per speaker.
            Example:
                history_buffer_size (history_n) = 100
                max_num_speakers = 4
                minimum_segments_per_buffer = 25
        history_buffer_seg_end (int):
            Index that indicates the boundary between history embedding sets and current processing buffer
            when history embedding vectors and current input embedding vectors are concatenated into a
            single matrix.

    Attributes for history buffer:

        history_embedding_buffer_emb (Tensor)
            Tensor containing speaker embedding vectors for saving the history of the previous
            speaker profile in the given audio session
        history_embedding_buffer_label (Tensor)
            Speaker label (cluster label) for embedding vectors saved in the history buffer
        Y_fullhist (Tensor)
            Tensor containing the speaker label hypothesis from start to current frame
    """

    def __init__(
        self,
        max_num_speakers: int = 8,
        max_rp_threshold: float = 0.15,
        enhanced_count_thres: float = 40,
        fixed_thres: float = -1.0,
        sparse_search_volume: int = 10,
        history_buffer_size: int = 150,
        current_buffer_size: int = 150,
        min_spk_counting_buffer_size: int = 3,
        min_frame_per_spk: int = 15,
        p_update_freq: int = 5,
        p_value_skip_frame_thres: int = 50,
        p_value_queue_size: int = 3,
        use_temporal_label_major_vote: bool = False,
        temporal_label_major_vote_buffer_size: int = 11,
        cuda: bool = False,
    ):
        super().__init__()
        self.max_num_speakers = max_num_speakers
        self.max_rp_threshold = max_rp_threshold
        self.enhanced_count_thres = enhanced_count_thres
        self.sparse_search_volume = sparse_search_volume
        self.fixed_thres = fixed_thres
        self.history_n = history_buffer_size
        self.current_n = current_buffer_size
        self.min_spk_counting_buffer_size = min_spk_counting_buffer_size
        self.min_frame_per_spk = min_frame_per_spk
        self.p_update_freq = p_update_freq
        self.p_value_skip_frame_thres = p_value_skip_frame_thres
        self.p_value_queue_size = p_value_queue_size
        self.use_temporal_label_major_vote = use_temporal_label_major_vote
        self.temporal_label_major_vote_buffer_size = temporal_label_major_vote_buffer_size
        self.cuda = cuda
        self.num_spk_stat: List[torch.Tensor] = [torch.tensor(1)]
        self.p_value_hist: List[torch.Tensor] = [torch.tensor(2)]

        # Initialize the counters and buffers in streaming system
        self.is_online = False
        self.max_embed_count = 0
        self.memory_margin = 0
        self.minimum_segments_per_buffer = int(self.history_n / self.max_num_speakers)
        self.history_buffer_seg_end = 0

        # Initialize the streaming buffer tensors
        self.history_embedding_buffer_emb = torch.tensor([])
        self.history_embedding_buffer_label = torch.tensor([])
        self.Y_fullhist = torch.tensor([])

    def onlineNMEanalysis(self, mat_in: torch.Tensor, frame_index: int) -> Tuple[int, int]:
        """
        To save the running time, the p-value is only estimated in the beginning of the session.
        After switching to online mode, the system uses the most common estimated p-value.
        Estimating p-value requires a plenty of computational resource. The less frequent estimation of
        p-value can speed up the clustering algorithm by a huge margin.

        Args:
            mat_in (Tensor):
                Tensor containing the affinity matrix for the current segments
            frame_index (int):
                Unique index for each segment and embedding vector

        Returns:
            est_num_of_spk: (int)
                The estimated number of speakers.
            p_hat_value: (int)
                The estimated p-value from NMESC method.
        """
        nmesc = NMESC(
            mat_in,
            max_num_speakers=self.max_num_speakers,
            max_rp_threshold=self.max_rp_threshold,
            sparse_search=True,
            maj_vote_spk_count=False,
            sparse_search_volume=self.sparse_search_volume,
            fixed_thres=self.fixed_thres,
            nme_mat_size=256,
            parallelism=False,
            device=mat_in.device,
            cuda=self.cuda,
        )
        if len(self.p_value_hist) == 0 or (
            frame_index < self.p_value_skip_frame_thres and frame_index % self.p_update_freq == 0
        ):
            est_num_of_spk, p_hat_value = nmesc.forward()
            self.p_value_hist.append(p_hat_value)
            if len(self.p_value_hist) > self.p_value_queue_size:
                self.p_value_hist.pop(0)
        p_hat_int_list: List[int] = [int(p) for p in self.p_value_hist]
        p_hat_value = torch.mode(torch.tensor(p_hat_int_list))[0].item()
        output = nmesc.getEigRatio(p_hat_value)
        g_p, est_num_of_spk = output[0], output[1].int()
        return est_num_of_spk, p_hat_value

    def speaker_counter_buffer(self, est_num_of_spk: int) -> torch.Tensor:
        """
        Use a queue to avoid unstable speaker counting results.

        Args:
            est_num_of_spk (int):
                Estimated number of speakers

        Returns:
            est_num_of_spk (torch.Tensor):
                Estimated number of speakers from the speaker counting buffer.
        """
        est_num_of_spk = torch.tensor(est_num_of_spk)
        self.num_spk_stat.append(est_num_of_spk)
        if len(self.num_spk_stat) > self.min_spk_counting_buffer_size:
            self.num_spk_stat.pop(0)
        num_spk_stat_tensor = torch.tensor([int(s) for s in self.num_spk_stat])
        num_spks_bincount = torch.bincount(num_spk_stat_tensor)
        est_num_of_spk = torch.argmax(num_spks_bincount)
        return est_num_of_spk

    def limit_frames_per_speaker(self, frame_index: int, est_num_of_spk: int) -> int:
        """
        Limit the estimated number of speakers in proportion to the number of speakers.

        Args:
            frame_index (int):
                Unique index for each segment and embedding vector
            est_num_of_spk (int):
                Estimated number of speakers
        
        Returns:
            (int) Estimated number of speakers capped by `self.min_frame_per_spk`
        """
        return min(est_num_of_spk, int(1 + frame_index // self.min_frame_per_spk))

    def online_spk_num_estimation(self, mat_in: torch.Tensor, frame_index: int) -> Tuple[int, torch.Tensor]:
        """
        Online version of speaker estimation involves speaker counting buffer and application of per-speaker
        frame count limit.

        Args:
            mat_in (Tensor):
                Raw affinity matrix containing similarity values of each pair of segments
            frame_index (int)
                Unique frame index of online processing pipeline

        Returns:
            est_num_of_spk (int):
                Estimated number of speakers
            affinity_mat (Tensor):
                Affinity matrix after applying the affinity threshold with `p_hat_value`
        """
        est_num_of_spk, p_hat_value = self.onlineNMEanalysis(mat_in, frame_index)
        affinity_mat = getAffinityGraphMat(mat_in, p_hat_value)
        raw_est_num_of_spk = self.speaker_counter_buffer(est_num_of_spk)
        est_num_of_spk = self.limit_frames_per_speaker(frame_index, raw_est_num_of_spk.item())
        return est_num_of_spk, affinity_mat

    def prepare_embedding_update(
        self, emb_in: torch.Tensor, segment_indexes_matrix: torch.Tensor
    ) -> Tuple[bool, int, torch.Tensor, torch.Tensor]:
        """
        This function performs the following tasks:
            1. Decide whether to extract more embeddings or not (by setting `is_update`)
        (Only if we need update):
            2. Calculate how many embeddings should be updated (set `new_emb_n` variable)
            3. Update history embedding vectors and save it to `pre_embs`.

        We only save the index and clustering label of each embedding.

        - Case-1: The very first step
            This else statement is for the very first diarization loop.
            This is the very first reduction frame.

        - Case-2: Number of embedding vectors is increased, therefore we need to update.
            Since there are new embeddings, we push the same amount (new_emb_n)
            of old embeddings to the history buffer.
            We should also update self.history_buffer_seg_end which is a pointer.
                update to history emb: emb_in[emb_idx_stt:emb_idx_end]
                update to history label: self.Y_fullhist[label_stt:_end]

        - Case-3: Number of embedding vectors is decreased
            If the number of embeddings is decreased compared to the last trial,
            then skip embedding merging.

        Variables:
            hist_curr_boundary (int):
                The current boundary of between history buffer and current buffer.
                This is the new history-current buffer boundary while self.history_buffer_seg_end is the old one.
                Thus, the new set of embedding vectors are collected from 
                `label_stt=self.hist_buffer_seg_end` to `label_end=hist_curr_boundary`.
            total_segments_processed_count (int):
                The number of segments that are processed so far in integer format.

        Args:
            emb_in (Tensor):
                Tensor containing embedding vectors
                Dimensions: (number of embedding vectors) x (embedding dimension)
            segment_indexes_matrix (Tensor):
                Tensor containing unique segment (embedding vector) index

        Returns:
            is_update (bool):
                Boolean indicates whether to update speaker embedding vectors.
            new_emb_n (int):
                The amount of embedding vectors that are exceeding FIFO queue size.
                new_emb_n is also an amount of embedding vectors that needs to be merged in history buffer.
            pre_embs (Tensor):
                Embedding vector matrix before merging.
                The subset of `pre_embs` embedding vectors will be merged.
                Dimensions: (number of embedding vectors) x (embedding dimension)
            pre_clus_labels (Tensor):
                A set of clustering labels for each embedding vector in `pre_embs`.
        """
        total_segments_processed_count = int(segment_indexes_matrix[-1] + 1)
        hist_curr_boundary = int(total_segments_processed_count - self.current_n)
        new_emb_n: int = 0
        pre_embs: torch.Tensor = torch.empty(0)
        pre_clus_labels: torch.Tensor = torch.empty(0)
        is_update = True

        if total_segments_processed_count > self.max_embed_count:
            # Case-1: The very first step
            if len(self.history_embedding_buffer_emb) == 0:
                new_emb_n = total_segments_processed_count - (self.current_n + self.history_n)
                hist_curr_boundary_emb_idx = get_first_arg_index(segment_indexes_matrix, hist_curr_boundary)
                pre_embs = emb_in[:hist_curr_boundary_emb_idx]
                pre_clus_labels = self.Y_fullhist[:hist_curr_boundary]

            # Case-2: Number of embedding vectors is increased, need to update history and its label
            else:
                # Calculate the number of new embedding vectors: `new_emb_n`
                label_stt, label_end = self.history_buffer_seg_end, hist_curr_boundary
                new_emb_n = label_end - label_stt
                # Add embedding vectors to `pre_embs` so that we can merge it with reducer function.
                emb_idx_stt = int(get_first_arg_index(segment_indexes_matrix, label_stt))
                emb_idx_end = int(get_first_arg_index(segment_indexes_matrix, label_end))
                pre_embs = torch.vstack((self.history_embedding_buffer_emb, emb_in[emb_idx_stt:emb_idx_end]))
                # Update labels for `pre_embs`
                pre_clus_labels = torch.hstack(
                    (self.history_embedding_buffer_label, self.Y_fullhist[label_stt:label_end])
                )

            if new_emb_n > self.current_n:
                raise ValueError(
                    "new_emb_n should be less than or equal to current buffer size (self.current_n)."
                    f" Getting too many segments: {new_emb_n} for the given current buffer size {self.current_n}."
                    " Please either (1) increase buffer size or (2) use longer segment lengths to get less number of segments."
                )
            elif new_emb_n <= 0:
                raise ValueError("Segment counting error. `new_emb_n` should be a positve integer number.")
            if pre_embs.shape[0] != pre_clus_labels.shape[0]:
                raise ValueError(
                    "`pre_embs` and `pre_clus_labels` should have the same length, "
                    f"but got {pre_embs.shape[0]} and {pre_clus_labels.shape[0]} respectively."
                )

        # Case-3: Number of embedding vectors is not increased.
        else:
            # There will be no embedding update, so new_emb_n is 0, pre_embs and pre_clus_labels are empty.
            is_update = False

        # Update the history buffer index for the next step
        self.history_buffer_seg_end = hist_curr_boundary
        self.max_embed_count = max(total_segments_processed_count, self.max_embed_count)
        return is_update, new_emb_n, pre_embs, pre_clus_labels

    def make_constant_length_emb(self, emb_in: torch.Tensor, base_segment_indexes: torch.Tensor) -> torch.Tensor:
        """
        This function deals with edge cases when the number of segments decreases and the number of embedding falls
        short for the labels.

        - ASR decoder occasionally returns less number of words compared to the previous frame.
        - In this case, we obtain fewer embedding vectors for the short period of time. To match the pre-defined
          length, the last embedding vector is repeated to fill the voidness.
        - The repeated embedding will be soon replaced by the actual embeddings once the system takes new frames.

        Args:
            emb_in (Tensor):
                If self.is_online is False:
                    `pre_embs` contains only current speaker embedding inputs, which is FIFO queue
                If self.is_online is True:
                    `pre_embs` contains history buffer and FIFO queue
            base_segment_indexes (Tensor):
                Tensor containing unique segment (embedding vector) index

        Returns:
            emb_curr (Tensor):
                Length preserved speaker embedding vectors
        """
        curr_clustered_segments = torch.where(base_segment_indexes >= self.history_buffer_seg_end)[0]

        # Check if the current buffer result is falling short compared to `self.current_n`.
        if emb_in[curr_clustered_segments].shape[0] < self.current_n:
            delta_count = self.current_n - emb_in[curr_clustered_segments].shape[0]
            fill_in_emb = torch.tile(emb_in[curr_clustered_segments][-1], (delta_count, 1))
            emb_curr = torch.vstack((emb_in[curr_clustered_segments], fill_in_emb))
        else:
            emb_curr = emb_in[curr_clustered_segments]
        return emb_curr

    def update_speaker_history_buffer(
        self, emb_in: torch.Tensor, base_segment_indexes: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        """
        Merge the given embedding vectors based on the calculate affinity matrix.
        if `is_update` is True, update the history buffer .

        Args:
            emb_in (Tensor):
                If self.is_online is False:
                    `emb` contains only current speaker embedding inputs, which is FIFO queue
                If self.is_online is True:
                    `emb` contains history buffer and FIFO queue
            base_segment_indexes (Tensor):
                Tensor containing unique segment (embedding vector) index

        Returns:
            history_embedding_buffer_emb (Tensor):
                Matrix containing merged embedding vectors of the previous frames.
                This matrix is referred to as "history buffer" in this class.
            is_update (bool):
                Boolean indicates whether to update speaker

        Example:

            at the frame index where `is_online` turns to True:

            |------hist-buffer------|-----FIFO-queue-----|

            self.history_n = 20
            self.current_n = 10

            Step (1)
            |-----------------------|ABCDEF--------------|

            If we get two more segments, "NN" as in the description:
            history buffer = 20
            current buffer = 12

            Step (2)
            |-----------------------|ABCDEF--------------XY|
                                    |---------emb_in-------| 

            The newly accepted embeddings go through a FIFO queue (first come, first merge)
            history buffer = 22
            current buffer = 10

            Step (3)
            |-----------------------AB|CDEF--------------XY|
            |---------pre_embs--------|

            After merging (reducing) the embedding set gets back to the original size:
            history buffer = 20
            current buffer = 10

            Step (4)
            |======================|CDEF--------------XY|
            |-----hist_emb_buff----|
            
            After clustering, `self.Y_fullhist` is updated as:

            |0000000000011111111111|11110000110010010011|

            The dimension of `self.Y_fullhist` is (`history_n + current_n`) x 1

        self.history_buffer_seg_end (int):
            The total number of segments that have been merged from the beginning of the session.
            (=`hist_curr_boundary`)
        """
        is_update, new_emb_n, pre_embs, pre_clus_labels = self.prepare_embedding_update(emb_in, base_segment_indexes)

        # Update the history/current_buffer boundary cursor
        total_emb, total_cluster_labels = [], []

        if is_update:
            # Calculate how many embedding vectors should be reduced per speaker
            class_target_vol = get_merge_quantity(
                num_to_be_removed=new_emb_n,
                pre_clus_labels=pre_clus_labels,
                min_count_per_cluster=self.minimum_segments_per_buffer,
            )

            # Merge the segments in the history buffer
            for spk_idx, target_num in enumerate(list(class_target_vol)):
                merged_embs, merged_clus_labels, _ = run_reducer(
                    pre_embs=pre_embs,
                    target_spk_idx=spk_idx,
                    merge_quantity=target_num,
                    pre_clus_labels=pre_clus_labels,
                )
                total_emb.append(merged_embs)
                total_cluster_labels.append(merged_clus_labels)

            # Update the speaker history buffer
            self.history_embedding_buffer_emb = torch.vstack(total_emb)
            self.history_embedding_buffer_label = torch.hstack(total_cluster_labels)
            if self.history_embedding_buffer_emb.shape[0] != self.history_n:
                raise ValueError("History embedding size is not maintained correctly.")
            if len(self.history_embedding_buffer_label) != self.history_n:
                raise ValueError("History label size is not maintained correctly.")

        else:
            total_emb.append(self.history_embedding_buffer_emb)
            total_cluster_labels.append(self.history_embedding_buffer_label)

        # `emb_curr` is the incumbent set of embeddings which is the the latest.
        emb_curr = self.make_constant_length_emb(emb_in, base_segment_indexes)
        total_emb.append(emb_curr)

        # Before perform clustering, we attach the current_n number of estimated speaker labels
        # from the previous clustering result.
        total_cluster_labels.append(self.Y_fullhist[-self.current_n :])

        history_and_current_emb = torch.vstack(total_emb)
        history_and_current_labels = torch.hstack(total_cluster_labels)
        if history_and_current_emb.shape[0] != len(history_and_current_labels):
            raise ValueError("`history_and_current_emb` has a mismatch in length with `history_and_current_labels`.")
        return history_and_current_emb, is_update

    def get_reduced_mat(self, emb_in: torch.Tensor, base_segment_indexes: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Choose whether we want to add embeddings to the memory or not.
        The processing buffer has size of (self.current_n + self.history_n).

        Case-1: If margin_seg_n > 0, this means we have more embedding vectors than we can hold in the processing buffer.
            - `is_online` should be `True`
            - reduce the number of embedding vectors by merging the closest ones.
                call `update_speaker_history_buffer` function

        Case-2: If margin_seg_n <= 0, this means that we can accept more embedding vectors and yet to fill the processing buffer.
            - `is_online` should be `False`
            - Replace `merged_emb` variable with the raw input `emb_in`.
            - `add_new` is `True`, since we are adding more embedding vectors to `merged_emb` variable.

        Args:
            emb_in (Tensor):
                If self.is_online is False:
                    `emb` contains only current speaker embedding inputs
            base_segment_indexes (Tensor):
                Tensor containing unique segment (embedding vector) index

        Returns:
            merged_emb (Tensor):
                Matrix containing merged embedding vectors of the previous frames.
                This matrix is referred to as "history buffer" in this class.
                If self.is_online is False:
                    `merged_emb` contains only current speaker embedding inputs
                If self.is_online is True:
                    `merged_emb` is a concatenated matrix with history embedding and current embedding inputs
            add_new (bool):
                Boolean that indicates whether there is a new set of segments. Depending on the VAD timestamps,
                the number of subsegments can be ocassionally decreased. If `add_new=True`, then it adds the newly
                acquired cluster labels.

        """
        margin_seg_n = emb_in.shape[0] - (self.current_n + self.history_n)
        if len(self.Y_fullhist) == 0 and margin_seg_n > 0:
            raise ValueError(
                "The number of incoming embedding vectors is larger than the total processing buffer size."
                "Please either (1) increase the history and current buffer size (2) or use longer segment lengths to reduce number of segments."
            )
        if margin_seg_n > 0:
            self.is_online = True
            merged_emb, add_new = self.update_speaker_history_buffer(
                emb_in=emb_in, base_segment_indexes=base_segment_indexes
            )
        else:
            self.is_online = False
            merged_emb = emb_in
            add_new = True
        return merged_emb, add_new

    def match_labels(self, Y_merged: torch.Tensor, add_new: bool) -> torch.Tensor:
        """
        This function matches the newly generated clustering label sequence with the existing speaker labels in the history buffer.
        `self.history_buffer_seg_end` is an integer index that tells to which point is history embedding contains from `self.Y_fullhist`.

        If embedding reducing is done correctly, we should discard  (0, self.history_n) amount and take
        (self.history_n, len(Y_merged)) from the new clustering output `Y_merged`.

        Args:
            Y_merged (Tensor):
                The newly generated clustering label sequence that may have different permutations with the existing
                speaker labels in the history buffer.
            add_new (bool):
                This variable indicates whether there is a new set of segments. Depending on the VAD timestamps,
                the number of subsegments can be occasionally decreased. If `add_new=True`, then it adds the newly
                acquired cluster labels.

        Returns:
            Y_out (Tensor):
                Permutation-matched speaker labels based on history buffer
        """
        if self.is_online:
            # Online clustering mode with history buffer
            Y_old = torch.hstack((self.history_embedding_buffer_label, self.Y_fullhist[self.history_buffer_seg_end :]))

            # Stitch the old history and new cluster labels
            Y_matched = stitch_cluster_labels(Y_old=Y_old, Y_new=Y_merged).to(Y_merged.device)
            if add_new:
                if Y_matched[self.history_n :].shape[0] != self.current_n:
                    raise ValueError("Update point sync is not correct.")
                # Concatenate the newly generated speaker labels
                Y_out = torch.hstack((self.Y_fullhist[: self.history_buffer_seg_end], Y_matched[self.history_n :]))
                self.Y_fullhist = Y_out
            else:
                # Do not update cumulative labels since there are no new segments.
                Y_out = self.Y_fullhist
        else:
            # If no memory is used, offline clustering is applied.
            Y_out = stitch_cluster_labels(Y_old=self.Y_fullhist, Y_new=Y_merged).to(Y_merged.device)
            self.Y_fullhist = Y_out
        return Y_out

    def forward(
        self,
        curr_emb,
        base_segment_indexes,
        max_num_speakers: int,
        max_rp_threshold: float,
        enhanced_count_thres: int,
        sparse_search_volume: int,
        frame_index: int,
        cuda: bool = False,
    ) -> torch.Tensor:
        """
        Wrapper function for torch.jit.script compatibility.
        NOTE: jit scripted classes only contain the methods which are included in the computation graph in the forward pass.
        """
        Y = self.forward_infer(
            curr_emb=curr_emb,
            base_segment_indexes=base_segment_indexes,
            max_num_speakers=max_num_speakers,
            max_rp_threshold=max_rp_threshold,
            enhanced_count_thres=enhanced_count_thres,
            sparse_search_volume=sparse_search_volume,
            frame_index=frame_index,
            cuda=cuda,
        )
        return Y

    def forward_infer(
        self,
        curr_emb: torch.Tensor,
        base_segment_indexes: torch.Tensor,
        max_num_speakers: int = 4,
        max_rp_threshold: float = 0.15,
        enhanced_count_thres: int = 40,
        sparse_search_volume: int = 10,
        fixed_thres: float = -1.0,
        frame_index: int = 0,
        cuda: bool = False,
    ) -> torch.Tensor:
        """
        Perform speaker clustering in online mode. Embedding vector set `emb` is expected to be containing
        history embeddings to count the number of speakers.

        Args:
            curr_emb (Tensor):
                Current embedding vector input.
            base_segment_indexes (Tensor):
                Tensor containing unique segment (embedding vector) index
            max_num_speakers (int):
                Maximum number of speakers to be detected during online diarization session
            max_rp_threshold (float):
                Limits the range of parameter search.
                Clustering performance can vary depending on this range.
                Default is 0.25.
            max_rp_threshold (float):
                Limits the range of parameter search.
                Clustering performance can vary depending on this range.
                Default is 0.15.
            frame_index (int):
                Unique index for each segment (also each embedding vector)
            cuda (bool):
                Boolean that determines whether cuda is used or not
            device (torch.device):
                `torch.device` variable

        Returns:
            Y (Tensor):
                Speaker labels for history embeddings and current embedding inputs
        """
        self.max_num_speakers = max_num_speakers
        self.max_rp_threshold = max_rp_threshold
        self.enhanced_count_thres = enhanced_count_thres
        self.sparse_search_volume = sparse_search_volume
        self.fixed_thres = fixed_thres
        # Merge the closest embeddings and reduce the size of the embedding count.

        if cuda and (curr_emb.device == torch.device("cpu") or base_segment_indexes.device == torch.device("cpu")):
            raise ValueError(f"CUDA is enabled but the input {curr_emb} or {base_segment_indexes} is not on the GPU.")

        merged_embs, add_new = self.get_reduced_mat(emb_in=curr_emb, base_segment_indexes=base_segment_indexes,)
        # Perform clustering on the embedding matrix containing history and current FIFO buffer merged_embeddings
        if merged_embs.shape[0] == 1:
            Y = torch.zeros((1,), dtype=torch.int32)
        else:
            mat = getCosAffinityMatrix(merged_embs)
            est_num_of_spk, affinity_mat = self.online_spk_num_estimation(mat, frame_index)
            spectral_model = SpectralClustering(n_clusters=est_num_of_spk, cuda=cuda, device=merged_embs.device)
            Y = spectral_model.forward(affinity_mat).to(merged_embs.device)
        # Match the permutation of the newly obtained speaker labels and the previous labels
        merged_clus_labels = self.match_labels(Y_merged=Y, add_new=add_new)
        return merged_clus_labels
