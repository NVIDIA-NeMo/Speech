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

from typing import Dict, List, Tuple

import torch
from torch.linalg import eigh, eigvalsh


def cos_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor, eps=torch.tensor(3.5e-4)) -> torch.Tensor:
    """
    Calculate cosine similarities of the given two set of tensors. The output is an N by N
    matrix where N is the number of feature vectors.

    Args:
        a (Tensor):
            Matrix containing speaker representation vectors. (N x embedding_dim)
        b (Tensor):
            Matrix containing speaker representation vectors. (N x embedding_dim)

    Returns:
        res (Tensor):
            N by N matrix containing the cosine similarities of the values.
    """
    # If number of embedding count is 1, it creates nan values
    if emb_a.shape[0] == 1 or emb_b.shape[0] == 1:
        raise ValueError(f"Number of feature vectors should be greater than 1 but got {emb_a.shape} and {emb_b.shape}")
    a_norm = emb_a / (torch.norm(emb_a, dim=1).unsqueeze(1) + eps)
    b_norm = emb_b / (torch.norm(emb_b, dim=1).unsqueeze(1) + eps)
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    res.fill_diagonal_(1)
    return res


def ScalerMinMax(X: torch.Tensor) -> torch.Tensor:
    """
    Min-max scale the input affinity matrix X, which will lead to a dynamic range of [0, 1].

    Args:
        X (Tensor):
            Matrix containing cosine similarity values among embedding vectors (N x N)

    Returns:
        v_norm (Tensor):
            Min-max normalized value of X.
    """
    v_min, v_max = X.min(), X.max()
    v_norm = (X - v_min) / (v_max - v_min)
    return v_norm


def getEuclideanDistance(
    specEmbA: torch.Tensor, specEmbB: torch.Tensor, device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Calculate Euclidean distances from the given feature tensors.

    Args:
        specEmbA (Tensor):
            Matrix containing spectral embedding vectors from eigenvalue decomposition (N x embedding_dim).
        specEmbB (Tensor):
            Matrix containing spectral embedding vectors from eigenvalue decomposition (N x embedding_dim).

    Returns:
        dis (Tensor):
            Euclidean distance values of the two sets of spectral embedding vectors.
    """
    specEmbA, specEmbB = specEmbA.to(device), specEmbB.to(device)
    A, B = specEmbA.unsqueeze(dim=1), specEmbB.unsqueeze(dim=0)
    dis = (A - B) ** 2.0
    dis = dis.sum(dim=-1).squeeze()
    return dis


def kmeans_plusplus_torch(
    X: torch.Tensor,
    n_clusters: int,
    random_state: int,
    n_local_trials: int = 30,
    device: torch.device = torch.device('cpu'),
):
    """
    Choose initial centroids for initializing k-means algorithm. The performance of
    k-means algorithm can vary significantly by the initial centroids. To alleviate
    this problem, k-means++ algorithm chooses initial centroids based on the probability
    proportional to the distance from the formally chosen centroids. The centroids
    selected by k-means++ algorithm improve the chance of getting more accurate and
    stable clustering results. The overall implementation of k-means++ algorithm is
    inspired by the numpy based k-means++ implementation in:
        https://github.com/scikit-learn/scikit-learn

    Originally, the implementation of the k-means++ algorithm in scikit-learn is based
    on the following research article:
        Arthur, David, and Sergei Vassilvitskii. k-means++: The advantages of careful
        seeding. Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete
        algorithms, Society for Industrial and Applied Mathematics (2007)

    Args:
        X (Tensor):
            Matrix containing cosine similarity values among embedding vectors (N x N)
        n_clusters (int):
            Maximum number of speakers for estimating number of speakers.
            Shows stable performance under 20.
        random_state (int):
            Seed variable for setting up a random state.
        n_local_trials (int):
            Number of trials for creating initial values of the center points.
        device (torch.device)
            Torch device variable.

    Returns:
        centers (Tensor):
            The coordinates for center points that are used for initializing k-means algorithm.
        indices (Tensor):
            The indices of the best candidate center points.
    """
    torch.manual_seed(random_state)
    X = X.to(device)
    n_samples, n_features = X.shape

    centers = torch.zeros(n_clusters, n_features, dtype=X.dtype)
    center_id = torch.randint(0, n_samples, (1,)).long()
    indices = torch.full([n_clusters,], -1, dtype=torch.int)

    centers[0] = X[center_id].squeeze(0)
    indices[0] = center_id.squeeze(0)

    centers = centers.to(device)
    closest_dist_diff = centers[0, None].repeat(1, X.shape[0]).view(X.shape[0], -1) - X
    closest_dist_sq = closest_dist_diff.pow(2).sum(dim=1).unsqueeze(dim=0)
    current_pot = closest_dist_sq.sum()

    for c in range(1, n_clusters):
        rand_vals = torch.rand(n_local_trials) * current_pot.item()

        if len(closest_dist_sq.shape) > 1:
            torch_cumsum = torch.cumsum(closest_dist_sq, dim=1)[0]
        else:
            torch_cumsum = torch.cumsum(closest_dist_sq, dim=0)

        candidate_ids = torch.searchsorted(torch_cumsum, rand_vals.to(device))

        N_ci = candidate_ids.shape[0]
        distance_diff = X[candidate_ids].repeat(1, X.shape[0]).view(X.shape[0] * N_ci, -1) - X.repeat(N_ci, 1)
        distance = distance_diff.pow(2).sum(dim=1).view(N_ci, -1)
        distance_to_candidates = torch.minimum(closest_dist_sq, distance)
        candidates_pot = distance_to_candidates.sum(dim=1)

        best_candidate = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = X[best_candidate]
        indices[c] = best_candidate
    return centers, indices


def kmeans_torch(
    X: torch.Tensor,
    num_clusters: int,
    threshold: float = 1e-4,
    iter_limit: int = 15,
    random_state: int = 0,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Run k-means algorithm on the given set of spectral embeddings in X. The threshold
    and iter_limit variables are set to show the best performance on speaker diarization
    tasks. The overall implementation of k-means algorithm is inspired by the k-means
    algorithm implemented in https://github.com/scikit-learn/scikit-learn.

    References:
        Arthur, David, and Sergei Vassilvitskii. k-means++: The advantages of careful
        seeding. Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete
        algorithms, Society for Industrial and Applied Mathematics (2007).

    Args:
        X (Tensor):
            Cosine similarity matrix calculated from speaker embeddings
        num_clusters (int):
            The estimated number of speakers.
        threshold (float):
            This threshold limits the change of center values. If the square of
            the center shift values are bigger than this threshold, the iteration stops.
        iter_limit (int):
            The maximum number of iterations that is allowed by the k-means algorithm.
        device (torch.device):
            Torch device variable

    Returns:
        selected_cluster_indices (Tensor):
            The assigned cluster labels from the k-means clustering.
    """
    # Convert tensor type to float
    X = X.float().to(device)
    input_size = X.shape[0]

    # Initialize the cluster centers with kmeans_plusplus algorithm.
    plusplus_init_states = kmeans_plusplus_torch(X, n_clusters=num_clusters, random_state=random_state, device=device)
    centers = plusplus_init_states[0]

    selected_cluster_indices = torch.zeros(input_size).long()

    for iter_count in range(iter_limit):
        euc_dist = getEuclideanDistance(X, centers, device=device)

        if len(euc_dist.shape) <= 1:
            break
        else:
            selected_cluster_indices = torch.argmin(euc_dist, dim=1)

        center_inits = centers.clone()

        for index in range(num_clusters):
            selected_cluster = torch.nonzero(selected_cluster_indices == index).squeeze().to(device)
            chosen_indices = torch.index_select(X, 0, selected_cluster)

            if chosen_indices.shape[0] == 0:
                chosen_indices = X[torch.randint(len(X), (1,))]

            centers[index] = chosen_indices.mean(dim=0)

        # Calculate the delta from center_inits to centers
        center_delta_pow = torch.pow((centers - center_inits), 2)
        center_shift_pow = torch.pow(torch.sum(torch.sqrt(torch.sum(center_delta_pow, dim=1))), 2)

        # If the cluster centers are not changing significantly, stop the loop.
        if center_shift_pow < threshold:
            break

    return selected_cluster_indices


def getTheLargestComponent(affinity_mat: torch.Tensor, seg_index: int, device: torch.device) -> torch.Tensor:
    """
    Find the largest affinity_mat connected components for each given node.
    This is for checking whether the affinity_mat is fully connected.

    Args:
        affinity_mat (Tensor):
            A square matrix (tensor) containing normalized cosine distance values
        seg_index (int):
            The segment index that is targeted to be explored.

    Returns:
        connected_nodes (Tensor):
            A tensor containing booleans that indicate whether the node is connected.
    """
    num_of_segments = affinity_mat.shape[0]

    connected_nodes = torch.zeros(num_of_segments, dtype=torch.bool).to(device)
    nodes_to_explore = torch.zeros(num_of_segments, dtype=torch.bool).to(device)

    nodes_to_explore[seg_index] = True
    nodes_to_explore = nodes_to_explore.to(device)
    for k in range(num_of_segments):
        last_num_component = connected_nodes.sum()
        torch.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break

        indices = (nodes_to_explore == torch.tensor(True)).nonzero().t().squeeze()
        if len(indices.size()) == 0:
            indices = indices.unsqueeze(0)
        for i in indices:
            neighbors = affinity_mat[i].to(device)
            torch.logical_or(nodes_to_explore, neighbors.squeeze(0), out=nodes_to_explore)
    return connected_nodes


def isGraphFullyConnected(affinity_mat: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Check whether the given affinity matrix is a fully connected graph.
    """
    return getTheLargestComponent(affinity_mat, 0, device).sum() == affinity_mat.shape[0]


def getKneighborsConnections(affinity_mat: torch.Tensor, p_value: int, mask_method: str = 'binary') -> torch.Tensor:
    """
    Binarize top-p values for each row from the given affinity matrix.

    Args:
        affinity_mat (Tensor):
            A square matrix (tensor) containing normalized cosine similarity values
        p_value (int):
            The number of top values that are selected from each row.
        mask_method (str):
            The method that is used to manipulate the affinity matrix. The default method is 'binary'.

    Returns:
        binarized_affinity_mat (Tensor):
            A binarized affinity matrix based on the given mask method.
    """
    dim = affinity_mat.shape
    binarized_affinity_mat = torch.zeros_like(affinity_mat).half()
    sorted_matrix = torch.argsort(affinity_mat, dim=1, descending=True)[:, :p_value]
    binarized_affinity_mat[sorted_matrix.T, torch.arange(affinity_mat.shape[0])] = (
        torch.ones(1).to(affinity_mat.device).half()
    )
    indices_row = sorted_matrix[:, :p_value].flatten()
    indices_col = torch.arange(dim[1]).repeat(p_value, 1).T.flatten()
    if mask_method == 'binary' or mask_method is None:
        binarized_affinity_mat[indices_row, indices_col] = (
            torch.ones(indices_row.shape[0]).to(affinity_mat.device).half()
        )
    elif mask_method == 'drop':
        binarized_affinity_mat[indices_row, indices_col] = affinity_mat[indices_row, indices_col].half()
    elif mask_method == 'sigmoid':
        binarized_affinity_mat[indices_row, indices_col] = torch.sigmoid(affinity_mat[indices_row, indices_col]).half()
    else:
        raise ValueError(f'Unknown mask method: {mask_method}')
    return binarized_affinity_mat


def getAffinityGraphMat(affinity_mat_raw: torch.Tensor, p_value: int) -> torch.Tensor:
    """
    Calculate a binarized graph matrix and
    symmetrize the binarized graph matrix.
    """
    X = affinity_mat_raw if p_value <= 0 else getKneighborsConnections(affinity_mat_raw, p_value)
    symm_affinity_mat = 0.5 * (X + X.T)
    return symm_affinity_mat


def getMinimumConnection(
    mat: torch.Tensor, max_N: torch.Tensor, n_list: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate connections until fully connect all the nodes in the graph.
    If the graph is not fully connected, it might generate inaccurate results.
    """
    p_value = torch.tensor(1)
    affinity_mat = getAffinityGraphMat(mat, p_value)
    for i, p_value in enumerate(n_list):
        fully_connected = isGraphFullyConnected(affinity_mat, device)
        affinity_mat = getAffinityGraphMat(mat, p_value)
        if fully_connected or p_value > max_N:
            break

    return affinity_mat, p_value


def getRepeatedList(mapping_argmat: torch.Tensor, score_mat_size: torch.Tensor) -> torch.Tensor:
    """
    Count the numbers in the mapping dictionary and create lists that contain
    repeated indices that will be used for creating a repeated affinity matrix.
    This repeated matrix is then used for fusing multiple affinity values.
    """
    repeat_list = torch.zeros(score_mat_size, dtype=torch.int32).to(mapping_argmat.device)
    idxs, counts = torch.unique(mapping_argmat, return_counts=True)
    repeat_list[idxs] = counts.int().to(mapping_argmat.device)
    return repeat_list


def get_argmin_mat(timestamps_in_scales: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Calculate the mapping between the base scale and other scales. A segment from a longer scale is
    repeatedly mapped to a segment from a shorter scale or the base scale.

    Args:
        timestamps_in_scales (list):
            List containing timestamp tensors for each scale.
            Each tensor has dimensions of (Number of base segments) x 2.

    Returns:
        session_scale_mapping_list (list):
            List containing argmin arrays indexed by scale index.
    """
    scale_list = list(range(len(timestamps_in_scales)))
    segment_anchor_list = []
    for scale_idx in scale_list:
        time_stamps_float = timestamps_in_scales[scale_idx]
        segment_anchor_list.append(torch.mean(time_stamps_float, dim=1))

    base_scale_idx = max(scale_list)
    base_scale_anchor = segment_anchor_list[base_scale_idx]
    session_scale_mapping_list = []
    for scale_idx in scale_list:
        curr_scale_anchor = segment_anchor_list[scale_idx]
        curr_mat = torch.tile(curr_scale_anchor, (base_scale_anchor.shape[0], 1))
        base_mat = torch.tile(base_scale_anchor, (curr_scale_anchor.shape[0], 1)).t()
        argmin_mat = torch.argmin(torch.abs(curr_mat - base_mat), dim=1)
        session_scale_mapping_list.append(argmin_mat)
    return session_scale_mapping_list


def getCosAffinityMatrix(emb: torch.Tensor) -> torch.Tensor:
    """
    Calculate cosine similarity values among speaker embeddings then min-max normalize
    the affinity matrix.

    Args:
        emb (Tensor):
            Matrix containing embedding vectors. emb variable should be float(FP32) type to make the data-type
            compatible with torch.mm operation for both CPU and GPU(CUDA).
            dimension: (Number of embedding vectors) x (embedding dimension)

    Returns:
        sim_d (Tensor):
            Matrix containing cosine similarity values among the given embedding vectors.
            dimension: (Number of embedding vectors) x (Number of embedding vectors)
    """
    if emb.shape[0] == 1:
        sim_d = torch.tensor([[1]]).to(emb.device)
    else:
        emb = emb.float()
        sim_d = cos_similarity(emb, emb)
        sim_d = ScalerMinMax(sim_d)
    return sim_d


def get_scale_interpolated_embs(
    multiscale_weights: torch.Tensor,
    embeddings_in_scales: List[torch.Tensor],
    timestamps_in_scales: List[torch.Tensor],
    device: torch.device = torch.device('cpu'),
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Generate a scale-interpolated single embedding vector by calculating the weighted sum
    of the multiple embedding vectors from different scales. The output is a set of embedding
    vectors corresponding to the base-scale segments.

    Args:
        multiscale_weights (Tensor):
            Tensor containing Multiscale weights
            Dimensions: (Number of scales) x 1
        embeddings_in_scales (list):
            List containing split embedding tensors by each scale
        timestamps_in_scales (list):
            List containing split timestamps tensors by each scale
        device (torch.device):
            Torch device variable

    Returns:
        context_emb (torch.tensor):
            A set of scale-interpolated embedding vectors.
            Dimensions: (Number of base-scale segments) x (Dimensions of embedding vector)
        session_scale_mapping_list (list):
            List containing argmin arrays indexed by scale index.
    """
    rep_mat_list = []
    multiscale_weights = multiscale_weights.to(device)
    session_scale_mapping_list = get_argmin_mat(timestamps_in_scales)
    scale_list = list(range(len(timestamps_in_scales)))
    for scale_idx in scale_list:
        mapping_argmat = session_scale_mapping_list[scale_idx]
        emb_t = embeddings_in_scales[scale_idx].to(device)
        mapping_argmat = mapping_argmat.to(device)
        repeat_list = getRepeatedList(mapping_argmat, torch.tensor(emb_t.shape[0])).to(device)
        rep_emb_t = torch.repeat_interleave(emb_t, repeats=repeat_list, dim=0)
        rep_mat_list.append(rep_emb_t)
    stacked_scale_embs = torch.stack(rep_mat_list)
    context_emb = torch.matmul(stacked_scale_embs.permute(2, 1, 0), multiscale_weights.t()).squeeze().t()
    if len(context_emb.shape) < 2:
        context_emb = context_emb.unsqueeze(0)
    context_emb = context_emb.to(device)
    return context_emb, session_scale_mapping_list


def getMultiScaleCosAffinityMatrix(
    multiscale_weights: torch.Tensor,
    embeddings_in_scales: List[torch.Tensor],
    timestamps_in_scales: List[torch.Tensor],
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Calculate cosine similarity values among speaker embeddings for each scale then
    apply multiscale weights to calculate the fused similarity matrix.
    NOTE: Due to CUDA memory limit, the embedding vectors in embeddings_in_scales are stored in `cpu` device.

    Args:
        multiscale_weights (Tensor):
            Tensor containing multiscale weights
            Dimensions: (Number of scales) x 1
        embeddings_in_scales (list):
            List containing split embedding tensors by each scale
        timestamps_in_scales (list):
            List containing split timestamps tensors by each scale
        device (torch.device):
            Torch device variable

    Returns:
        fused_sim_d (Tensor):
            An affinity matrix that is obtained by calculating the weighted sum of 
            the multiple affinity matrices from the different scales.
    """
    multiscale_weights = torch.squeeze(multiscale_weights, dim=0).to(device)
    session_scale_mapping_list = get_argmin_mat(timestamps_in_scales)
    scale_list = list(range(len(timestamps_in_scales)))
    fused_sim_d = torch.zeros(len(timestamps_in_scales[-1]), len(timestamps_in_scales[-1])).to(device)
    for scale_idx in scale_list:
        mapping_argmat = session_scale_mapping_list[scale_idx]
        emb_t = embeddings_in_scales[scale_idx].half().to(device)
        score_mat_torch = getCosAffinityMatrix(emb_t)
        repeat_list = getRepeatedList(mapping_argmat, torch.tensor(score_mat_torch.shape[0])).to(device)
        repeated_tensor_0 = torch.repeat_interleave(score_mat_torch, repeats=repeat_list, dim=0).to(device)
        repeated_tensor_1 = torch.repeat_interleave(repeated_tensor_0, repeats=repeat_list, dim=1).to(device)
        fused_sim_d += multiscale_weights[scale_idx] * repeated_tensor_1
    return fused_sim_d


def getLaplacian(X: torch.Tensor) -> torch.Tensor:
    """
    Calculate a laplacian matrix from an affinity matrix X.
    """
    X.fill_diagonal_(0)
    D = torch.sum(torch.abs(X), dim=1)
    D = torch.diag_embed(D)
    L = D - X
    return L


def eigDecompose(laplacian: torch.Tensor, cuda: bool, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate eigenvalues and eigenvectors from the Laplacian matrix.
    """
    if cuda:
        if device is None:
            device = torch.cuda.current_device()
        laplacian = laplacian.float().to(device)
    else:
        laplacian = laplacian.float().to(torch.device('cpu'))
    lambdas, diffusion_map = eigh(laplacian)
    return lambdas, diffusion_map


def eigValueSh(laplacian: torch.Tensor, cuda: bool, device: torch.device) -> torch.Tensor:
    """
    Calculate only eigenvalues from the Laplacian matrix.
    """
    if cuda:
        if device is None:
            device = torch.cuda.current_device()
        laplacian = laplacian.float().to(device)
    else:
        laplacian = laplacian.float().to(torch.device('cpu'))
    lambdas = eigvalsh(laplacian)
    return lambdas


def getLamdaGaplist(lambdas: torch.Tensor) -> torch.Tensor:
    """
    Calculate the gaps between lambda values.
    """
    if torch.is_complex(lambdas):
        lambdas = torch.real(lambdas)
    return lambdas[1:] - lambdas[:-1]


def addAnchorEmb(emb: torch.Tensor, anchor_sample_n: int, anchor_spk_n: int, sigma: float) -> torch.Tensor:
    """
    Add randomly generated synthetic embeddings to make eigenanalysis more stable.
    We refer to these embeddings as anchor embeddings.

    emb (Tensor):
        The input embedding from the embedding extractor.
    anchor_sample_n (int):
        Number of embedding samples per speaker.
        anchor_sample_n = 10 is recommended.
    anchor_spk_n (int):
        Number of speakers for synthetic embedding.
        anchor_spk_n = 3 is recommended.
    sigma (int):
        The amplitude of synthetic noise for each embedding vector.
        If the sigma value is too small, under-counting could happen.
        If the sigma value is too large, over-counting could happen.
        sigma = 50 is recommended.
    """
    emb_dim = emb.shape[1]
    std_org = torch.std(emb, dim=0)
    sigma = torch.tensor(sigma).to(emb.device)
    new_emb_list = []
    for _ in range(anchor_spk_n):
        emb_m = torch.tile(torch.randn(1, emb_dim), (anchor_sample_n, 1)).to(emb.device)
        emb_noise = torch.randn(anchor_sample_n, emb_dim).T.to(emb.device)
        emb_noise = torch.matmul(
            torch.diag(std_org), emb_noise / torch.max(torch.abs(emb_noise), dim=0)[0].unsqueeze(0)
        ).T
        emb_gen = emb_m + sigma * emb_noise
        new_emb_list.append(emb_gen)

    new_emb_list.append(emb)
    new_emb_np = torch.vstack(new_emb_list)
    return new_emb_np


def getEnhancedSpeakerCount(
    emb: torch.Tensor,
    random_test_count: int = 5,
    anchor_spk_n: int = 3,
    anchor_sample_n: int = 10,
    sigma: float = 50,
    cuda: bool = False,
) -> torch.Tensor:
    """
    Calculate the number of speakers using NME analysis with anchor embeddings. Add dummy speaker
    embedding vectors and run speaker counting multiple times to enhance the speaker counting accuracy
    for the short audio samples.

    Args:
        emb (Tensor):
            The input embedding from the embedding extractor.
        cuda (bool):
            Use cuda for the operations if cuda==True.
        random_test_count (int):
            Number of trials of the enhanced counting with randomness.
            The higher the count, the more accurate the enhanced counting is.
        anchor_spk_n (int):
            Number of speakers for synthetic embedding.
            anchor_spk_n = 3 is recommended.
        anchor_sample_n (int):
            Number of embedding samples per speaker.
            anchor_sample_n = 10 is recommended.
        sigma (float):
            The amplitude of synthetic noise for each embedding vector.
            If the sigma value is too small, under-counting could happen.
            If the sigma value is too large, over-counting could happen.
            sigma = 50 is recommended.

    Returns:
        comp_est_num_of_spk (Tensor):
            The estimated number of speakers. `anchor_spk_n` is subtracted from the estimated
            number of speakers to factor out the dummy speaker embedding vectors.
    """
    est_num_of_spk_list: List[int] = []
    for seed in range(random_test_count):
        torch.manual_seed(seed)
        emb_aug = addAnchorEmb(emb, anchor_sample_n, anchor_spk_n, sigma)
        mat = getCosAffinityMatrix(emb_aug)
        nmesc = NMESC(
            mat,
            max_num_speakers=emb.shape[0],
            max_rp_threshold=0.15,
            sparse_search=True,
            sparse_search_volume=10,
            fixed_thres=-1.0,
            nme_mat_size=300,
            cuda=cuda,
        )
        est_num_of_spk, _ = nmesc.forward()
        est_num_of_spk_list.append(est_num_of_spk.item())
    comp_est_num_of_spk = torch.tensor(max(torch.mode(torch.tensor(est_num_of_spk_list))[0].item() - anchor_spk_n, 1))
    return comp_est_num_of_spk


def split_input_data(
    embeddings_in_scales: torch.Tensor,
    timestamps_in_scales: torch.Tensor,
    multiscale_segment_counts: torch.LongTensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Split multiscale embeddings and multiscale timestamps and put split scale-wise data into python lists.
    This formatting function is needed to make the input type as `torch.Tensor`.

    Args:
        embeddings_in_scales (Tensor):
            Concatenated Torch tensor containing embeddings in multiple scales
        timestamps_in_scales (Tensor):
            Concatenated Torch tensor containing timestamps in multiple scales
        multiscale_segment_counts (LongTensor):
            Concatenated Torch LongTensor containing number of segments per each scale

    Returns:
        embeddings_in_scales (list):
            List containing split embedding tensors by each scale
        timestamps_in_scales (list):
            List containing split timestamps tensors by each scale
    """
    split_index: List[int] = multiscale_segment_counts.tolist()
    embeddings_in_scales = torch.split(embeddings_in_scales, split_index, dim=0)
    timestamps_in_scales = torch.split(timestamps_in_scales, split_index, dim=0)
    embeddings_in_scales, timestamps_in_scales = list(embeddings_in_scales), list(timestamps_in_scales)
    return embeddings_in_scales, timestamps_in_scales


def estimateNumofSpeakers(
    affinity_mat: torch.Tensor, max_num_speakers: int, cuda: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimate the number of speakers using eigendecomposition on the Laplacian Matrix.

    Args:
        affinity_mat (Tensor):
            N by N affinity matrix
        max_num_speakers (int):
            Maximum number of clusters to consider for each session
        cuda (bool):
            If cuda available eigendecomposition is computed on GPUs.

    Returns:
        num_of_spk (Tensor):
            The estimated number of speakers
        lambdas (Tensor):
            The lambda values from eigendecomposition
        lambda_gap (Tensor):
            The gap between the lambda values from eigendecomposition
    """
    laplacian = getLaplacian(affinity_mat)
    lambdas = eigValueSh(laplacian, cuda=cuda, device=affinity_mat.device)
    lambdas = torch.sort(lambdas)[0]
    lambda_gap = getLamdaGaplist(lambdas)
    num_of_spk = torch.argmax(lambda_gap[: min(max_num_speakers, lambda_gap.shape[0])]) + 1
    return num_of_spk, lambdas, lambda_gap


class SpectralClustering:
    """
    Perform spectral clustering by calculating spectral embeddings then run k-means clustering
    algorithm on the spectral embeddings.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        random_state: int = 0,
        n_random_trials: int = 1,
        cuda: bool = False,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize the variables needed for spectral clustering and k-means++.

        Args:
            n_clusters (int):
                Number of the estimated (or oracle) number of speakers
            random_state (int):
                Random seed that determines a random state of k-means initialization.
            n_random_trials (int):
                Number of trials with different random seeds for k-means initialization.
                k-means++ algorithm is executed for multiple times then the final result
                is obtained by taking a majority vote.
            cuda (bool):
                if cuda=True, spectral clustering is done on GPU.
            device (torch.device):
                Torch device variable
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_random_trials = max(n_random_trials, 1)
        self.cuda = cuda
        self.device = device

    def forward(self, X) -> torch.Tensor:
        """
        Call self.clusterSpectralEmbeddings() function to predict cluster labels.

        Args:
            X (Tensor):
                Affinity matrix input

        Returns:
            labels (Tensor):
                clustering label output
        """
        if X.shape[0] != X.shape[1]:
            raise ValueError("The affinity matrix is not a square matrix.")
        labels = self.clusterSpectralEmbeddings(X, cuda=self.cuda, device=self.device)
        return labels

    def clusterSpectralEmbeddings(
        self, affinity: torch.Tensor, cuda: bool = False, device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        Perform k-means clustering on spectral embeddings. To alleviate the effect of randomness,
        k-means clustering is performed for (self.n_random_trials) times then the final labels are obtained
        by taking a majority vote. If speed is the major concern, self.n_random_trials should be set to 1.
        n_random_trials=30 is recommended to see an improved result.

        Args:
            affinity (Tensor):
                Affinity matrix input
            cuda (torch.bool):
                Use cuda for spectral clustering if cuda=True
            device (torch.device):
                Torch device variable

        Returns:
            labels (Tensor):
                clustering label output

        """
        spectral_emb = self.getSpectralEmbeddings(affinity, n_spks=self.n_clusters, cuda=cuda)
        labels_set = []

        for random_state_seed in range(self.random_state, self.random_state + self.n_random_trials):
            _labels = kmeans_torch(
                X=spectral_emb, num_clusters=self.n_clusters, random_state=random_state_seed, device=device
            )
            labels_set.append(_labels)
        stacked_labels = torch.stack(labels_set)
        label_index = torch.mode(torch.mode(stacked_labels, 0)[1])[0]
        labels = stacked_labels[label_index]
        return labels

    def getSpectralEmbeddings(self, affinity_mat: torch.Tensor, n_spks: int = 8, cuda: bool = False) -> torch.Tensor:
        """
        Calculate eigenvalues and eigenvectors to extract spectral embeddings.

        Args:
            affinity (Tensor):
                Affinity matrix input
            cuda (torch.bool):
                Use cuda for spectral clustering if cuda=True
            device (torch.device):
                Torch device variable

        Returns:
            labels (Tensor):
                clustering label output
        """
        laplacian = getLaplacian(affinity_mat)
        _, diffusion_map_ = eigDecompose(laplacian, cuda=cuda, device=affinity_mat.device)
        diffusion_map = diffusion_map_[:, :n_spks]
        inv_idx = torch.arange(diffusion_map.size(1) - 1, -1, -1).long()
        embedding = diffusion_map.T[inv_idx, :]
        return embedding[:n_spks].T


class NMESC:
    """
    Normalized Maximum Eigengap based Spectral Clustering (NME-SC)
    uses Eigengap analysis to get an estimated p-value for
    affinity binarization and an estimated number of speakers.

    p_value (also referred to as p_neighbors) is for taking
    top p number of affinity values and convert those to 1 while
    convert the rest of values to 0.

    p_value can be also tuned on a development set without performing
    NME-analysis. Fixing p_value brings about significantly faster clustering
    speed, but the performance is limited to the development set.

    References:
        Tae Jin Park et al., Auto-Tuning Spectral Clustering for Speaker Diarization
        Using Normalized Maximum Eigengap, IEEE Signal Processing Letters 27 (2019),
        https://arxiv.org/abs/2003.02405

    Args:
        Please refer to def __init__().

    Methods:
        NMEanalysis():
            Performs NME-analysis to estimate p_value and the number of speakers
        subsampleAffinityMat(nme_mat_size):
            Subsamples the number of speakers to reduce the computational load
        getPvalueList():
            Generates a list containing p-values that need to be examined.
        getEigRatio(p_neighbors):
            Calculates g_p, which is a ratio between p_neighbors and the maximum eigengap
        getLamdaGaplist(lambdas):
            Calculates lambda gap values from an array contains lambda values
        estimateNumofSpeakers(affinity_mat):
            Estimates the number of speakers using lambda gap list
    """

    def __init__(
        self,
        mat: torch.Tensor,
        max_num_speakers: int = 10,
        max_rp_threshold: float = 0.15,
        sparse_search: bool = True,
        sparse_search_volume: int = 30,
        nme_mat_size: int = 512,
        use_subsampling_for_nme: bool = True,
        fixed_thres: float = -1.0,
        maj_vote_spk_count: bool = False,
        parallelism: bool = True,
        cuda: bool = False,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Args:
            mat (Tensor):
                Cosine similarity matrix calculated from the provided speaker embeddings.
            max_num_speakers (int):
                Maximum number of speakers for estimating number of speakers.
                Shows stable performance under 20.
            max_rp_threshold (float):
                Limits the range of parameter search.
                Clustering performance can vary depending on this range.
                Default is 0.25.
            sparse_search (bool):
                To increase the speed of parameter estimation, sparse_search=True
                limits the number of p_values we search.
            sparse_search_volume (int):
                Number of p_values we search during NME analysis.
                Default is 30. The lower the value, the faster NME-analysis becomes.
                However, a value lower than 20 might cause a poor parameter estimation.
            nme_mat_size (int):
                Targeted size of matrix for NME analysis.
            use_subsampling_for_nme (bool):
                Use subsampling to reduce the calculational complexity.
                Default is True.
            fixed_thres (float or None):
                A fixed threshold which can be used instead of estimating the
                threshold with NME analysis. If fixed_thres is float,
                it skips the NME analysis part.
            maj_vote_spk_count (bool):
                If True, take a majority vote on all p-values in the given range to estimate the number of speakers.
                The majority voting may contribute to surpress overcounting of the speakers and improve speaker
                counting accuracy.
            parallelism (bool):
                If True, turn on parallelism based on torch.jit.script library.
            cuda (bool):
                Use cuda for Eigen decomposition if cuda=True.
            device (torch.device):
                Torch device variable

        """
        self.max_num_speakers: int = max_num_speakers
        self.max_rp_threshold: float = max_rp_threshold
        self.use_subsampling_for_nme: bool = use_subsampling_for_nme
        self.nme_mat_size: int = nme_mat_size
        self.sparse_search: bool = sparse_search
        self.sparse_search_volume: int = sparse_search_volume
        self.min_p_value = torch.tensor(2)
        self.fixed_thres: float = fixed_thres
        self.eps = 1e-10
        self.max_N = torch.tensor(0)
        self.mat: torch.Tensor = mat
        self.p_value_list: torch.Tensor = self.min_p_value.unsqueeze(0)
        self.cuda: bool = cuda
        self.device: torch.device = device
        self.maj_vote_spk_count: bool = maj_vote_spk_count
        self.parallelism: bool = parallelism

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Subsample the input matrix to reduce the computational load.

        Returns:
            est_num_of_spk (Tensor):
                Estimated number of speakers from NMESC approach
            p_hat_value (Tensor):
                Estimated p-value (determines how many neighboring values to be selected)
        """
        if self.use_subsampling_for_nme:
            subsample_ratio = self.subsampleAffinityMat(self.nme_mat_size)
        else:
            subsample_ratio = torch.tensor(1)

        # Scans p_values and find a p_value that generates the smallest g_p value.
        results: List[torch.Tensor] = []
        est_spk_n_dict: Dict[int, torch.Tensor] = {}
        self.p_value_list = self.getPvalueList()
        p_volume = self.p_value_list.shape[0]
        eig_ratio_list = torch.zeros(p_volume,)
        est_num_of_spk_list = torch.zeros(p_volume,)

        if self.parallelism:
            futures: List[torch.jit.Future[torch.Tensor]] = []
            for p_idx, p_value in enumerate(self.p_value_list):
                futures.append(torch.jit.fork(self.getEigRatio, p_value))
            for future in futures:
                results.append(torch.jit.wait(future))

        else:
            for p_idx, p_value in enumerate(self.p_value_list):
                results.append(self.getEigRatio(p_value))

        # Retrieve the eigen analysis results
        for p_idx, p_value in enumerate(self.p_value_list):
            output = results[p_idx]
            g_p, est_num_of_spk = output[0], output[1].int()
            eig_ratio_list[p_idx] = g_p
            est_spk_n_dict[p_value.item()] = est_num_of_spk
            est_num_of_spk_list[p_idx] = est_num_of_spk

        index_nn = torch.argmin(eig_ratio_list)
        rp_p_value = self.p_value_list[index_nn]
        affinity_mat = getAffinityGraphMat(self.mat, rp_p_value)

        # Checks whether the affinity graph is fully connected.
        # If not, it adds a minimum number of connections to make it fully connected.
        if not isGraphFullyConnected(affinity_mat, device=self.device):
            affinity_mat, rp_p_value = getMinimumConnection(
                self.mat, self.max_N, self.p_value_list, device=self.device
            )

        p_hat_value = (subsample_ratio * rp_p_value).type(torch.int)
        if self.maj_vote_spk_count:
            est_num_of_spk = torch.mode(torch.tensor(est_num_of_spk_list))[0]
        else:
            est_num_of_spk = est_spk_n_dict[rp_p_value.item()]
        return est_num_of_spk, p_hat_value

    def subsampleAffinityMat(self, nme_mat_size: int) -> torch.Tensor:
        """
        Perform subsampling of affinity matrix.
        This subsampling is for calculational complexity, not for performance.
        The smaller nme_mat_size is,
            - the bigger the chance of missing a speaker.
            - the faster p-value estimation speed (based on eigen decomposition).

        The recommended nme_mat_size is 250~750.
        However, if there are speakers who speak for very short period of time in the recording,
        this subsampling might make the system miss underrepresented speakers.
        Use this variable with caution.

        Args:
            nme_mat_size (int):
                The targeted matrix size

        Returns:
            subsample_ratio (float):
                The ratio between nme_mat_size and the original matrix size
        """
        subsample_ratio = torch.max(torch.tensor(1), torch.tensor(self.mat.shape[0] / nme_mat_size)).type(torch.int)
        self.mat = self.mat[:: subsample_ratio.item(), :: subsample_ratio.item()]
        return subsample_ratio

    def getEigRatio(self, p_neighbors: int) -> torch.Tensor:
        """
        For a given p_neighbors value, calculate g_p, which is a ratio between p_neighbors and the
        maximum eigengap values.
        References:
            Tae Jin Park et al., Auto-Tuning Spectral Clustering for Speaker Diarization Using
            Normalized Maximum Eigengap, IEEE Signal Processing Letters 27 (2019),
            https://arxiv.org/abs/2003.02405

        Args:
            p_neighbors (int):
                Determines how many binary graph connections we want to keep for each row.

        Returns:
            est_num_of_spk (int):
                Estimated number of speakers
            g_p (float):
                The ratio between p_neighbors value and the maximum eigen gap value.
        """
        affinity_mat = getAffinityGraphMat(self.mat, p_neighbors)
        est_num_of_spk, lambdas, lambda_gap_list = estimateNumofSpeakers(
            affinity_mat, self.max_num_speakers, self.cuda
        )
        arg_sorted_idx = torch.argsort(lambda_gap_list[: self.max_num_speakers], descending=True)
        max_key = arg_sorted_idx[0]
        max_eig_gap = lambda_gap_list[max_key] / (torch.max(lambdas).item() + self.eps)
        g_p = (p_neighbors / self.mat.shape[0]) / (max_eig_gap + self.eps)
        return torch.stack([g_p, est_num_of_spk])

    def getPvalueList(self) -> torch.Tensor:
        """
        Generates a p-value (p_neighbour) list for searching. p_value_list must include 2 (min_p_value)
        since at least one neighboring segment should be selected other than itself.

        If fixed_thres value is specified, then only one p-value is specified.
        If fixed_thres is not provided, multiple p-values are searched.
            If sparse_search is True:
                - Limit the number of p-values to be searched to sparse_search_volume.
                - N should be at least 2 to include a number greater than 1.
            If sparse_search is False:
                - Scan all the p_values from 1 to max_N
                - If sparse_search is False, NMESC analysis could take more time compared to sparse_search = True.

        Returns:
            p_value_list (Tensor):
                Tensor containing the p_values to be searched.
        """
        if self.fixed_thres is not None and self.fixed_thres > 0.0:
            self.max_N = torch.max(
                torch.floor(torch.tensor(self.mat.shape[0] * self.fixed_thres)).type(torch.int), self.min_p_value
            )
            p_value_list = self.max_N.unsqueeze(0).int()
        else:
            self.max_N = torch.max(
                torch.floor(torch.tensor(self.mat.shape[0] * self.max_rp_threshold)).type(torch.int), self.min_p_value
            )
            if self.sparse_search:
                search_volume = torch.min(self.max_N, torch.tensor(self.sparse_search_volume).type(torch.int))
                # search at least two values
                N = torch.max(search_volume, torch.tensor(2))
                # avoid repeating values by limiting the step size
                steps = min(self.max_N, N)
                p_value_list = torch.linspace(start=1, end=self.max_N, steps=steps).type(torch.int)
            else:
                p_value_list = torch.arange(1, self.max_N + 1)
        if p_value_list.shape[0] == 0:
            raise ValueError("p_value_list should not be empty.")
        return p_value_list


class SpeakerClustering(torch.nn.Module):
    def __init__(
        self,
        min_samples_for_nmesc: int = 6,
        nme_mat_size: int = 512,
        sparse_search: bool = True,
        maj_vote_spk_count: bool = False,
        parallelism: bool = False,
        cuda: bool = False,
    ):
        """
        Clustering method for speaker diarization based on cosine similarity.
        NME-SC part is converted to torch.tensor based operations in NeMo 1.9.

        Args:
            min_samples_for_nmesc (int):
                The minimum number of samples required for NME clustering. This avoids
                zero p_neighbour_lists. If the input has fewer segments than min_samples,
                it is directed to the enhanced speaker counting mode.
            sparse_search (bool):
                Toggle sparse search mode. If True, limit the size of p_value_list to sparse_search_volume.
            maj_vote_spk_count (bool):
                If True, take a majority vote on all p-values in the given range to estimate the number of speakers.
                The majority voting may contribute to surpress overcounting of the speakers and improve speaker
                counting accuracy.
            parallelism (bool):
                Use dynamic parallelism feature in torch.jit compiler to accelerate the p-value search.
            cuda (bool):
                Boolean variable for toggling cuda availability.
        """
        super().__init__()
        self.min_samples_for_nmesc: int = min_samples_for_nmesc
        self.nme_mat_size: int = nme_mat_size
        self.sparse_search: bool = sparse_search
        self.parallelism: bool = parallelism
        self.cuda: bool = cuda
        self.maj_vote_spk_count: bool = maj_vote_spk_count
        self.embeddings_in_scales: List[torch.Tensor] = [torch.Tensor(0)]
        self.timestamps_in_scales: List[torch.Tensor] = [torch.Tensor(0)]
        self.device = torch.device("cuda") if self.cuda else torch.device("cpu")

    def forward(self, param_dict: Dict[str, torch.Tensor]) -> torch.LongTensor:
        """
        A function wrapper designed for inference in exported script format.

        Note:
            Dict is used to allow easy inference of the exported jit model in Triton server using easy to understand
            naming convention.
            See https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#special-conventions-for-pytorch-backend

        Args:
            param_dict (dict):
                    Dictionary containing the arguments for speaker clustering.
                    See `forward_infer` function for the argument information.

            Returns:
                Y (LongTensor):
                    Speaker labels for the segments in the given input embeddings.
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

    def forward_infer(
        self,
        embeddings_in_scales: torch.Tensor,
        timestamps_in_scales: torch.Tensor,
        multiscale_segment_counts: torch.LongTensor,
        multiscale_weights: torch.Tensor,
        oracle_num_speakers: int = -1,
        max_rp_threshold: float = 0.15,
        max_num_speakers: int = 8,
        enhanced_count_thres: int = 40,
        sparse_search_volume: int = 30,
        fixed_thres: float = -1.0,
        kmeans_random_trials: int = 1,
    ) -> torch.LongTensor:
        """
        Calculate affinity matrix using timestamps and speaker embeddings, run NME analysis to estimate the best
        p-value and perform spectral clustering based on the estimated p-value and the calculated affinity matrix.

        Caution:
            For the sake of compatibility with libtorch, python boolean `False` is replaced with `torch.LongTensor(-1)`.

        Args:
            Dict containing following keys associated with tensors.
            embeddings (Tensor):
                Concatenated Torch tensor containing embeddings in multiple scales
                This tensor has dimensions of (Number of base segments) x (Embedding Dimension)
            timestamps (Tensor):
                Concatenated Torch tensor containing timestamps in multiple scales.
                This tensor has dimensions of (Total number of segments all scales) x 2
                Example:
                    >>> timestamps_in_scales = \
                    >>> torch.tensor([0.4, 1.4], [0.9, 1.9], [1.4, 2.4], ... [121.2, 122.2]])

            multiscale_segment_counts (LongTensor):
                Concatenated Torch tensor containing number of segments per each scale
                This tensor has dimensions of (Number of scales)
                Example:
                    >>> multiscale_segment_counts = torch.LongTensor([31, 52, 84, 105, 120])

            multiscale_weights (Tensor):
                Multi-scale weights that are used when affinity scores are merged.
                Example:
                    >>> multiscale_weights = torch.tensor([1.4, 1.3, 1.2, 1.1, 1.0])

            oracle_num_speakers (int):
                The number of speakers in a session from the reference transcript
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
                enhanced_count_thres = 80 is recommended.
            sparse_search_volume (int):
                Number of p_values we search during NME analysis.
                Default is 30. The lower the value, the faster NME-analysis becomes.
                Lower than 20 might cause a poor parameter estimation.
            fixed_thres (float):
                If fixed_thres value is provided, NME-analysis process will be skipped.
                This value should be optimized on a development set to obtain a quality result.
                Default is None and performs NME-analysis to estimate the threshold.
            kmeans_random_trials (int):
                Number of random trials for initializing k-means clustering. More trials
                will result in a more stable clustering result. Default is 1.

        Returns:
            Y (LongTensor):
                Speaker labels for the segments in the given input embeddings.
        """
        self.embeddings_in_scales, self.timestamps_in_scales = split_input_data(
            embeddings_in_scales, timestamps_in_scales, multiscale_segment_counts
        )
        # Last slot is the base scale embeddings
        emb = self.embeddings_in_scales[-1]

        # Cases for extreamly short sessions
        if emb.shape[0] == 1:
            return torch.zeros((1,), dtype=torch.int64)
        elif emb.shape[0] <= max(enhanced_count_thres, self.min_samples_for_nmesc) and oracle_num_speakers < 0:
            est_num_of_spk_enhanced = getEnhancedSpeakerCount(emb=emb, cuda=self.cuda)
        else:
            est_num_of_spk_enhanced = torch.tensor(-1)

        if oracle_num_speakers > 0:
            max_num_speakers = oracle_num_speakers

        mat = getMultiScaleCosAffinityMatrix(
            multiscale_weights, self.embeddings_in_scales, self.timestamps_in_scales, self.device
        )

        nmesc = NMESC(
            mat,
            max_num_speakers=max_num_speakers,
            max_rp_threshold=max_rp_threshold,
            sparse_search=self.sparse_search,
            sparse_search_volume=sparse_search_volume,
            fixed_thres=fixed_thres,
            nme_mat_size=self.nme_mat_size,
            maj_vote_spk_count=self.maj_vote_spk_count,
            parallelism=self.parallelism,
            cuda=self.cuda,
            device=self.device,
        )

        # If there are less than `min_samples_for_nmesc` segments, est_num_of_spk is 1.
        if mat.shape[0] > self.min_samples_for_nmesc:
            est_num_of_spk, p_hat_value = nmesc.forward()
            affinity_mat = getAffinityGraphMat(mat, p_hat_value)
        else:
            nmesc.fixed_thres = max_rp_threshold
            est_num_of_spk, p_hat_value = nmesc.forward()
            affinity_mat = mat

        # n_clusters is number of speakers estimated from spectral clustering.
        if oracle_num_speakers > 0:
            n_clusters = int(oracle_num_speakers)
        elif est_num_of_spk_enhanced > 0:
            n_clusters = int(est_num_of_spk_enhanced.item())
        else:
            n_clusters = int(est_num_of_spk.item())

        spectral_model = SpectralClustering(
            n_clusters=n_clusters, n_random_trials=kmeans_random_trials, cuda=self.cuda, device=self.device
        )
        Y = spectral_model.forward(affinity_mat)
        return Y
