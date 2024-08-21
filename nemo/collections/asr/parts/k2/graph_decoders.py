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

from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from jiwer import wer as word_error_rate
from omegaconf import DictConfig

from nemo.collections.asr.parts.k2.classes import GraphIntersectDenseConfig
from nemo.collections.asr.parts.k2.loss_mixins import CtcK2Mixin, RnntK2Mixin
from nemo.collections.asr.parts.k2.utils import (
    create_supervision,
    invert_permutation,
    levenshtein_graph_k2,
    load_graph,
)
from nemo.collections.asr.parts.submodules.wfst_decoder import (
    AbstractWFSTDecoder,
    WfstNbestHypothesis,
    collapse_tokenword_hypotheses,
)
from nemo.core.utils.k2_guard import k2
from nemo.utils import logging


class BaseDecoder(object):
    """Base graph decoder with topology for decoding graph.
    Typically uses the same parameters as for the corresponding loss function.

    Can do decoding and forced alignment.

    cfg takes precedence over all optional parameters
    We keep explicit parameter setting to be able to create an instance without the need of a config.
    """

    @abstractmethod
    def __init__(
        self,
        num_classes: int,
        blank: int,
        cfg: Optional[DictConfig] = None,
        intersect_pruned: bool = False,
        intersect_conf: GraphIntersectDenseConfig = GraphIntersectDenseConfig(),
        topo_type: str = "default",
        topo_with_self_loops: bool = True,
        device: torch.device = torch.device("cpu"),
    ):

        if cfg is not None:
            intersect_pruned = cfg.get("intersect_pruned", intersect_pruned)
            intersect_conf = cfg.get("intersect_conf", intersect_conf)
            topo_type = cfg.get("topo_type", topo_type)
            topo_with_self_loops = cfg.get("topo_with_self_loops", topo_with_self_loops)

        self.num_classes = num_classes
        self.blank = blank
        self.intersect_pruned = intersect_pruned
        self.device = device
        self.topo_type = topo_type
        self.topo_with_self_loops = topo_with_self_loops
        self.pad_fsavec = self.topo_type == "ctc_compact"
        self.intersect_conf = intersect_conf
        self.graph_compiler = None  # expected to be initialized in child classes
        self.base_graph = None  # expected to be initialized in child classes
        self.decoding_graph = None

    def to(self, device: torch.device):
        if self.graph_compiler.device != device:
            self.graph_compiler.to(device)
        if self.base_graph.device != device:
            self.base_graph = self.base_graph.to(device)
        if self.decoding_graph is not None and self.decoding_graph.device != device:
            self.decoding_graph = self.decoding_graph.to(device)
        self.device = device

    def update_graph(self, graph: 'k2.Fsa'):
        raise NotImplementedError

    def _decode_impl(
        self,
        log_probs: torch.Tensor,
        supervisions: torch.Tensor,
        return_lattices: bool = False,
        return_ilabels: bool = False,
        output_aligned: bool = True,
    ) -> Union['k2.Fsa', Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        if self.decoding_graph is None:
            self.decoding_graph = self.base_graph

        if log_probs.device != self.device:
            self.to(log_probs.device)
        emissions_graphs = self._prepare_emissions_graphs(log_probs, supervisions)

        if self.intersect_pruned:
            lats = k2.intersect_dense_pruned(
                a_fsas=self.decoding_graph,
                b_fsas=emissions_graphs,
                search_beam=self.intersect_conf.search_beam,
                output_beam=self.intersect_conf.output_beam,
                min_active_states=self.intersect_conf.min_active_states,
                max_active_states=self.intersect_conf.max_active_states,
            )
        else:
            indices = torch.zeros(emissions_graphs.dim0(), dtype=torch.int32, device=self.device)
            dec_graphs = (
                k2.index_fsa(self.decoding_graph, indices)
                if self.decoding_graph.shape[0] == 1
                else self.decoding_graph
            )
            lats = k2.intersect_dense(dec_graphs, emissions_graphs, self.intersect_conf.output_beam)
        self.decoding_graph = None

        order = supervisions[:, 0]
        if return_lattices:
            lats = k2.index_fsa(lats, invert_permutation(order).to(device=log_probs.device))
            if self.blank != 0:
                # change only ilabels
                # suppose self.blank == self.num_classes - 1
                lats.labels = torch.where(lats.labels == 0, self.blank, lats.labels - 1)
            return lats
        else:
            shortest_path_fsas = k2.index_fsa(
                k2.shortest_path(lats, True),
                invert_permutation(order).to(device=log_probs.device),
            )
            return self._extract_labels_and_probabilities(shortest_path_fsas, return_ilabels, output_aligned)

    def decode(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        return_lattices: bool = False,
        return_ilabels: bool = False,
        output_aligned: bool = True,
    ) -> Union['k2.Fsa', Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        log_probs, supervisions, _, _ = self._prepare_log_probs_and_targets(log_probs, log_probs_length, None, None)
        return self._decode_impl(
            log_probs,
            supervisions,
            return_lattices=return_lattices,
            return_ilabels=return_ilabels,
            output_aligned=output_aligned,
        )

    def align(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        return_lattices: bool = False,
        return_ilabels: bool = False,
        output_aligned: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        log_probs, supervisions, targets, target_lengths = self._prepare_log_probs_and_targets(
            log_probs, log_probs_length, targets, target_lengths
        )
        order = supervisions[:, 0].to(dtype=torch.long)
        self.decoding_graph = self.graph_compiler.compile(targets[order], target_lengths[order])
        return self._decode_impl(
            log_probs,
            supervisions,
            return_lattices=return_lattices,
            return_ilabels=return_ilabels,
            output_aligned=output_aligned,
        )


class CtcDecoder(BaseDecoder, CtcK2Mixin):
    """Regular CTC graph decoder with custom topologies.
    Available topologies:
        -   `default`, with or without self-loops
        -   `compact`, with or without self-loops
        -   `shared_blank`, with or without self-loops
        -   `minimal`, without self-loops

    Can do decoding and forced alignment.
    """

    def __init__(
        self,
        num_classes: int,
        blank: int,
        cfg: Optional[DictConfig] = None,
        intersect_pruned: bool = False,
        intersect_conf: GraphIntersectDenseConfig = GraphIntersectDenseConfig(),
        topo_type: str = "default",
        topo_with_self_loops: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            num_classes, blank, cfg, intersect_pruned, intersect_conf, topo_type, topo_with_self_loops, device
        )
        from nemo.collections.asr.parts.k2.graph_compilers import CtcTopologyCompiler

        self.graph_compiler = CtcTopologyCompiler(
            self.num_classes, self.blank, self.topo_type, self.topo_with_self_loops, self.device
        )
        self.base_graph = k2.create_fsa_vec([self.graph_compiler.ctc_topo_inv.invert()]).to(self.device)


class RnntAligner(BaseDecoder, RnntK2Mixin):
    """RNNT graph decoder with the `minimal` topology.
    If predictor_window_size is not provided, this decoder works as a Viterbi over regular RNNT lattice.
    With predictor_window_size provided, it applies uniform pruning when compiling Emission FSAs
    to reduce memory and compute consumption.

    Can only do forced alignment.
    """

    def __init__(
        self,
        num_classes: int,
        blank: int,
        cfg: Optional[DictConfig] = None,
        intersect_pruned: bool = False,
        intersect_conf: GraphIntersectDenseConfig = GraphIntersectDenseConfig(),
        topo_type: str = "default",
        topo_with_self_loops: bool = True,
        predictor_window_size: int = 0,
        predictor_step_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        if cfg is not None:
            topo_type = cfg.get("topo_type", topo_type)
            predictor_window_size = cfg.get("predictor_window_size", predictor_window_size)
            predictor_step_size = cfg.get("predictor_step_size", predictor_step_size)
        if topo_type != "minimal":
            raise NotImplementedError(f"Only topo_type=`minimal` is supported at the moment.")
        super().__init__(
            num_classes, blank, cfg, intersect_pruned, intersect_conf, topo_type, topo_with_self_loops, device
        )
        self.predictor_window_size = predictor_window_size
        self.predictor_step_size = predictor_step_size
        from nemo.collections.asr.parts.k2.graph_compilers import RnntTopologyCompiler

        self.graph_compiler = RnntTopologyCompiler(
            self.num_classes,
            self.blank,
            self.topo_type,
            self.topo_with_self_loops,
            self.device,
            max_adapter_length=self.predictor_window_size,
        )
        self.base_graph = self.graph_compiler.base_graph

    def decode(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        return_lattices: bool = False,
        return_ilabels: bool = False,
        output_aligned: bool = True,
    ) -> Union['k2.Fsa', Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        raise NotImplementedError("RNNT decoding is not implemented. Only .align(...) method is supported.")

    def align(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        return_lattices: bool = False,
        return_ilabels: bool = False,
        output_aligned: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        assert self.predictor_window_size == 0 or log_probs.size(2) <= self.predictor_window_size + 1

        return super().align(
            log_probs,
            log_probs_length,
            targets,
            target_lengths,
            return_lattices=return_lattices,
            return_ilabels=return_ilabels,
            output_aligned=output_aligned,
        )


class TokenLMDecoder(BaseDecoder):
    """Graph decoder with token_lm-based decoding graph.
    Available topologies:
        -   `default`, with or without self-loops
        -   `compact`, with or without self-loops
        -   `shared_blank`, with or without self-loops
        -   `minimal`, without self-loops

    Can do decoding and forced alignment.

    cfg takes precedence over all optional parameters
    We keep explicit parameter setting to be able to create an instance without the need of a config.
    """

    def __init__(
        self,
        num_classes: int,
        blank: int,
        cfg: Optional[DictConfig] = None,
        token_lm: Optional[Union['k2.Fsa', str]] = None,
        intersect_pruned: bool = False,
        intersect_conf: GraphIntersectDenseConfig = GraphIntersectDenseConfig(),
        topo_type: str = "default",
        topo_with_self_loops: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            num_classes, blank, cfg, intersect_pruned, intersect_conf, topo_type, topo_with_self_loops, device
        )
        if cfg is not None:
            token_lm = cfg.get("token_lm", token_lm)
        if token_lm is not None:
            self.token_lm = load_graph(token_lm) if isinstance(token_lm, str) else token_lm
            if self.token_lm is not None:
                self.update_graph(self.token_lm)
            else:
                logging.warning(
                    f"""token_lm was set to None. Use this for debug 
                                purposes only or call .update_graph(token_lm) before using."""
                )
        else:
            logging.warning(
                f"""token_lm was set to None. Use this for debug
                            purposes only or call .update_graph(token_lm) before using."""
            )
            self.token_lm = None

    def update_graph(self, graph: 'k2.Fsa'):
        self.token_lm = graph
        token_lm = self.token_lm.clone()
        if hasattr(token_lm, "aux_labels"):
            delattr(token_lm, "aux_labels")
        labels = token_lm.labels
        if labels.max() != self.num_classes - 1:
            raise ValueError(f"token_lm is not compatible with the num_classes: {labels.unique()}, {self.num_classes}")
        self.graph_compiler = CtcNumGraphCompiler(
            self.num_classes, self.blank, self.topo_type, self.topo_with_self_loops, self.device, token_lm
        )
        self.base_graph = k2.create_fsa_vec([self.graph_compiler.base_graph]).to(self.device)


class K2WfstDecoder(AbstractWFSTDecoder):
    """
    Used for performing WFST decoding of the logprobs with the k2 WFST decoder.

    Args:
      lm_fst:
        Kaldi-type language model WFST or its path.

      decoding_mode:
        Decoding mode. Choices: `nbest`, `lattice`.

      beam_size:
        Beam width (float) for the WFST decoding.

      config:
        Riva Decoder config.

      tokenword_disambig_id:
        Tokenword disambiguation index. Set to -1 to disable the tokenword mode.

      lm_weight:
        Language model weight in decoding.

      nbest_size:
        N-best size for decoding_mode == `nbest`

      device:
        Device for running decoding. Choices: `cuda`, `cpu`.
    """

    def __init__(
        self,
        lm_fst: Union['k2.Fsa', Path, str],
        decoding_mode: str = 'nbest',
        beam_size: float = 10.0,
        config: Optional[GraphIntersectDenseConfig] = None,
        tokenword_disambig_id: int = -1,
        lm_weight: float = 1.0,
        nbest_size: int = 1,
        device: str = "cuda",
    ):
        self._nbest_size = nbest_size
        self._device = device
        super().__init__(lm_fst, decoding_mode, beam_size, config, tokenword_disambig_id, lm_weight)

    def _set_decoder_config(self, config: Optional[GraphIntersectDenseConfig] = None):
        if config is None:
            config = GraphIntersectDenseConfig()
            config.search_beam = 20.0
            config.output_beam = self._beam_size
            config.max_active_states = 10000
        self._config = config

    def _set_decoding_mode(self, decoding_mode: str):
        if decoding_mode not in ('nbest', 'lattice'):
            raise ValueError(f"Unsupported mode: {decoding_mode}")
        self._decoding_mode = decoding_mode

    @torch.inference_mode(False)
    def _init_decoder(self):
        lm_fst = load_graph(self._lm_fst) if isinstance(self._lm_fst, (Path, str)) else self._lm_fst.clone()
        lm_fst.lm_scores = lm_fst.scores.clone()
        self._lm_fst = lm_fst.to(device=self._device)

        if self._id2word is None:
            self._id2word = {
                int(line.split()[1]): line.split()[0]
                for line in self._lm_fst.aux_labels_sym.to_str().strip().split("\n")
            }
            word2id = self._id2word.__class__(map(reversed, self._id2word.items()))
            word_unk_id = word2id["<unk>"]
            self._word2id = defaultdict(lambda: word_unk_id)
            for k, v in word2id.items():
                self._word2id[k] = v
        if self._id2token is None:
            self._id2token = {
                int(line.split()[1]): line.split()[0] for line in self._lm_fst.labels_sym.to_str().strip().split("\n")
            }
            token2id = self._id2token.__class__(map(reversed, self._id2token.items()))
            token_unk_id = token2id["<unk>"]
            self._token2id = defaultdict(lambda: token_unk_id)
            for k, v in token2id.items():
                self._token2id[k] = v

    def _beam_size_setter(self, value: float):
        if self._beam_size != value:
            self._config.output_beam = value
            self._beam_size = value

    def _lm_weight_setter(self, value: float):
        if self._lm_weight != value:
            self._lm_weight = value

    @property
    def nbest_size(self):
        return self._nbest_size

    @nbest_size.setter
    def nbest_size(self, value: float):
        self._nbest_size_setter(value)

    def _nbest_size_setter(self, value: float):
        if self._nbest_size != value:
            self._nbest_size = value

    def _decoding_mode_setter(self, value: str):
        if self._decoding_mode != value:
            self._set_decoding_mode(value)

    @torch.inference_mode(False)
    def _decode_lattice(self, emissions_fsas: 'k2.DenseFsaVec', order: torch.Tensor) -> 'k2.Fsa':
        """
        Decodes logprobs into k2-type lattices.

        Args:
          emissions_fsas:
            A k2.DenseFsaVec of the predicted log-probabilities.
          order:
            A torch.Tensor that stores the order of the emissions_fsas elements.

        Returns:
          k2-type FsaVec.
        """
        lats = k2.intersect_dense_pruned(
            a_fsas=self._lm_fst,
            b_fsas=emissions_fsas,
            search_beam=self._config.search_beam,
            output_beam=self._config.output_beam,
            min_active_states=self._config.min_active_states,
            max_active_states=self._config.max_active_states,
            frame_idx_name="frame_idx",
            allow_partial=True,
        )
        lats = k2.connect(k2.expand_ragged_attributes(lats))
        lats.am_scores = lats.scores - lats.lm_scores
        if self._lm_weight != 1.0:
            lats.scores = lats.am_scores + self._lm_weight * lats.lm_scores
        # just in case
        lats.__dict__["_properties"] = None
        return k2.index_fsa(lats, invert_permutation(order).to(device=self._device))

    @torch.inference_mode(False)
    def decode(
        self, log_probs: torch.Tensor, log_probs_length: torch.Tensor
    ) -> Union[List[WfstNbestHypothesis], List['k2.Fsa']]:
        """
        Decodes logprobs into recognition hypotheses.

        Args:
          log_probs:
            A torch.Tensor of the predicted log-probabilities of shape [Batch, Time, Vocabulary].

          log_probs_length:
            A torch.Tensor of length `Batch` which contains the lengths of the log_probs elements.

        Returns:
          List of recognition hypotheses.
        """
        supervisions = create_supervision(log_probs_length)
        order = supervisions[:, 0]
        emissions_fsas = k2.DenseFsaVec(log_probs.to(device=self._device), supervisions)
        lats = self._decode_lattice(emissions_fsas, order)
        hypotheses = self._post_decode(lats)
        return hypotheses

    @torch.inference_mode(False)
    def _post_decode(self, hypotheses: 'k2.Fsa') -> Union[List[WfstNbestHypothesis], List['k2.Fsa']]:
        """
        Does various post-processing of the recognition hypotheses.

        Args:
          hypotheses:
            FsaVec of k2-type lattices.

        Returns:
          List of processed recognition hypotheses.
        """
        if self._decoding_mode == 'nbest':
            hypotheses_fsa = hypotheses
            hypotheses = []
            if self._nbest_size == 1:
                shortest_path_fsas = k2.shortest_path(hypotheses_fsa, True)
                scores = shortest_path_fsas.get_tot_scores(True, False).tolist()
                # direct iterating does not work as expected
                for i in range(shortest_path_fsas.shape[0]):
                    fsa = shortest_path_fsas[i]
                    non_eps_mask = fsa.aux_labels > 0
                    words = [self._id2word[l] for l in fsa.aux_labels[non_eps_mask].tolist()]
                    alignment = fsa.labels[fsa.labels > 0].tolist()
                    # some timesteps may be 0 if self.open_vocabulary_decoding
                    timesteps = fsa.frame_idx[non_eps_mask]
                    timesteps_left = timesteps[:-1]
                    timesteps_right = timesteps[1:]
                    timesteps_right_zero_mask = timesteps_right == 0
                    timesteps_right[timesteps_right_zero_mask] = timesteps_left[timesteps_right_zero_mask]
                    timesteps[1:] = timesteps_right
                    timesteps = timesteps.tolist()
                    hypotheses.append(
                        WfstNbestHypothesis(
                            tuple(
                                [
                                    tuple([tuple(words), tuple(timesteps), tuple(alignment), -scores[i]]),
                                ]
                            )
                        )
                    )
            else:
                nbest_fsas = k2.Nbest.from_lattice(hypotheses_fsa, self._nbest_size)
                nbest_fsas.fsa.frame_idx = k2.index_select(hypotheses_fsa.frame_idx, nbest_fsas.kept_path.values)
                scores = nbest_fsas.fsa.get_tot_scores(True, False).tolist()
                nbest_hypothesis_list = [[] for _ in range(nbest_fsas.shape.dim0)]
                for i, j in enumerate(nbest_fsas.shape.row_ids(1)):
                    fsa = nbest_fsas.fsa[i]
                    non_eps_mask = fsa.aux_labels > 0
                    words = [self._id2word[l] for l in fsa.aux_labels[non_eps_mask].tolist()]
                    alignment = fsa.labels[fsa.labels > 0].tolist()
                    # some timesteps may be 0 if self.open_vocabulary_decoding
                    timesteps = fsa.frame_idx[non_eps_mask]
                    timesteps_left = timesteps[:-1]
                    timesteps_right = timesteps[1:]
                    timesteps_right_zero_mask = timesteps_right == 0
                    timesteps_right[timesteps_right_zero_mask] = timesteps_left[timesteps_right_zero_mask]
                    timesteps[1:] = timesteps_right
                    timesteps = timesteps.tolist()
                    nbest_hypothesis_list[j].append(
                        tuple([tuple(words), tuple(timesteps), tuple(alignment), -scores[i]])
                    )
                for nbest_hypothesis in nbest_hypothesis_list:
                    hypotheses.append(WfstNbestHypothesis(tuple(nbest_hypothesis)))
            return (
                collapse_tokenword_hypotheses(hypotheses, self._id2word[self._tokenword_disambig_id])
                if self._open_vocabulary_decoding
                else hypotheses
            )
        else:
            return [hypotheses[i].to(device="cpu") for i in range(len(hypotheses))]

    @torch.inference_mode(False)
    def calibrate_lm_weight(
        self, log_probs: torch.Tensor, log_probs_length: torch.Tensor, reference_texts: List[str]
    ) -> Tuple[float, float]:
        """
        Calibrates LM weight to achieve the best WER for given logprob-text pairs.

        Args:
          log_probs:
            A torch.Tensor of the predicted log-probabilities of shape [Batch, Time, Vocabulary].

          log_probs_length:
            A torch.Tensor of length `Batch` which contains the lengths of the log_probs elements.

          reference_texts:
            List of reference word sequences.

        Returns:
          Pair of (best_lm_weight, best_wer).
        """
        assert len(log_probs) == len(reference_texts)
        decoding_mode_backup = self.decoding_mode
        lm_weight_backup = self.lm_weight
        nbest_size_backup = self.nbest_size
        self.decoding_mode = "lattice"
        lattices = self.decode(log_probs, log_probs_length)
        best_lm_weight, best_wer = -1.0, float('inf')
        self.decoding_mode = "nbest"
        self.nbest_size = 1
        for lm_weight in range(1, 21):  # enough for most cases
            lm_weight_act = lm_weight / 10
            for lat in lattices:
                lat.scores = lat.am_scores + lm_weight_act * lat.lm_scores
            hypotheses = self._post_decode(lattices)
            wer = word_error_rate([" ".join(h[0].words) for h in hypotheses], reference_texts)
            if wer < best_wer:
                best_lm_weight, best_wer = lm_weight_act, wer
        self.nbest_size = nbest_size_backup
        self.decoding_mode = decoding_mode_backup
        self.lm_weight = lm_weight_backup
        return best_lm_weight, best_wer

    @torch.inference_mode(False)
    def calculate_oracle_wer(
        self, log_probs: torch.Tensor, log_probs_length: torch.Tensor, reference_texts: List[str]
    ) -> Tuple[float, List[float]]:
        """
        Calculates the oracle (the best possible WER for given logprob-text pairs.

        Args:
          log_probs:
            A torch.Tensor of the predicted log-probabilities of shape [Batch, Time, Vocabulary].

          log_probs_length:
            A torch.Tensor of length `Batch` which contains the lengths of the log_probs elements.

          reference_texts:
            List of reference word sequences.

        Returns:
          Pair of (oracle_wer, oracle_wer_per_utterance).
        """
        if self._open_vocabulary_decoding:
            raise NotImplementedError
        assert len(log_probs) == len(reference_texts)
        word_ids = [[self._word2id[w] for w in text.split()] for text in reference_texts]
        counts = torch.tensor([len(wid) for wid in word_ids])
        decoding_mode_backup = self.decoding_mode
        self.decoding_mode = "lattice"
        lattices = self.decode(log_probs, log_probs_length)
        oracle_disambig = max(self._id2word.keys()) + 1
        lattices.aux_labels[lattices.aux_labels == 0] = oracle_disambig
        lattices = lattices.invert()
        delattr(lattices, 'aux_labels')
        hyps = levenshtein_graph_k2(lattices).invert()
        refs = levenshtein_graph_k2(k2.linear_fsa(word_ids))
        refs, arc_map = k2.add_epsilon_self_loops(refs, ret_arc_map=True)
        labels = refs.labels.clone()
        labels[arc_map == -1] = oracle_disambig
        refs.labels = labels
        refs.__dict__["_properties"] = None
        refs = k2.arc_sort(refs)
        ali_lats = k2.compose(hyps, refs, treat_epsilons_specially=False)
        ali_lats = k2.remove_epsilon_self_loops(ali_lats)
        # TODO: find out why it fails for some utterances
        try:
            alignment = k2.shortest_path(ali_lats, use_double_scores=True)
        except RuntimeError as e:
            logging.warning("calculate_oracle_wer failed")
            return -1.0, []
        scores = -alignment.get_tot_scores(True, True).to(dtype=torch.int64)
        wer_per_utt = scores / counts
        self.decoding_mode = decoding_mode_backup
        return (scores.sum() / counts.sum()).item(), wer_per_utt.tolist()
