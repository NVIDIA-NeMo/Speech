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

import copy
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm.auto import tqdm

from nemo.collections.asr.data.audio_to_ctm_dataset import FrameCtmUnit
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.utils import logging


class AlignerWrapperModel(ASRModel):
    """ASR model wrapper to perform alignment building.
    Functionality is limited to the components needed to build an alignment."""

    def __init__(self, model: ASRModel, cfg: DictConfig):
        model_cfg = model.cfg
        for ds in ("train_ds", "validation_ds", "test_ds"):
            if ds in model_cfg:
                model_cfg[ds] = None
        super().__init__(cfg=model_cfg, trainer=model.trainer)
        self._model = model
        self.alignment_type = cfg.get("alignment_type", "forced")
        self.word_output = cfg.get("word_output", True)
        self.cpu_decoding = cfg.get("cpu_decoding", False)
        self.decode_batch_size = cfg.get("decode_batch_size", 0)

        # list possible alignment types here for future work
        if self.alignment_type == "forced":
            pass
        elif self.alignment_type == "argmax":
            pass
        elif self.alignment_type == "loose":
            raise NotImplementedError(f"alignment_type=`{self.alignment_type}` is not supported at the moment.")
        elif self.alignment_type == "rnnt_decoding_aux":
            raise NotImplementedError(f"alignment_type=`{self.alignment_type}` is not supported at the moment.")
        else:
            raise RuntimeError(f"Unsupported alignment type: {self.alignment_type}")

        self._init_model_specific(cfg)

    def _init_ctc_alignment_specific(self, cfg: DictConfig):
        """Part of __init__ intended to initialize attributes specific to the alignment type for CTC models.

        This method is not supposed to be called outside of __init__.
        """
        # do nothing for regular CTC with `argmax` alignment type
        if self.alignment_type == "argmax" and not hasattr(self._model, "use_graph_lm"):
            return

        from nemo.collections.asr.modules.graph_decoder import ViterbiDecoderWithGraph

        if self.alignment_type == "forced":
            if hasattr(self._model, "use_graph_lm"):
                if self._model.use_graph_lm:
                    self.graph_decoder = self._model.transcribe_decoder
                    self._model.use_graph_lm = False
                else:
                    self.graph_decoder = ViterbiDecoderWithGraph(
                        num_classes=self.blank_id, backend="k2", dec_type="topo", return_type="1best"
                    )
                # override split_batch_size
                self.graph_decoder.split_batch_size = self.decode_batch_size
            else:
                self.graph_decoder = ViterbiDecoderWithGraph(
                    num_classes=self.blank_id, split_batch_size=self.decode_batch_size,
                )
            # override decoder args if a config is provided
            decoder_module_cfg = cfg.get("decoder_module_cfg", None)
            if decoder_module_cfg is not None:
                self.graph_decoder._decoder.intersect_pruned = decoder_module_cfg.get("intersect_pruned")
                self.graph_decoder._decoder.intersect_conf = decoder_module_cfg.get("intersect_conf")
            return

        if self.alignment_type == "argmax":
            # we use transcribe_decoder to get topology-independent output
            if not self._model.use_graph_lm:
                self._model.transcribe_decoder = ViterbiDecoderWithGraph(
                    num_classes=self.blank_id, backend="k2", dec_type="topo", return_type="1best"
                )
            # override decoder args
            self._model.transcribe_decoder.return_ilabels = False
            self._model.transcribe_decoder.output_aligned = True
            self._model.transcribe_decoder.split_batch_size = self.decode_batch_size
            self._model.use_graph_lm = False
            return

    def _init_rnnt_alignment_specific(self, cfg: DictConfig):
        """Part of __init__ intended to initialize attributes specific to the alignment type for RNNT models.

        This method is not supposed to be called outside of __init__.
        """
        if self.alignment_type == "argmax":
            return

        from nemo.collections.asr.modules.graph_decoder import ViterbiDecoderWithGraph

        if self.alignment_type == "forced":
            self.predictor_window_size = cfg.rnnt_cfg.get("predictor_window_size", 0)
            self.predictor_step_size = cfg.rnnt_cfg.get("predictor_step_size", 0)

            from nemo.collections.asr.parts.k2.utils import apply_rnnt_prune_ranges, get_uniform_rnnt_prune_ranges

            self.prepare_pruned_outputs = lambda encoder_outputs, encoded_len, decoder_outputs, transcript_len: apply_rnnt_prune_ranges(
                encoder_outputs,
                decoder_outputs,
                get_uniform_rnnt_prune_ranges(
                    encoded_len,
                    transcript_len,
                    self.predictor_window_size + 1,
                    self.predictor_step_size,
                    encoder_outputs.size(1),
                ).to(device=encoder_outputs.device),
            )

            from nemo.collections.asr.parts.k2.classes import GraphModuleConfig

            self.graph_decoder = ViterbiDecoderWithGraph(
                num_classes=self.blank_id,
                backend="k2",
                dec_type="topo_rnnt_ali",
                split_batch_size=self.decode_batch_size,
                graph_module_cfg=OmegaConf.structured(
                    GraphModuleConfig(
                        topo_type="minimal",
                        predictor_window_size=self.predictor_window_size,
                        predictor_step_size=self.predictor_step_size,
                    )
                ),
            )
            # override decoder args if a config is provided
            decoder_module_cfg = cfg.get("decoder_module_cfg", None)
            if decoder_module_cfg is not None:
                self.graph_decoder._decoder.intersect_pruned = decoder_module_cfg.get("intersect_pruned")
                self.graph_decoder._decoder.intersect_conf = decoder_module_cfg.get("intersect_conf")
            return

    def _init_model_specific(self, cfg: DictConfig):
        """Part of __init__ intended to initialize attributes specific to the model type.

        This method is not supposed to be called outside of __init__.
        """
        from nemo.collections.asr.models.ctc_models import EncDecCTCModel

        if isinstance(self._model, EncDecCTCModel):
            self.model_type = "ctc"
            self.blank_id = self._model.decoder.num_classes_with_blank - 1
            self._predict_impl = self._predict_impl_ctc

            prob_suppress_index = cfg.ctc_cfg.get("prob_suppress_index", -1)
            prob_suppress_value = cfg.ctc_cfg.get("prob_suppress_value", 1.0)
            if prob_suppress_value > 1 or prob_suppress_value <= 0:
                raise ValueError(f"Suppression value has to be in (0,1]: {prob_suppress_value}")
            if prob_suppress_index < -(self.blank_id + 1) or prob_suppress_index > self.blank_id:
                raise ValueError(
                    f"Suppression index for the provided model has to be in [{-self.blank_id+1},{self.blank_id}]: {prob_suppress_index}"
                )
            self.prob_suppress_index = (
                self._model.decoder.num_classes_with_blank + prob_suppress_index
                if prob_suppress_index < 0
                else prob_suppress_index
            )
            self.prob_suppress_value = prob_suppress_value

            self._init_ctc_alignment_specific(cfg)
            return

        from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel

        if isinstance(self._model, EncDecRNNTModel):
            self.model_type = "rnnt"
            self.blank_id = self._model.joint.num_classes_with_blank - 1
            self.log_softmax = None if self._model.joint.log_softmax is None else not self._model.joint.log_softmax
            self._predict_impl = self._predict_impl_rnnt

            decoding_config = copy.deepcopy(self._model.cfg.decoding)
            decoding_config.strategy = "greedy_batch"
            with open_dict(decoding_config):
                decoding_config.preserve_alignments = True
                decoding_config.fused_batch_size = -1
            self._model.change_decoding_strategy(decoding_config)
            self._init_rnnt_alignment_specific(cfg)
            return

        raise RuntimeError(f"Unsupported model type: {type(self._model)}")

    def _rnnt_joint_pruned(
        self,
        encoder_outputs: torch.Tensor,
        encoded_len: torch.Tensor,
        decoder_outputs: torch.Tensor,
        transcript_len: torch.Tensor,
    ) -> torch.Tensor:
        """A variant of the RNNT Joiner tensor calculation with pruned Encoder and Predictor sum.
        Only the uniform pruning is supported at the moment.
        """
        encoder_outputs = self._model.joint.enc(encoder_outputs.transpose(1, 2))  # (B, T, H)
        decoder_outputs = self._model.joint.pred(decoder_outputs.transpose(1, 2))  # (B, U, H)

        encoder_outputs_pruned, decoder_outputs_pruned = self.prepare_pruned_outputs(
            encoder_outputs, encoded_len, decoder_outputs, transcript_len
        )
        res = self._model.joint.joint_net(encoder_outputs_pruned + decoder_outputs_pruned)
        # copied from model.joint.joint(...)
        if self._model.joint.log_softmax is None:
            if not res.is_cuda:
                res = res.log_softmax(dim=-1)
        else:
            if self._model.joint.log_softmax:
                res = res.log_softmax(dim=-1)
        return res

    def _apply_prob_suppress(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Multiplies probability of an element with index self.prob_suppress_index by self.prob_suppress_value times
        with stochasticity preservation of the log_probs tensor.
        
        Often used to suppress <blank> probability of the output of a CTC model.
        
        Example:
            For
                - log_probs = torch.log(torch.tensor([0.015, 0.085, 0.9]))
                - self.prob_suppress_index = -1
                - self.prob_suppress_value = 0.5
            the result of _apply_prob_suppress(log_probs) is
                - torch.log(torch.tensor([0.0825, 0.4675, 0.45]))
        """
        exp_probs = (log_probs).exp()
        x = exp_probs[:, :, self.prob_suppress_index]
        # we cannot do y=1-x because exp_probs can be not stochastic due to numerical limitations
        y = torch.cat(
            [exp_probs[:, :, : self.prob_suppress_index], exp_probs[:, :, self.prob_suppress_index + 1 :]], 2
        ).sum(-1)
        b1 = torch.full((exp_probs.shape[0], exp_probs.shape[1], 1), self.prob_suppress_value, device=log_probs.device)
        b2 = ((1 - self.prob_suppress_value * x) / y).unsqueeze(2).repeat(1, 1, exp_probs.shape[-1] - 1)
        return (
            exp_probs * torch.cat([b2[:, :, : self.prob_suppress_index], b1, b2[:, :, self.prob_suppress_index :]], 2)
        ).log()

    def _prepare_ctc_argmax_predictions(
        self, log_probs: torch.Tensor, encoded_len: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Obtains argmax predictions with corresponding probabilities.
        Replaces consecutive repeated indices in the argmax predictions with the <blank> index.
        """
        if hasattr(self._model, "transcribe_decoder"):
            predictions, _, probs = self.transcribe_decoder.forward(log_probs=log_probs, log_probs_length=encoded_len)
        else:
            greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
            probs_tensor, _ = log_probs.exp().max(dim=-1, keepdim=False)
            predictions, probs = [], []
            for i in range(log_probs.shape[0]):
                utt_len = encoded_len[i]
                probs.append(probs_tensor[i, :utt_len])
                pred_candidate = greedy_predictions[i, :utt_len].cpu()
                # replace consecutive tokens with <blank>
                previous = self.blank_id
                for j in range(utt_len):
                    p = pred_candidate[j]
                    if p == previous and previous != self.blank_id:
                        pred_candidate[j] = self.blank_id
                    previous = p
                predictions.append(pred_candidate.to(device=greedy_predictions.device))
        return predictions, probs

    def _predict_impl_rnnt_argmax(
        self,
        encoded: torch.Tensor,
        encoded_len: torch.Tensor,
        transcript: torch.Tensor,
        transcript_len: torch.Tensor,
        sample_id: torch.Tensor,
    ) -> List[Tuple[int, 'FrameCtmUnit']]:
        """Builds time alignment of an encoded sequence.
        This method assumes that the RNNT model is used and the alignment type is `argmax`.

        It produces a list of sample ids and fours: (label, start_frame, length, probability), called FrameCtmUnit.
        """
        hypotheses = self._model.decoding.rnnt_decoder_predictions_tensor(
            encoded, encoded_len, return_hypotheses=True
        )[0]
        results = []
        for s_id, hypothesis in zip(sample_id, hypotheses):
            pred_ids = hypothesis.y_sequence.tolist()
            tokens = self._model.decoding.decode_ids_to_tokens(pred_ids)
            token_begin = hypothesis.timestep
            token_len = [j - i for i, j in zip(token_begin, token_begin[1:] + [len(hypothesis.alignments)])]
            # we have no token probabilities for the argmax rnnt setup
            token_prob = [1.0] * len(tokens)
            if self.word_output:
                words = [w for w in self._model.decoding.decode_tokens_to_str(pred_ids).split(" ") if w != ""]
                words, word_begin, word_len, word_prob = (
                    self._process_tokens_to_words(tokens, token_begin, token_len, token_prob, words)
                    if hasattr(self._model, "tokenizer")
                    else self._process_char_with_space_to_words(tokens, token_begin, token_len, token_prob, words)
                )
                results.append(
                    (s_id, [FrameCtmUnit(t, b, l, p) for t, b, l, p in zip(words, word_begin, word_len, word_prob)])
                )
            else:
                results.append(
                    (
                        s_id,
                        [FrameCtmUnit(t, b, l, p) for t, b, l, p in zip(tokens, token_begin, token_len, token_prob)],
                    )
                )
        return results

    def _process_tokens_to_words(
        self,
        tokens: List[str],
        token_begin: List[int],
        token_len: List[int],
        token_prob: List[float],
        words: List[str],
    ) -> Tuple[List[str], List[int], List[int], List[float]]:
        """Transforms alignment information from token level to word level.

        Used when self._model.tokenizer is present.
        """
        # suppose that there are no whitespaces
        assert len(self._model.tokenizer.text_to_tokens(words[0])) == len(
            self._model.tokenizer.text_to_tokens(words[0] + " ")
        )
        word_begin, word_len, word_prob = [], [], []
        token_len_nonzero = [(t_l if t_l > 0 else 1) for t_l in token_len]
        i = 0
        for word in words:
            loc_tokens = self._model.tokenizer.text_to_tokens(word)
            step = len(loc_tokens)
            # we assume that an empty word consists of only one token
            # drop current token
            if step == 0:
                token_begin[i + 1] = token_begin[i]
                token_len[i + 1] += token_len[i]
                token_len_nonzero[i + 1] += token_len_nonzero[i]
                del tokens[i], token_begin[i], token_len[i], token_len_nonzero[i], token_prob[i]
                continue
            # fix <unk> tokenization
            if step == 2 and loc_tokens[-1] == "??":
                step -= 1
            j = i + step
            word_begin.append(token_begin[i])
            word_len.append(sum(token_len[i:j]))
            denominator = sum(token_len_nonzero[i:j])
            word_prob.append(sum(token_prob[k] * token_len_nonzero[k] for k in range(i, j)) / denominator)
            i = j
        return words, word_begin, word_len, word_prob

    def _process_char_with_space_to_words(
        self,
        tokens: List[str],
        token_begin: List[int],
        token_len: List[int],
        token_prob: List[float],
        words: List[str],
    ) -> Tuple[List[str], List[int], List[int], List[float]]:
        """Transforms alignment information from character level to word level.
        This method includes separator (typically the space) information in the results.

        Used with character-based models (no self._model.tokenizer).
        """
        # suppose that there are no whitespaces anywhere except between words
        space_idx = (np.array(tokens) == " ").nonzero()[0].tolist()
        assert len(words) == len(space_idx) + 1
        token_len_nonzero = [(t_l if t_l > 0 else 1) for t_l in token_len]
        if len(space_idx) == 0:
            word_begin = [token_begin[0]]
            word_len = [sum(token_len)]
            denominator = sum(token_len_nonzero)
            word_prob = [sum(t_p * t_l for t_p, t_l in zip(token_prob, token_len_nonzero)) / denominator]
        else:
            space_word = "[SEP]"
            word_begin = [token_begin[0]]
            word_len = [sum(token_len[: space_idx[0]])]
            denominator = sum(token_len_nonzero[: space_idx[0]])
            word_prob = [sum(token_prob[k] * token_len_nonzero[k] for k in range(space_idx[0])) / denominator]
            words_with_space = [words[0]]
            for word, i, j in zip(words[1:], space_idx, space_idx[1:] + [len(tokens)]):
                # append space
                word_begin.append(token_begin[i])
                word_len.append(token_len[i])
                word_prob.append(token_prob[i])
                words_with_space.append(space_word)
                # append next word
                word_begin.append(token_begin[i + 1])
                word_len.append(sum(token_len[i + 1 : j]))
                denominator = sum(token_len_nonzero[i + 1 : j])
                word_prob.append(sum(token_prob[k] * token_len_nonzero[k] for k in range(i + 1, j)) / denominator)
                words_with_space.append(word)
            words = words_with_space
        return words, word_begin, word_len, word_prob

    def _results_to_ctmUnits(
        self, s_id: int, pred: torch.Tensor, prob: torch.Tensor
    ) -> Tuple[int, List['FrameCtmUnit']]:
        """Transforms predictions with probabilities to a list of FrameCtmUnit objects, 
        containing frame-level alignment information (label, start, duration, probability), for a given sample id.

        Alignment information can be either token-based (char, wordpiece, ...) or word-based.
        """
        if len(pred) == 0:
            return (s_id, [])

        non_blank_idx = (pred != self.blank_id).nonzero(as_tuple=True)[0].cpu()
        pred_ids = pred[non_blank_idx].tolist()
        prob_list = prob.tolist()
        if self.model_type == "rnnt":
            wer_module = self._model.decoding
            # for rnnt forced alignment we always have num_blanks == num_frames,
            # thus len(pred) == num_frames + num_non_blanks
            token_begin = non_blank_idx - torch.arange(len(non_blank_idx))
            token_end = torch.cat((token_begin[1:], torch.tensor([len(pred) - len(non_blank_idx)])))
        else:
            wer_module = self._model._wer
            token_begin = non_blank_idx
            token_end = torch.cat((token_begin[1:], torch.tensor([len(pred)])))
        tokens = wer_module.decode_ids_to_tokens(pred_ids)
        token_len = (token_end - token_begin).tolist()
        token_begin = token_begin.tolist()
        token_prob = [
            sum(prob_list[i:j]) / (j - i)
            for i, j in zip(non_blank_idx.tolist(), non_blank_idx[1:].tolist() + [len(pred)])
        ]
        if self.word_output:
            words = wer_module.decode_tokens_to_str(pred_ids).split(" ")
            words, word_begin, word_len, word_prob = (
                self._process_tokens_to_words(tokens, token_begin, token_len, token_prob, words)
                if hasattr(self._model, "tokenizer")
                else self._process_char_with_space_to_words(tokens, token_begin, token_len, token_prob, words)
            )
            return s_id, [FrameCtmUnit(t, b, l, p) for t, b, l, p in zip(words, word_begin, word_len, word_prob)]
        return s_id, [FrameCtmUnit(t, b, l, p) for t, b, l, p in zip(tokens, token_begin, token_len, token_prob)]

    def _predict_impl_ctc(
        self,
        encoded: torch.Tensor,
        encoded_len: torch.Tensor,
        transcript: torch.Tensor,
        transcript_len: torch.Tensor,
        sample_id: torch.Tensor,
    ) -> List[Tuple[int, 'FrameCtmUnit']]:
        """Builds time alignment of an encoded sequence.
        This method assumes that the CTC model is used.

        It produces a list of sample ids and fours: (label, start_frame, length, probability), called FrameCtmUnit.
        """
        log_probs = encoded

        if self.prob_suppress_value != 1.0:
            log_probs = self._apply_prob_suppress(log_probs)

        if self.alignment_type == "argmax":
            predictions, probs = self._prepare_ctc_argmax_predictions(log_probs, encoded_len)
        elif self.alignment_type == "forced":
            if self.cpu_decoding:
                log_probs, encoded_len, transcript, transcript_len = (
                    log_probs.cpu(),
                    encoded_len.cpu(),
                    transcript.cpu(),
                    transcript_len.cpu(),
                )
            predictions, probs = self.graph_decoder.align(log_probs, encoded_len, transcript, transcript_len)
        else:
            raise NotImplementedError()

        return [
            self._results_to_ctmUnits(s_id, pred, prob)
            for s_id, pred, prob in zip(sample_id.tolist(), predictions, probs)
        ]

    def _predict_impl_rnnt(
        self,
        encoded: torch.Tensor,
        encoded_len: torch.Tensor,
        transcript: torch.Tensor,
        transcript_len: torch.Tensor,
        sample_id: torch.Tensor,
    ) -> List[Tuple[int, 'FrameCtmUnit']]:
        """Builds time alignment of an encoded sequence.
        This method assumes that the RNNT model is used.

        It produces a list of sample ids and fours: (label, start_frame, length, probability), called FrameCtmUnit.
        """
        if self.alignment_type == "argmax":
            return self._predict_impl_rnnt_argmax(encoded, encoded_len, transcript, transcript_len, sample_id)
        elif self.alignment_type == "forced":
            decoded = self._model.decoder(targets=transcript, target_length=transcript_len)[0]
            log_probs = (
                self._rnnt_joint_pruned(encoded, encoded_len, decoded, transcript_len)
                if self.predictor_window_size > 0 and self.predictor_window_size < transcript_len.max()
                else self._model.joint(encoder_outputs=encoded, decoder_outputs=decoded)
            )
            apply_log_softmax = True if self.log_softmax is None and encoded.is_cuda else self.log_softmax
            if apply_log_softmax:
                log_probs = log_probs.log_softmax(dim=-1)
            if self.cpu_decoding:
                log_probs, encoded_len, transcript, transcript_len = (
                    log_probs.cpu(),
                    encoded_len.cpu(),
                    transcript.cpu(),
                    transcript_len.cpu(),
                )
            predictions, probs = self.graph_decoder.align(log_probs, encoded_len, transcript, transcript_len)
            return [
                self._results_to_ctmUnits(s_id, pred, prob)
                for s_id, pred, prob in zip(sample_id.tolist(), predictions, probs)
            ]
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> List[Tuple[int, 'FrameCtmUnit']]:
        signal, signal_len, transcript, transcript_len, sample_id = batch

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self._model.forward(processed_signal=signal, processed_signal_length=signal_len)[:2]
        else:
            encoded, encoded_len = self._model.forward(input_signal=signal, input_signal_length=signal_len)[:2]

        return self._predict_impl(encoded, encoded_len, transcript, transcript_len, sample_id)

    @torch.no_grad()
    def transcribe(
        self, manifest: List[str], batch_size: int = 4, num_workers: int = None, verbose: bool = True,
    ) -> List['FrameCtmUnit']:
        """
        Does alignment. Use this method for debugging and prototyping.

        Args:

            manifest: path to dataset JSON manifest file (in NeMo format). \
        Recommended length per audio file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
        Bigger will result in better throughput performance but would use more memory.
            num_workers: (int) number of workers for DataLoader
            verbose: (bool) whether to display tqdm progress bar

        Returns:
            A list of four: (label, start_frame, length, probability), called FrameCtmUnit, \
            in the same order as in the manifest.
        """
        hypotheses = []
        # Model's mode and device
        mode = self._model.training
        device = next(self._model.parameters()).device
        dither_value = self._model.preprocessor.featurizer.dither
        pad_to_value = self._model.preprocessor.featurizer.pad_to

        if num_workers is None:
            num_workers = min(batch_size, os.cpu_count() - 1)

        try:
            self._model.preprocessor.featurizer.dither = 0.0
            self._model.preprocessor.featurizer.pad_to = 0

            # Switch model to evaluation mode
            self._model.eval()
            # Freeze the encoder and decoder modules
            self._model.encoder.freeze()
            self._model.decoder.freeze()
            if hasattr(self._model, "joint"):
                self._model.joint.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)

            config = {
                'manifest_filepath': manifest,
                'batch_size': batch_size,
                'num_workers': num_workers,
            }
            temporary_datalayer = self._model._setup_transcribe_dataloader(config)
            for test_batch in tqdm(temporary_datalayer, desc="Aligning", disable=not verbose):
                test_batch[0] = test_batch[0].to(device)
                test_batch[1] = test_batch[1].to(device)
                hypotheses += [unit for i, unit in self.predict_step(test_batch, 0)]
                del test_batch
        finally:
            # set mode back to its original value
            self._model.train(mode=mode)
            self._model.preprocessor.featurizer.dither = dither_value
            self._model.preprocessor.featurizer.pad_to = pad_to_value

            logging.set_verbosity(logging_level)
            if mode is True:
                self._model.encoder.unfreeze()
                self._model.decoder.unfreeze()
                if hasattr(self._model, "joint"):
                    self._model.joint.unfreeze()
        return hypotheses

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        raise RuntimeError("This module cannot be used in training.")

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        raise RuntimeError("This module cannot be used in validation.")

    def setup_test_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        raise RuntimeError("This module cannot be used in testing.")
