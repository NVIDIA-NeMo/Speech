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

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import HypothesisType, LengthsType, LogprobsType, NeuralType
from nemo.utils import logging

DEFAULT_TOKEN_OFFSET = 100


def pack_hypotheses(
    hypotheses: List[rnnt_utils.NBestHypotheses], logitlen: torch.Tensor,
) -> List[rnnt_utils.NBestHypotheses]:

    if logitlen is not None:
        if hasattr(logitlen, 'cpu'):
            logitlen_cpu = logitlen.to('cpu')
        else:
            logitlen_cpu = logitlen

    for idx, hyp in enumerate(hypotheses):  # type: rnnt_utils.NBestHypotheses
        for candidate_idx, cand in enumerate(hyp.n_best_hypotheses):
            cand.y_sequence = torch.tensor(cand.y_sequence, dtype=torch.long)

            if logitlen is not None:
                cand.length = logitlen_cpu[idx]

            if cand.dec_state is not None:
                cand.dec_state = _states_to_device(cand.dec_state)

    return hypotheses


def _states_to_device(dec_state, device='cpu'):
    if torch.is_tensor(dec_state):
        dec_state = dec_state.to(device)

    elif isinstance(dec_state, (list, tuple)):
        dec_state = tuple(_states_to_device(dec_i, device) for dec_i in dec_state)

    return dec_state


class AbstractBeamCTCInfer(Typing):
    """A beam CTC decoder.

    Provides a common abstraction for sample level beam decoding.

    Args:
        blank_id: int, index of the blank token. Can be 0 or len(vocabulary).
        beam_size: int, size of the beam used in the underlying beam search engine.

    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "decoder_output": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "decoder_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"predictions": [NeuralType(elements_type=HypothesisType())]}

    def __init__(self, blank_id: int, beam_size: int):
        self.blank_id = blank_id

        if beam_size < 1:
            raise ValueError("Beam search size cannot be less than 1!")

        self.beam_size = beam_size

        # Variables set by corresponding setter methods
        self.vocab = None
        self.decoding_type = None
        self.tokenizer = None

        # Utility maps for vocabulary
        self.vocab_index_map = None
        self.index_vocab_map = None

        # Internal variable, used to prevent double reduction of consecutive tokens (ctc collapse)
        self.override_fold_consecutive_value = None

    def set_vocabulary(self, vocab: List[str]):
        """
        Set the vocabulary of the decoding framework.

        Args:
            vocab: List of str. Each token corresponds to its location in the vocabulary emitted by the model.
                Note that this vocabulary must NOT contain the "BLANK" token.
        """
        self.vocab = vocab
        self.vocab_index_map = {v: i for i, v in enumerate(vocab)}
        self.index_vocab_map = {i: v for i, v in enumerate(vocab)}

    def set_decoding_type(self, decoding_type: str):
        """
        Sets the decoding type of the framework. Can support either char or subword models.

        Args:
            decoding_type: Str corresponding to decoding type. Only supports "char" and "subword".
        """
        decoding_type = decoding_type.lower()
        supported_types = ['char', 'subword']

        if decoding_type not in supported_types:
            raise ValueError(
                f"Unsupported decoding type. Supported types = {supported_types}.\n" f"Given = {decoding_type}"
            )

        self.decoding_type = decoding_type

    def set_tokenizer(self, tokenizer: TokenizerSpec):
        """
        Set the tokenizer of the decoding framework.

        Args:
            tokenizer: NeMo tokenizer object, which inherits from TokenizerSpec.
        """
        self.tokenizer = tokenizer

    @typecheck()
    def forward(
        self, decoder_output: torch.Tensor, decoder_lengths: torch.Tensor,
    ) -> Tuple[List[Union[rnnt_utils.Hypothesis, rnnt_utils.NBestHypotheses]]]:
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-repressively.

        Args:
            decoder_output: A tensor of size (batch, timesteps, features) or (batch, timesteps) (each timestep is a label).
            decoder_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class BeamCTCInfer(AbstractBeamCTCInfer):
    """A greedy CTC decoder.

    Provides a common abstraction for sample level and batch level greedy decoding.

    Args:
        blank_index: int index of the blank token. Can be 0 or len(vocabulary).
        preserve_alignments: Bool flag which preserves the history of logprobs generated during
            decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `logprobs` in it. Here, `logprobs` is a torch.Tensors.
        compute_timestamps: A bool flag, which determines whether to compute the character/subword, or
                word based timestamp mapping the output log-probabilities to discrite intervals of timestamps.
                The timestamps will be available in the returned Hypothesis.timestep as a dictionary.

    """

    def __init__(
        self,
        blank_id: int,
        beam_size: int,
        search_type: str = "default",
        return_best_hypothesis: bool = True,
        preserve_alignments: bool = False,
        compute_timestamps: bool = False,
        beam_alpha: float = 1.0,
        beam_beta: float = 0.0,
        kenlm_path: str = None,
        flashlight_cfg: Optional['FlashlightConfig'] = None,
        pyctcdecode_cfg: Optional['PyCTCDecodeConfig'] = None,
    ):
        super().__init__(blank_id=blank_id, beam_size=beam_size)

        self.search_type = search_type
        self.return_best_hypothesis = return_best_hypothesis
        self.preserve_alignments = preserve_alignments
        self.compute_timestamps = compute_timestamps

        if self.compute_timestamps:
            raise ValueError(f"Currently this flag is not supported for beam search algorithms.")

        self.vocab = None  # This must be set by specific method by user before calling forward() !

        if search_type == "default" or search_type == "nemo":
            self.search_algorithm = self.default_beam_search
        elif search_type == "pyctcdecode":
            self.search_algorithm = self._pyctcdecode_beam_search
        elif search_type == "flashlight":
            self.search_algorithm = self.flashlight_beam_search
        else:
            raise NotImplementedError(
                f"The search type ({search_type}) supplied is not supported!\n"
                f"Please use one of : (default, nemo, pyctcdecode)"
            )

        # Log the beam search algorithm
        logging.info(f"Beam search algorithm: {search_type}")

        self.beam_alpha = beam_alpha
        self.beam_beta = beam_beta

        # Default beam search args
        self.kenlm_path = kenlm_path

        # PyCTCDecode params
        if pyctcdecode_cfg is None:
            pyctcdecode_cfg = PyCTCDecodeConfig()
        self.pyctcdecode_cfg = pyctcdecode_cfg  # type: PyCTCDecodeConfig

        if flashlight_cfg is None:
            flashlight_cfg = FlashlightConfig()
        self.flashlight_cfg = flashlight_cfg

        # Default beam search scorer functions
        self.default_beam_scorer = None
        self.pyctcdecode_beam_scorer = None
        self.flashlight_beam_scorer = None
        self.token_offset = 0

    @typecheck()
    def forward(
        self, decoder_output: torch.Tensor, decoder_lengths: torch.Tensor,
    ) -> Tuple[List[Union[rnnt_utils.Hypothesis, rnnt_utils.NBestHypotheses]]]:
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-repressively.

        Args:
            decoder_output: A tensor of size (batch, timesteps, features).
            decoder_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        if self.vocab is None:
            raise RuntimeError("Please set the vocabulary with `set_vocabulary()` before calling this function.")

        if self.decoding_type is None:
            raise ValueError("Please set the decoding type with `set_decoding_type()` before calling this function.")

        with torch.no_grad(), torch.inference_mode():
            # Process each sequence independently
            prediction_tensor = decoder_output

            if prediction_tensor.ndim != 3:
                raise ValueError(
                    f"`decoder_output` must be a tensor of shape [B, T, V] (log probs, float). "
                    f"Provided shape = {prediction_tensor.shape}"
                )

            # determine type of input - logprobs or labels
            out_len = decoder_lengths if decoder_lengths is not None else None
            hypotheses = self.search_algorithm(prediction_tensor, out_len)

            # Pack results into Hypotheses
            packed_result = pack_hypotheses(hypotheses, decoder_lengths)

            # Pack the result
            if self.return_best_hypothesis and isinstance(packed_result[0], rnnt_utils.NBestHypotheses):
                packed_result = [res.n_best_hypotheses[0] for res in packed_result]  # type: Hypothesis

        return (packed_result,)

    @torch.no_grad()
    def default_beam_search(
        self, x: torch.Tensor, out_len: torch.Tensor
    ) -> List[Union[rnnt_utils.Hypothesis, rnnt_utils.NBestHypotheses]]:
        """
        Open Seq2Seq Beam Search Algorithm (DeepSpeed)

        Args:
            x: Tensor of shape [B, T, V+1], where B is the batch size, T is the maximum sequence length,
                and V is the vocabulary size. The tensor contains log-probabilities.
            out_len: Tensor of shape [B], contains lengths of each sequence in the batch.

        Returns:
            A list of NBestHypotheses objects, one for each sequence in the batch.
        """
        if self.compute_timestamps:
            raise ValueError(
                f"Beam Search with strategy `{self.search_type}` does not support time stamp calculation!"
            )

        if self.default_beam_scorer is None:
            # Check for filepath
            if self.kenlm_path is None or not os.path.exists(self.kenlm_path):
                raise FileNotFoundError(
                    f"KenLM binary file not found at : {self.kenlm_path}. "
                    f"Please set a valid path in the decoding config."
                )

            # perform token offset for subword models
            if self.decoding_type == 'subword':
                vocab = [chr(idx + self.token_offset) for idx in range(len(self.vocab))]
            else:
                # char models
                vocab = self.vocab

            # Must import at runtime to avoid circular dependency due to module level import.
            from nemo.collections.asr.modules.beam_search_decoder import BeamSearchDecoderWithLM

            self.default_beam_scorer = BeamSearchDecoderWithLM(
                vocab=vocab,
                lm_path=self.kenlm_path,
                beam_width=self.beam_size,
                alpha=self.beam_alpha,
                beta=self.beam_beta,
                num_cpus=max(1, os.cpu_count()),
                input_tensor=False,
            )

        x = x.to('cpu')

        with typecheck.disable_checks():
            data = [x[sample_id, : out_len[sample_id], :].softmax(dim=-1) for sample_id in range(len(x))]
            beams_batch = self.default_beam_scorer.forward(log_probs=data, log_probs_length=None)

        # For each sample in the batch
        nbest_hypotheses = []
        for beams_idx, beams in enumerate(beams_batch):
            # For each beam candidate / hypothesis in each sample
            hypotheses = []
            for candidate_idx, candidate in enumerate(beams):
                hypothesis = rnnt_utils.Hypothesis(
                    score=0.0, y_sequence=[], dec_state=None, timestep=[], last_token=None
                )

                # For subword encoding, NeMo will double encode the subword (multiple tokens) into a
                # singular unicode id. In doing so, we preserve the semantic of the unicode token, and
                # compress the size of the final KenLM ARPA / Binary file.
                # In order to do double encoding, we shift the subword by some token offset.
                # This step is ignored for character based models.
                if self.decoding_type == 'subword':
                    pred_token_ids = [ord(c) - self.token_offset for c in candidate[1]]
                else:
                    # Char models
                    pred_token_ids = [self.vocab_index_map[c] for c in candidate[1]]

                # We preserve the token ids and the score for this hypothesis
                hypothesis.y_sequence = pred_token_ids
                hypothesis.score = candidate[0]

                # If alignment must be preserved, we preserve a view of the output logprobs.
                # Note this view is shared amongst all beams within the sample, be sure to clone it if you
                # require specific processing for each sample in the beam.
                # This is done to preserve memory.
                if self.preserve_alignments:
                    hypothesis.alignments = x[beams_idx][: out_len[beams_idx]]

                hypotheses.append(hypothesis)

            # Wrap the result in NBestHypothesis.
            hypotheses = rnnt_utils.NBestHypotheses(hypotheses)
            nbest_hypotheses.append(hypotheses)

        return nbest_hypotheses

    @torch.no_grad()
    def _pyctcdecode_beam_search(
        self, x: torch.Tensor, out_len: torch.Tensor
    ) -> List[Union[rnnt_utils.Hypothesis, rnnt_utils.NBestHypotheses]]:
        """
        PyCTCDecode Beam Search Algorithm. Should support Char and Subword models.

        Args:
            x: Tensor of shape [B, T, V+1], where B is the batch size, T is the maximum sequence length,
                and V is the vocabulary size. The tensor contains log-probabilities.
            out_len: Tensor of shape [B], contains lengths of each sequence in the batch.

        Returns:
            A list of NBestHypotheses objects, one for each sequence in the batch.
        """
        if self.compute_timestamps:
            raise ValueError(
                f"Beam Search with strategy `{self.search_type}` does not support time stamp calculation!"
            )

        try:
            import pyctcdecode
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                f"Could not load `pyctcdecode` library. Please install it from pip using :\n"
                f"pip install --upgrade pyctcdecode"
            )

        if self.pyctcdecode_beam_scorer is None:
            self.pyctcdecode_beam_scorer = pyctcdecode.build_ctcdecoder(
                labels=self.vocab, kenlm_model_path=self.kenlm_path, alpha=self.beam_alpha, beta=self.beam_beta
            )  # type: pyctcdecode.BeamSearchDecoderCTC

        x = x.to('cpu').numpy()

        with typecheck.disable_checks():
            beams_batch = []
            for sample_id in range(len(x)):
                logprobs = x[sample_id, : out_len[sample_id], :]
                result = self.pyctcdecode_beam_scorer.decode_beams(
                    logprobs,
                    beam_width=self.beam_size,
                    beam_prune_logp=self.pyctcdecode_cfg.beam_prune_logp,
                    token_min_logp=self.pyctcdecode_cfg.token_min_logp,
                    prune_history=self.pyctcdecode_cfg.prune_history,
                    hotwords=self.pyctcdecode_cfg.hotwords,
                    hotword_weight=self.pyctcdecode_cfg.hotword_weight,
                    lm_start_state=None,
                )  # Output format: text, last_lm_state, text_frames, logit_score, lm_score
                beams_batch.append(result)

        nbest_hypotheses = []
        for beams_idx, beams in enumerate(beams_batch):
            hypotheses = []
            for candidate_idx, candidate in enumerate(beams):
                # Candidate = (text, last_lm_state, text_frames, logit_score, lm_score)
                hypothesis = rnnt_utils.Hypothesis(
                    score=0.0, y_sequence=[], dec_state=None, timestep=[], last_token=None
                )

                # TODO: Requires token ids to be returned rather than text.
                if self.decoding_type == 'subword':
                    if self.tokenizer is None:
                        raise ValueError("Tokenizer must be provided for subword decoding. Use set_tokenizer().")

                    pred_token_ids = self.tokenizer.text_to_ids(candidate[0])
                else:
                    if self.vocab is None:
                        raise ValueError("Vocab must be provided for character decoding. Use set_vocab().")

                    chars = list(candidate[0])
                    pred_token_ids = [self.vocab_index_map[c] for c in chars]

                hypothesis.y_sequence = pred_token_ids
                hypothesis.text = candidate[0]  # text
                hypothesis.score = candidate[4]  # score

                # Inject word level timestamps
                hypothesis.timestep = candidate[2]  # text_frames

                if self.preserve_alignments:
                    hypothesis.alignments = torch.from_numpy(x[beams_idx][: out_len[beams_idx]])

                hypotheses.append(hypothesis)

            hypotheses = rnnt_utils.NBestHypotheses(hypotheses)
            nbest_hypotheses.append(hypotheses)

        return nbest_hypotheses

    @torch.no_grad()
    def flashlight_beam_search(
        self, x: torch.Tensor, out_len: torch.Tensor
    ) -> List[Union[rnnt_utils.Hypothesis, rnnt_utils.NBestHypotheses]]:
        """
        Flashlight Beam Search Algorithm. Should support Char and Subword models.

        Args:
            x: Tensor of shape [B, T, V+1], where B is the batch size, T is the maximum sequence length,
                and V is the vocabulary size. The tensor contains log-probabilities.
            out_len: Tensor of shape [B], contains lengths of each sequence in the batch.

        Returns:
            A list of NBestHypotheses objects, one for each sequence in the batch.
        """
        if self.compute_timestamps:
            raise ValueError(
                f"Beam Search with strategy `{self.search_type}` does not support time stamp calculation!"
            )

        if self.flashlight_beam_scorer is None:
            # Check for filepath
            if self.kenlm_path is None or not os.path.exists(self.kenlm_path):
                raise FileNotFoundError(
                    f"KenLM binary file not found at : {self.kenlm_path}. "
                    f"Please set a valid path in the decoding config."
                )

            # perform token offset for subword models
            # if self.decoding_type == 'subword':
            #    vocab = [chr(idx + self.token_offset) for idx in range(len(self.vocab))]
            # else:
            #    # char models
            #    vocab = self.vocab

            # Must import at runtime to avoid circular dependency due to module level import.
            from nemo.collections.asr.modules.flashlight_decoder import FlashLightKenLMBeamSearchDecoder

            self.flashlight_beam_scorer = FlashLightKenLMBeamSearchDecoder(
                lm_path=self.kenlm_path,
                vocabulary=self.vocab,
                tokenizer=self.tokenizer,
                lexicon_path=self.flashlight_cfg.lexicon_path,
                boost_path=self.flashlight_cfg.boost_path,
                beam_size=self.beam_size,
                beam_size_token=self.flashlight_cfg.beam_size_token,
                beam_threshold=self.flashlight_cfg.beam_threshold,
                lm_weight=self.beam_alpha,
                word_score=self.beam_beta,
                unk_weight=self.flashlight_cfg.unk_weight,
                sil_weight=self.flashlight_cfg.sil_weight,
            )

        x = x.to('cpu')

        with typecheck.disable_checks():
            beams_batch = self.flashlight_beam_scorer.forward(log_probs=x)

        # For each sample in the batch
        nbest_hypotheses = []
        for beams_idx, beams in enumerate(beams_batch):
            # For each beam candidate / hypothesis in each sample
            hypotheses = []
            for candidate_idx, candidate in enumerate(beams):
                hypothesis = rnnt_utils.Hypothesis(
                    score=0.0, y_sequence=[], dec_state=None, timestep=[], last_token=None
                )

                # We preserve the token ids and the score for this hypothesis
                hypothesis.y_sequence = candidate['tokens'].tolist()
                hypothesis.score = candidate['score']

                # If alignment must be preserved, we preserve a view of the output logprobs.
                # Note this view is shared amongst all beams within the sample, be sure to clone it if you
                # require specific processing for each sample in the beam.
                # This is done to preserve memory.
                if self.preserve_alignments:
                    hypothesis.alignments = x[beams_idx][: out_len[beams_idx]]

                hypotheses.append(hypothesis)

            # Wrap the result in NBestHypothesis.
            hypotheses = rnnt_utils.NBestHypotheses(hypotheses)
            nbest_hypotheses.append(hypotheses)

        return nbest_hypotheses

    def set_decoding_type(self, decoding_type: str):
        super().set_decoding_type(decoding_type)

        # Please check train_kenlm.py in scripts/asr_language_modeling/ to find out why we need
        # TOKEN_OFFSET for BPE-based models
        if self.decoding_type == 'subword':
            self.token_offset = DEFAULT_TOKEN_OFFSET


@dataclass
class PyCTCDecodeConfig:
    # These arguments cannot be imported from pyctcdecode (optional dependency)
    # Therefore we copy the values explicitly
    # Taken from pyctcdecode.constant
    beam_prune_logp: float = -10.0
    token_min_logp: float = -5.0
    prune_history: bool = False
    hotwords: Optional[List[str]] = None
    hotword_weight: float = 10.0


@dataclass
class FlashlightConfig:
    lexicon_path: Optional[str] = None
    boost_path: Optional[str] = None
    beam_size_token: int = 16
    beam_threshold: float = 20.0
    unk_weight: float = -math.inf
    sil_weight: float = 0.0


@dataclass
class BeamCTCInferConfig:
    beam_size: int
    search_type: str = 'default'
    preserve_alignments: bool = False
    compute_timestamps: bool = False
    return_best_hypothesis: bool = True

    beam_alpha: float = 1.0
    beam_beta: float = 0.0
    kenlm_path: Optional[str] = None

    flashlight_cfg: Optional[FlashlightConfig] = FlashlightConfig()
    pyctcdecode_cfg: Optional[PyCTCDecodeConfig] = PyCTCDecodeConfig()
