# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import gc
import tempfile
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from jiwer import wer as word_error_rate
from omegaconf import DictConfig

from nemo.collections.asr.parts.utils.wfst_utils import TW_BREAK, kaldifst_importer

RIVA_DECODER_INSTALLATION_MESSAGE = (
    "riva decoder is not installed or is installed incorrectly.\n"
    "please run `bash scripts/installers/install_riva_decoder.sh` or `pip install riva-asrlib-decoder` to install."
)


def riva_decoder_importer():
    """Import helper function that returns Riva asrlib decoder package or raises ImportError exception."""
    try:
        import riva.asrlib.decoder.python_decoder as riva_decoder
    except (ImportError, ModuleNotFoundError):
        raise ImportError(RIVA_DECODER_INSTALLATION_MESSAGE)
    return riva_decoder


def _riva_config_to_dict(conf: Any) -> Dict[str, Any]:
    """
    Helper function for parsing Riva configs (namely BatchedMappedDecoderCudaConfig) into a dictionary.

    Args:
      conf:
        Inner Riva config.

    Returns:
      Dictionary corresponding to the Riva config.
    """
    result = {}
    for name in conf.__dir__():
        if not name.startswith("__"):
            attribute = getattr(conf, name)
            result[name] = (
                attribute if attribute.__class__.__module__ == 'builtins' else _riva_config_to_dict(attribute)
            )
    return result


def _fill_inner_riva_config_(riva_conf, nemo_conf):
    """
    Helper function for filling Riva configs (namely BatchedMappedDecoderCudaConfig)
    according to the corresponding NeMo config.

    Note: in-place for the first argument.

    Args:
      riva_conf:
        Inner Riva config.

      nemo_conf:
        Corresponding NeMo config.
    """
    for nemo_k, nemo_v in nemo_conf.items():
        if isinstance(nemo_v, DictConfig):
            _fill_inner_riva_config_(getattr(riva_conf, nemo_k), nemo_v)
        else:
            setattr(riva_conf, nemo_k, nemo_v)


class RivaDecoderConfig(DictConfig):
    """
    NeMo config for the RivaGpuWfstDecoder.
    """

    def __init__(self):
        try:
            riva_decoder = riva_decoder_importer()

            config = riva_decoder.BatchedMappedDecoderCudaConfig()
            config.online_opts.lattice_postprocessor_opts.acoustic_scale = 10.0
            config.n_input_per_chunk = 50
            config.online_opts.decoder_opts.default_beam = 20.0
            config.online_opts.decoder_opts.max_active = 10000
            config.online_opts.determinize_lattice = True
            config.online_opts.max_batch_size = 800
            config.online_opts.num_channels = 800
            config.online_opts.frame_shift_seconds = 1  # not actual frame shift
            config.online_opts.lattice_postprocessor_opts.word_ins_penalty = 0.0

            content = _riva_config_to_dict(config)
        except ImportError:
            content = {}
        super().__init__(content)


class WfstNbestUnit(NamedTuple):
    """
    Container for a single RivaGpuWfstDecoder n-best hypothesis.
    """

    words: Tuple[str]
    timesteps: Tuple[int]
    alignment: Tuple[int]
    score: float


class WfstNbestHypothesis:
    """
    Container for the RivaGpuWfstDecoder n-best results represented as a list of WfstNbestUnit objects.
    """

    def __init__(self, raw_hypotheses: Tuple[Tuple[Tuple[str], Tuple[int], Tuple[int], float]]):
        for i, rh in enumerate(raw_hypotheses):
            assert isinstance(rh[0], tuple), f"{rh[0]}"
            assert isinstance(rh[1], tuple), f"{rh[1]}, {rh[0]}"
            assert isinstance(rh[2], tuple), f"{rh[2]}"
            assert isinstance(rh[3], float), f"{rh[3]}"
            assert len(rh[0]) == len(rh[1]) or len(rh[1]) == 0, "words do not match timesteps"

        self._hypotheses = sorted([WfstNbestUnit(*rh) for rh in raw_hypotheses], key=lambda hyp: hyp.score)
        self._shape0 = len(self._hypotheses)
        self._shape1 = [len(h.words) for h in self._hypotheses]
        self._has_timesteps = len(self._hypotheses[0].timesteps) > 0
        self._has_alignment = len(self._hypotheses[0].alignment) > 0

    def __iter__(self):
        yield from self._hypotheses

    def __getitem__(self, index):
        return self._hypotheses[index]

    def __len__(self):
        return self.shape0

    def replace_unit_(
        self, index: int, new_unit: Union[WfstNbestUnit, Tuple[Tuple[str], Tuple[int], Tuple[int], float]]
    ):
        """
        Replaces a WfstNbestUnit by index.

        Note: in-place operation.

        Args:
          index:
            Index of the unit to be replaced.

          new_unit:
            Replacement unit.
        """
        assert 0 <= index < self.shape0
        assert (
            self.has_timesteps
            and len(new_unit[0]) == len(new_unit[1])
            or not self.has_timesteps
            and len(new_unit[1]) == 0
        )
        assert (
            index == 0
            and (len(self._hypotheses) == 1 or new_unit[3] <= self._hypotheses[index + 1].score)
            or index == self.shape0 - 1
            and self._hypotheses[index - 1].score <= new_unit[3]
            or self._hypotheses[index - 1].score <= new_unit[3] <= self._hypotheses[index + 1].score
        )

        if not isinstance(new_unit, WfstNbestUnit):
            new_unit = WfstNbestUnit(*new_unit)
        self._hypotheses[index] = new_unit
        self._shape1[index] = len(new_unit.words)

    @property
    def shape0(self):
        return self._shape0

    @property
    def shape1(self):
        return self._shape1

    @property
    def has_timesteps(self):
        return self._has_timesteps

    @property
    def has_alignment(self):
        return self._has_alignment


def collapse_tokenword_hypotheses(
    hypotheses: List[WfstNbestHypothesis], tokenword_disambig_str: str
) -> List[WfstNbestHypothesis]:
    """
    Searches for tokenwords in the input hypotheses and collapses them into words.

    Args:
      hypotheses:
        List of input WfstNbestHypothesis.

      tokenword_disambig_str:
        Tokenword disambiguation symbol (e.g. `#1`).

    Returns:
      List of WfstNbestHypothesis.
    """
    new_hypotheses = copy.deepcopy(hypotheses)
    for hyp in new_hypotheses:
        for k, h_unit in enumerate(hyp):
            twds_list = []
            for i, word in enumerate(h_unit.words):
                if word == tokenword_disambig_str:
                    twds_list.append(i)
            if len(twds_list) > 0:
                # a rare case when the recognition stopped before completing the tokenword
                old_words = list(h_unit.words)
                old_timesteps = list(h_unit.timesteps)
                words_len = len(old_words)
                if len(twds_list) % 2 == 1:
                    twds_list.append(words_len)
                new_words, new_timesteps = [], []
                j_prev = 0
                for i, j in zip(twds_list[::2], twds_list[1::2]):
                    new_words += old_words[j_prev:i]
                    # drop tokenword disambig -> remove token disanbig suffix -> remove word begin mark
                    new_word = "".join(old_words[i + 1 : j]).replace(f"{TW_BREAK}{tokenword_disambig_str}", "")[1:]
                    new_words.append(new_word)
                    new_timesteps += old_timesteps[j_prev:i] + [
                        old_timesteps[i],
                    ]
                    j_prev = j + 1
                if j_prev < words_len:
                    new_words += old_words[j_prev:words_len]
                    new_timesteps += old_timesteps[j_prev:words_len]
                hyp.replace_unit_(k, (tuple(new_words), tuple(new_timesteps), h_unit.alignment, h_unit.score))
    return new_hypotheses


class AbstractWFSTDecoder(ABC):
    """
    Used for performing WFST decoding of the logprobs.

    Args:
      lm_fst:
        Language model WFST.

      decoding_mode:
        Decoding mode. E.g. `nbest`.

      beam_size:
        Beam width (float) for the WFST decoding.

      config:
        Decoder config.

      tokenword_disambig_id:
        Tokenword disambiguation index. Set to -1 to disable the tokenword mode.

      lm_weight:
        Language model weight in decoding.
    """

    def __init__(
        self,
        lm_fst: Any,
        decoding_mode: str,
        beam_size: float,
        config: Optional[Any],
        tokenword_disambig_id: int = -1,
        lm_weight: float = 1.0,
    ):
        self._lm_fst = lm_fst
        self._beam_size = beam_size
        self._tokenword_disambig_id = tokenword_disambig_id
        self._open_vocabulary_decoding = self._tokenword_disambig_id >= 0
        self._lm_weight = lm_weight
        self._id2word, self._word2id = None, None
        self._id2token, self._token2id = None, None
        self._decoding_mode, self._config, self._decoder = None, None, None

        self._set_decoding_mode(decoding_mode)
        self._set_decoder_config(config)
        self._init_decoder()

    @abstractmethod
    def _set_decoder_config(self, config: Optional[Any] = None):
        pass

    @abstractmethod
    def _set_decoding_mode(self, decoding_mode: str):
        pass

    @abstractmethod
    def _init_decoder(self):
        pass

    @property
    def decoding_mode(self):
        return self._decoding_mode

    @decoding_mode.setter
    def decoding_mode(self, value: str):
        self._decoding_mode_setter(value)

    @abstractmethod
    def _decoding_mode_setter(self, value: str):
        pass

    @property
    def beam_size(self):
        return self._beam_size

    @beam_size.setter
    def beam_size(self, value: float):
        self._beam_size_setter(value)

    @abstractmethod
    def _beam_size_setter(self, value: float):
        pass

    @property
    def lm_weight(self):
        return self._lm_weight

    @lm_weight.setter
    def lm_weight(self, value: float):
        self._lm_weight_setter(value)

    @abstractmethod
    def _lm_weight_setter(self, value: float):
        pass

    @property
    def tokenword_disambig_id(self):
        return self._tokenword_disambig_id

    @property
    def open_vocabulary_decoding(self):
        return self._open_vocabulary_decoding

    @abstractmethod
    def decode(self, log_probs: torch.Tensor, log_probs_length: torch.Tensor) -> List[Any]:
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
        pass

    @abstractmethod
    def _post_decode(self, hypotheses: List[Any]) -> List[Any]:
        """
        Does various post-processing of the recognition hypotheses.

        Args:
          hypotheses:
            List of recognition hypotheses.

        Returns:
          List of processed recognition hypotheses.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass


class RivaGpuWfstDecoder(AbstractWFSTDecoder):
    """
    Used for performing WFST decoding of the logprobs with the Riva WFST decoder.

    Args:
      lm_fst:
        Kaldi-type language model WFST or its path.

      decoding_mode:
        Decoding mode. Choices: `nbest`, `mbr`, `lattice`.

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
    """

    def __init__(
        self,
        lm_fst: Union['kaldifst.StdFst', Path, str],
        decoding_mode: str = 'mbr',
        beam_size: float = 10.0,
        config: Optional['RivaDecoderConfig'] = None,
        tokenword_disambig_id: int = -1,
        lm_weight: float = 1.0,
        nbest_size: int = 1,
    ):
        self._nbest_size = nbest_size
        self._load_word_lattice = None
        super().__init__(lm_fst, decoding_mode, beam_size, config, tokenword_disambig_id, lm_weight)

    def _set_decoder_config(self, config: Optional['RivaDecoderConfig'] = None):
        if config is None or len(config) == 0:
            config = RivaDecoderConfig()
        if not hasattr(config, "online_opts"):
            # most likely empty config
            # call importer to raise the exception + installation message
            riva_decoder_importer()
            # just in case
            raise RuntimeError("Unexpected config error. Please debug manually.")
        config.online_opts.decoder_opts.lattice_beam = self._beam_size
        config.online_opts.lattice_postprocessor_opts.lm_scale = (
            self._lm_weight * config.online_opts.lattice_postprocessor_opts.acoustic_scale
        )
        config.online_opts.lattice_postprocessor_opts.nbest = self._nbest_size
        self._config = config

    def _init_decoder(self):

        # use importers instead of direct import to possibly get an installation message
        kaldifst = kaldifst_importer()
        riva_decoder = riva_decoder_importer()

        from nemo.collections.asr.parts.utils.wfst_utils import load_word_lattice

        self._load_word_lattice = load_word_lattice
        # BatchedMappedDecoderCuda supports filepaths only
        # TODO: fix when possible
        lm_fst = self._lm_fst
        tmp_fst = None
        tmp_fst_file = None
        if isinstance(lm_fst, (Path, str)):
            # We only read lm_fst to extract words.txt and num_tokens_with_blank
            tmp_fst = kaldifst.StdVectorFst.read(lm_fst)
        elif isinstance(lm_fst, (kaldifst.StdVectorFst, kaldifst.StdConstFst)):
            tmp_fst = lm_fst
            tmp_fst_file = tempfile.NamedTemporaryFile(mode='w+t')
            tmp_fst.write(tmp_fst_file.name)
            lm_fst = tmp_fst_file.name
        else:
            raise ValueError(f"Unsupported lm_fst type: {type(lm_fst)}")

        # we assume that lm_fst has at least one disambig after real tokens
        num_tokens_with_blank = tmp_fst.input_symbols.find('#0') - 1
        if self._id2word is None:
            self._id2word = {
                int(line.split("\t")[1]): line.split("\t")[0]
                for line in str(tmp_fst.output_symbols).strip().split("\n")
            }
            word2id = self._id2word.__class__(map(reversed, self._id2word.items()))
            word_unk_id = word2id["<unk>"]
            self._word2id = defaultdict(lambda: word_unk_id)
            for k, v in word2id.items():
                self._word2id[k] = v
        if self._id2token is None:
            self._id2token = {
                int(line.split("\t")[1]): line.split("\t")[0]
                for line in str(tmp_fst.input_symbols).strip().split("\n")
            }
            token2id = self._id2token.__class__(map(reversed, self._id2token.items()))
            token_unk_id = token2id["<unk>"]
            self._token2id = defaultdict(lambda: token_unk_id)
            for k, v in token2id.items():
                self._token2id[k] = v
        with tempfile.NamedTemporaryFile(mode='w+t') as words_tmp:
            tmp_fst.output_symbols.write_text(words_tmp.name)
            config = riva_decoder.BatchedMappedDecoderCudaConfig()
            _fill_inner_riva_config_(config, self._config)
            self._decoder = riva_decoder.BatchedMappedDecoderCuda(
                config, lm_fst, words_tmp.name, num_tokens_with_blank
            )
        if tmp_fst_file:
            tmp_fst_file.close()

    def _set_decoding_mode(self, decoding_mode: str):
        if decoding_mode == 'nbest':
            self._decode = self._decode_nbest
        elif decoding_mode == 'mbr':
            self._decode = self._decode_mbr
        elif decoding_mode == 'lattice':
            self._decode = self._decode_lattice
        else:
            raise ValueError(f"Unsupported mode: {decoding_mode}")
        self._decoding_mode = decoding_mode

    def _beam_size_setter(self, value: float):
        if self._beam_size != value:
            self._release_gpu_memory()
            self._config.online_opts.decoder_opts.lattice_beam = value
            self._init_decoder()
            self._beam_size = value

    def _lm_weight_setter(self, value: float):
        if self._lm_weight != value:
            self._release_gpu_memory()
            self._config.online_opts.lattice_postprocessor_opts.lm_scale = (
                value * self._config.online_opts.lattice_postprocessor_opts.acoustic_scale
            )
            self._init_decoder()
            self._lm_weight = value

    def _decoding_mode_setter(self, value: str):
        if self._decoding_mode != value:
            self._set_decoding_mode(value)

    @property
    def nbest_size(self):
        return self._nbest_size

    @nbest_size.setter
    def nbest_size(self, value: float):
        self._nbest_size_setter(value)

    def _nbest_size_setter(self, value: float):
        if self._nbest_size != value:
            self._release_gpu_memory()
            self._config.online_opts.lattice_postprocessor_opts.nbest = value
            self._init_decoder()
            self._nbest_size = value

    def _decode_nbest(
        self, log_probs: torch.Tensor, log_probs_length: torch.Tensor
    ) -> List[WfstNbestHypothesis]:  # words, timesteps, alignment, score
        """
        Decodes logprobs into recognition hypotheses via the N-best decoding decoding.

        Args:
          log_probs:
            A torch.Tensor of the predicted log-probabilities of shape [Batch, Time, Vocabulary].

          log_probs_length:
            A torch.Tensor of length `Batch` which contains the lengths of the log_probs elements.

        Returns:
          List of WfstNbestHypothesis with empty alignment and trivial score.
        """
        hypotheses_nbest = self._decoder.decode_nbest(log_probs, log_probs_length)
        hypotheses = []
        for nh in hypotheses_nbest:
            nbest_container = []
            for h in nh:
                words, timesteps = [], []
                for w, t in zip(h.words, h.word_start_times_seconds):
                    if w != 0:
                        words.append(self._id2word[w])
                        timesteps.append(int(t))
                alignment = [ilabel - 1 for ilabel in h.ilabels]
                score = h.score
                nbest_container.append(tuple([tuple(words), tuple(timesteps), tuple(alignment), score]))
            hypotheses.append(WfstNbestHypothesis(tuple(nbest_container)))
        return hypotheses

    def _decode_mbr(self, log_probs: torch.Tensor, log_probs_length: torch.Tensor) -> List[WfstNbestHypothesis]:
        """
        Decodes logprobs into recognition hypotheses via the Minimum Bayes Risk (MBR) decoding.

        Args:
          log_probs:
            A torch.Tensor of the predicted log-probabilities of shape [Batch, Time, Vocabulary].

          log_probs_length:
            A torch.Tensor of length `Batch` which contains the lengths of the log_probs elements.

        Returns:
          List of WfstNbestHypothesis with empty alignment and trivial score.
        """
        hypotheses_mbr = self._decoder.decode_mbr(log_probs, log_probs_length)
        hypotheses = []
        for h in hypotheses_mbr:
            words, timesteps = [], []
            for e in h:
                words.append(e[0])
                timesteps.append(int(e[1]))
            hypotheses.append(WfstNbestHypothesis(tuple([tuple([tuple(words), tuple(timesteps), tuple(), 0.0])])))
        return hypotheses

    def _decode_lattice(self, log_probs: torch.Tensor, log_probs_length: torch.Tensor) -> List['KaldiWordLattice']:
        """
        Decodes logprobs into kaldi-type lattices.

        Args:
          log_probs:
            A torch.Tensor of the predicted log-probabilities of shape [Batch, Time, Vocabulary].

          log_probs_length:
            A torch.Tensor of length `Batch` which contains the lengths of the log_probs elements.

        Returns:
          List of KaldiWordLattice.
        """
        with tempfile.NamedTemporaryFile() as tmp_lat:
            tmp_lat_name = f"{tmp_lat.name}.lats"
            self._decoder.decode_write_lattice(
                log_probs, log_probs_length, [str(i) for i in range(len(log_probs))], f"ark,t:{tmp_lat_name}"
            )
            hypotheses_lattice = self._load_word_lattice(
                tmp_lat_name, self._id2word, self._id2word
            )  # input and output token ids are the same
            hypotheses = [hypotheses_lattice[str(i)] for i in range(len(log_probs))]
        return hypotheses

    def decode(
        self, log_probs: torch.Tensor, log_probs_length: torch.Tensor
    ) -> Union[List[WfstNbestHypothesis], List['KaldiWordLattice']]:
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
        log_probs = log_probs.contiguous()
        log_probs_length = log_probs_length.to(torch.long).to('cpu').contiguous()
        hypotheses = self._decode(log_probs, log_probs_length)
        hypotheses = self._post_decode(hypotheses)
        return hypotheses

    def _post_decode(
        self, hypotheses: Union[List[WfstNbestHypothesis], List['KaldiWordLattice']]
    ) -> Union[List[WfstNbestHypothesis], List['KaldiWordLattice']]:
        """
        Does various post-processing of the recognition hypotheses.

        Args:
          hypotheses:
            List of recognition hypotheses.

        Returns:
          List of processed recognition hypotheses.
        """
        if self._open_vocabulary_decoding and self._decoding_mode in ('nbest', 'mbr'):
            return collapse_tokenword_hypotheses(hypotheses, self._id2word[self._tokenword_disambig_id])
        else:
            return hypotheses

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
        self.decoding_mode = "mbr"
        best_lm_weight, best_wer = -1.0, float('inf')
        for lm_weight in range(1, 21):  # enough for most cases
            self.lm_weight = lm_weight / 10
            hypotheses = self.decode(log_probs, log_probs_length)
            wer = word_error_rate([" ".join(h[0].words) for h in hypotheses], reference_texts)
            print(lm_weight, wer)
            if wer < best_wer:
                best_lm_weight, best_wer = self.lm_weight, wer
        self.decoding_mode = decoding_mode_backup
        self.lm_weight = lm_weight_backup
        return best_lm_weight, best_wer

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
        decoding_mode_backup = self.decoding_mode
        self.decoding_mode = "lattice"
        lattices = self.decode(log_probs, log_probs_length)
        scores, counts, wer_per_utt = [], [], []
        for lattice, text in zip(lattices, reference_texts):
            word_ids = [self._word2id[w] for w in text.strip().split()]
            counts.append(len(word_ids) if word_ids else 1)
            scores.append(lattice.edit_distance(word_ids))
            wer_per_utt.append(scores[-1] / counts[-1])
        self.decoding_mode = decoding_mode_backup
        return sum(scores) / sum(counts), wer_per_utt

    def _release_gpu_memory(self):
        """
        Forces freeing of GPU memory by deleting the Riva decoder object.
        """
        try:
            del self._decoder
        except Exception:
            # apparently self._decoder was previously deleted, do nothing
            pass
        gc.collect()

    def __del__(self):
        self._release_gpu_memory()
