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
import math
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import torch
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import WER, CTCDecoding, CTCDecodingConfig
from nemo.collections.asr.metrics.wer_bpe import WERBPE, CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from nemo.collections.asr.parts.utils.audio_utils import get_samples
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map, get_uniqname_from_filepath
from nemo.collections.asr.parts.utils.streaming_utils import AudioFeatureIterator, FrameBatchASR
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging

__all__ = ['ASRDecoderTimeStamps']

try:
    from pyctcdecode import build_ctcdecoder

    PYCTCDECODE = True
except ImportError:
    PYCTCDECODE = False


def if_none_get_default(param, default_value):
    return (param, default_value)[param is None]


class WERBPE_TS(WERBPE):
    """
    This is WERBPE_TS class that is modified for generating word_timestamps with logits.
    The functions in WER class is modified to save the word_timestamps whenever BPE token
    is being saved into a list.
    This class is designed to support ASR models based on CTC and BPE.
    Please refer to the definition of WERBPE class for more information.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        batch_dim_index=0,
        use_cer=False,
        ctc_decode=None,
        log_prediction=True,
        dist_sync_on_step=False,
    ):
        if ctc_decode is not None:
            logging.warning(f'`ctc_decode` was set to {ctc_decode}. Note that this is ignored.')

        decoding_cfg = CTCBPEDecodingConfig(batch_dim_index=batch_dim_index)
        decoding = CTCBPEDecoding(decoding_cfg, tokenizer=tokenizer)
        super().__init__(decoding, use_cer, log_prediction, dist_sync_on_step)

    def ctc_decoder_predictions_tensor_with_ts(
        self, time_stride, predictions: torch.Tensor, predictions_len: torch.Tensor = None
    ) -> List[str]:
        hypotheses, timestamps, word_timestamps = [], [], []
        # '⁇' string should be removed since it causes error during string split.
        unk = '⁇'
        prediction_cpu_tensor = predictions.long().cpu()
        # iterate over batch
        self.time_stride = time_stride
        for ind in range(prediction_cpu_tensor.shape[self.decoding.batch_dim_index]):
            prediction = prediction_cpu_tensor[ind].detach().numpy().tolist()
            if predictions_len is not None:
                prediction = prediction[: predictions_len[ind]]
            # CTC decoding procedure
            decoded_prediction, char_ts, timestamp_list = [], [], []
            previous = self.decoding.blank_id
            for pdx, p in enumerate(prediction):
                if (p != previous or previous == self.decoding.blank_id) and p != self.decoding.blank_id:
                    decoded_prediction.append(p)
                    char_ts.append(round(pdx * self.time_stride, 2))
                    timestamp_list.append(round(pdx * self.time_stride, 2))

                previous = p

            hypothesis = self.decode_tokens_to_str_with_ts(decoded_prediction)
            hypothesis = hypothesis.replace(unk, '')
            word_ts = self.get_ts_from_decoded_prediction(decoded_prediction, hypothesis, char_ts)

            hypotheses.append(hypothesis)
            timestamps.append(timestamp_list)
            word_timestamps.append(word_ts)
        return hypotheses, timestamps, word_timestamps

    def decode_tokens_to_str_with_ts(self, tokens: List[int]) -> str:
        hypothesis = self.decoding.tokenizer.ids_to_text(tokens)
        return hypothesis

    def decode_ids_to_tokens_with_ts(self, tokens: List[int]) -> List[str]:
        token_list = self.decoding.tokenizer.ids_to_tokens(tokens)
        return token_list

    def get_ts_from_decoded_prediction(self, decoded_prediction: List[str], hypothesis: List[str], char_ts: List[str]):
        decoded_char_list = self.decoding.tokenizer.ids_to_tokens(decoded_prediction)
        stt_idx, end_idx = 0, len(decoded_char_list) - 1
        stt_ch_idx, end_ch_idx = 0, 0
        space = '▁'
        word_ts, word_seq = [], []
        word_open_flag = False
        for idx, ch in enumerate(decoded_char_list):

            # If the symbol is space and not an end of the utterance, move on
            if idx != end_idx and (space == ch and space in decoded_char_list[idx + 1]):
                continue
            # If the symbol is unkown symbol such as '<unk>' symbol, move on
            elif ch in ['<unk>']:
                continue

            if (idx == stt_idx or space == decoded_char_list[idx - 1] or (space in ch and len(ch) > 1)) and (
                ch != space
            ):
                _stt = char_ts[idx]
                stt_ch_idx = idx
                word_open_flag = True

            if word_open_flag and ch != space and (idx == end_idx or space in decoded_char_list[idx + 1]):
                _end = round(char_ts[idx] + self.time_stride, 2)
                end_ch_idx = idx
                word_open_flag = False
                word_ts.append([_stt, _end])
                stitched_word = ''.join(decoded_char_list[stt_ch_idx : end_ch_idx + 1]).replace(space, '')
                word_seq.append(stitched_word)
        assert len(word_ts) == len(hypothesis.split()), "Text hypothesis does not match word timestamps."
        return word_ts


class WER_TS(WER):
    """
    This is WER class that is modified for generating timestamps with logits.
    The functions in WER class is modified to save the timestamps whenever character
    is being saved into a list.
    This class is designed to support ASR models based on CTC and Character-level tokens.
    Please refer to the definition of WER class for more information.
    """

    def __init__(
        self,
        vocabulary,
        batch_dim_index=0,
        use_cer=False,
        ctc_decode=None,
        log_prediction=True,
        dist_sync_on_step=False,
    ):
        if ctc_decode is not None:
            logging.warning(f'`ctc_decode` was set to {ctc_decode}. Note that this is ignored.')

        decoding_cfg = CTCDecodingConfig(batch_dim_index=batch_dim_index)
        decoding = CTCDecoding(decoding_cfg, vocabulary=vocabulary)
        super().__init__(decoding, use_cer, log_prediction, dist_sync_on_step)

    def decode_tokens_to_str_with_ts(self, tokens: List[int], timestamps: List[int]) -> str:
        """
        Take frame-level tokens and timestamp list and collect the timestamps for
        start and end of each word.
        """
        token_list, timestamp_list = self.decode_ids_to_tokens_with_ts(tokens, timestamps)
        hypothesis = ''.join(self.decoding.decode_ids_to_tokens(tokens))
        return hypothesis, timestamp_list

    def decode_ids_to_tokens_with_ts(self, tokens: List[int], timestamps: List[int]) -> List[str]:
        token_list, timestamp_list = [], []
        for i, c in enumerate(tokens):
            if c != self.decoding.blank_id:
                token_list.append(self.decoding.labels_map[c])
                timestamp_list.append(timestamps[i])
        return token_list, timestamp_list

    def ctc_decoder_predictions_tensor_with_ts(
        self, predictions: torch.Tensor, predictions_len: torch.Tensor = None,
    ) -> List[str]:
        """
        A shortened version of the original function ctc_decoder_predictions_tensor().
        Replaced decode_tokens_to_str() function with decode_tokens_to_str_with_ts().
        """
        hypotheses, timestamps = [], []
        prediction_cpu_tensor = predictions.long().cpu()
        for ind in range(prediction_cpu_tensor.shape[self.decoding.batch_dim_index]):
            prediction = prediction_cpu_tensor[ind].detach().numpy().tolist()
            if predictions_len is not None:
                prediction = prediction[: predictions_len[ind]]

            # CTC decoding procedure with timestamps
            decoded_prediction, decoded_timing_list = [], []
            previous = self.decoding.blank_id
            for pdx, p in enumerate(prediction):
                if (p != previous or previous == self.decoding.blank_id) and p != self.decoding.blank_id:
                    decoded_prediction.append(p)
                    decoded_timing_list.append(pdx)
                previous = p

            text, timestamp_list = self.decode_tokens_to_str_with_ts(decoded_prediction, decoded_timing_list)
            hypotheses.append(text)
            timestamps.append(timestamp_list)

        return hypotheses, timestamps


def get_wer_feat_logit(audio_file_path, asr, frame_len, tokens_per_chunk, delay, model_stride_in_secs):
    """
    Create a preprocessor to convert audio samples into raw features,
    Normalization will be done per buffer in frame_bufferer.
    """
    asr.reset()
    asr.read_audio_file_and_return(audio_file_path, delay, model_stride_in_secs)
    hyp, tokens, log_prob = asr.transcribe_with_ts(tokens_per_chunk, delay)
    return hyp, tokens, log_prob


class FrameBatchASR_Logits(FrameBatchASR):
    """
    A class for streaming frame-based ASR.
    Inherits from FrameBatchASR and adds new capability of returning the logit output.
    Please refer to FrameBatchASR for more detailed information.
    """

    def __init__(
        self,
        asr_model: Type[EncDecCTCModelBPE],
        frame_len: float = 1.6,
        total_buffer: float = 4.0,
        batch_size: int = 4,
    ):
        super().__init__(asr_model, frame_len, total_buffer, batch_size)
        self.all_logprobs = []

    def clear_buffer(self):
        self.all_logprobs = []
        self.all_preds = []

    def read_audio_file_and_return(self, audio_filepath: str, delay: float, model_stride_in_secs: float):
        samples = get_samples(audio_filepath)
        samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
        frame_reader = AudioFeatureIterator(samples, self.frame_len, self.raw_preprocessor, self.asr_model.device)
        self.set_frame_reader(frame_reader)

    @torch.no_grad()
    def _get_batch_preds(self):
        device = self.asr_model.device
        for batch in iter(self.data_loader):

            feat_signal, feat_signal_len = batch
            feat_signal, feat_signal_len = feat_signal.to(device), feat_signal_len.to(device)
            log_probs, encoded_len, predictions = self.asr_model(
                processed_signal=feat_signal, processed_signal_length=feat_signal_len
            )
            preds = torch.unbind(predictions)
            for pred in preds:
                self.all_preds.append(pred.cpu().numpy())
            log_probs_tup = torch.unbind(log_probs)
            for log_prob in log_probs_tup:
                self.all_logprobs.append(log_prob)
            del encoded_len
            del predictions

    def transcribe_with_ts(
        self, tokens_per_chunk: int, delay: int,
    ):
        self.infer_logits()
        self.unmerged = []
        self.part_logprobs = []
        for idx, pred in enumerate(self.all_preds):
            decoded = pred.tolist()
            _stt, _end = len(decoded) - 1 - delay, len(decoded) - 1 - delay + tokens_per_chunk
            self.unmerged += decoded[len(decoded) - 1 - delay : len(decoded) - 1 - delay + tokens_per_chunk]
            self.part_logprobs.append(self.all_logprobs[idx][_stt:_end, :])
        self.unmerged_logprobs = torch.cat(self.part_logprobs, 0)
        assert (
            len(self.unmerged) == self.unmerged_logprobs.shape[0]
        ), "Unmerged decoded result and log prob lengths are different."
        return self.greedy_merge(self.unmerged), self.unmerged, self.unmerged_logprobs


class ASRDecoderTimeStamps:
    """
    A class designed for extracting word timestamps while the ASR decoding process.
    This class contains a few setups for a slew of NeMo ASR models such as QuartzNet, CitriNet and ConformerCTC models.
    """

    def __init__(self, cfg_diarizer):
        self.manifest_filepath = cfg_diarizer.manifest_filepath
        self.params = cfg_diarizer.asr.parameters
        self.ctc_decoder_params = cfg_diarizer.asr.ctc_decoder_parameters
        self.ASR_model_name = cfg_diarizer.asr.model_path
        self.nonspeech_threshold = self.params.asr_based_vad_threshold
        self.root_path = None
        self.run_ASR = None
        self.encdec_class = None
        self.AUDIO_RTTM_MAP = audio_rttm_map(self.manifest_filepath)
        self.audio_file_list = [value['audio_filepath'] for _, value in self.AUDIO_RTTM_MAP.items()]

    def set_asr_model(self):
        """
        Initialize the parameters for the given ASR model.
        Currently, the following NGC models are supported:

            stt_en_quartznet15x5,
            stt_en_citrinet*,
            stt_en_conformer_ctc*

        To assign a proper decoding function for generating timestamp output,
        the name of .nemo file should include the architecture name such as:
        'quartznet', 'conformer', and 'citrinet'.

        decoder_delay_in_sec is the amount of delay that is compensated during the word timestamp extraction.
        word_ts_anchor_offset is the reference point for a word and used for matching the word with diarization labels.
        Each ASR model has a different optimal decoder delay and word timestamp anchor offset.
        To obtain an optimized diarization result with ASR, decoder_delay_in_sec and word_ts_anchor_offset
        need to be searched on a development set.
        """
        if 'quartznet' in self.ASR_model_name.lower():
            self.run_ASR = self.run_ASR_QuartzNet_CTC
            self.encdec_class = EncDecCTCModel
            self.decoder_delay_in_sec = if_none_get_default(self.params['decoder_delay_in_sec'], 0.04)
            self.word_ts_anchor_offset = if_none_get_default(self.params['word_ts_anchor_offset'], 0.12)
            self.asr_batch_size = if_none_get_default(self.params['asr_batch_size'], 4)
            self.model_stride_in_secs = 0.02

        elif 'conformer' in self.ASR_model_name.lower():
            self.run_ASR = self.run_ASR_BPE_CTC
            self.encdec_class = EncDecCTCModelBPE
            self.decoder_delay_in_sec = if_none_get_default(self.params['decoder_delay_in_sec'], 0.08)
            self.word_ts_anchor_offset = if_none_get_default(self.params['word_ts_anchor_offset'], 0.12)
            self.asr_batch_size = if_none_get_default(self.params['asr_batch_size'], 16)
            self.model_stride_in_secs = 0.04
            # Conformer requires buffered inference and the parameters for buffered processing.
            self.chunk_len_in_sec = 5
            self.total_buffer_in_secs = 25

        elif 'citrinet' in self.ASR_model_name.lower():
            self.run_ASR = self.run_ASR_CitriNet_CTC
            self.encdec_class = EncDecCTCModelBPE
            self.decoder_delay_in_sec = if_none_get_default(self.params['decoder_delay_in_sec'], 0.16)
            self.word_ts_anchor_offset = if_none_get_default(self.params['word_ts_anchor_offset'], 0.2)
            self.asr_batch_size = if_none_get_default(self.params['asr_batch_size'], 4)
            self.model_stride_in_secs = 0.08

        else:
            raise ValueError(f"Cannot find the ASR model class for: {self.params['self.ASR_model_name']}")

        if self.ASR_model_name.endswith('.nemo'):
            asr_model = self.encdec_class.restore_from(restore_path=self.ASR_model_name)
        else:
            asr_model = self.encdec_class.from_pretrained(model_name=self.ASR_model_name, strict=False)

        if self.ctc_decoder_params['pretrained_language_model']:
            if not PYCTCDECODE:
                raise ImportError(
                    'LM for beam search decoding is provided but pyctcdecode is not installed. Install pyctcdecode using PyPI: pip install pyctcdecode'
                )
            self.beam_search_decoder = self.load_LM_for_CTC_decoder(asr_model)
        else:
            self.beam_search_decoder = None

        asr_model.eval()
        return asr_model

    def load_LM_for_CTC_decoder(self, asr_model: Type[Union[EncDecCTCModel, EncDecCTCModelBPE]]):
        """
        Load a language model for CTC decoder (pyctcdecode).
        Note that only EncDecCTCModel and EncDecCTCModelBPE models can use pyctcdecode.
        """
        kenlm_model = self.ctc_decoder_params['pretrained_language_model']
        logging.info(f"Loading language model : {self.ctc_decoder_params['pretrained_language_model']}")

        if 'EncDecCTCModelBPE' in str(type(asr_model)):
            vocab = asr_model.tokenizer.tokenizer.get_vocab()
            labels = list(vocab.keys())
            labels[0] = "<unk>"
        elif 'EncDecCTCModel' in str(type(asr_model)):
            labels = asr_model.decoder.vocabulary
        else:
            raise ValueError(f"Cannot find a vocabulary or tokenizer for: {self.params['self.ASR_model_name']}")

        decoder = build_ctcdecoder(
            labels, kenlm_model, alpha=self.ctc_decoder_params['alpha'], beta=self.ctc_decoder_params['beta']
        )
        return decoder

    def run_ASR_QuartzNet_CTC(self, asr_model: Type[EncDecCTCModel]) -> Tuple[Dict, Dict]:
        """
        Launch QuartzNet ASR model and collect logit, timestamps and text output.

        Args:
            asr_model (class):
                The loaded NeMo ASR model.

        Returns:
            words_dict (dict):
                Dictionary containing the sequence of words from hypothesis.
            word_ts_dict (dict):
                Dictionary containing the time-stamps of words.
        """
        words_dict, word_ts_dict = {}, {}

        wer_ts = WER_TS(
            vocabulary=asr_model.decoder.vocabulary,
            batch_dim_index=0,
            use_cer=asr_model._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=asr_model._cfg.get("log_prediction", False),
        )

        with torch.cuda.amp.autocast():
            transcript_logits_list = asr_model.transcribe(
                self.audio_file_list, batch_size=self.asr_batch_size, logprobs=True
            )
            for idx, logit_np in enumerate(transcript_logits_list):
                uniq_id = get_uniqname_from_filepath(self.audio_file_list[idx])
                if self.beam_search_decoder:
                    logging.info(
                        f"Running beam-search decoder on {uniq_id} with LM {self.ctc_decoder_params['pretrained_language_model']}"
                    )
                    hyp_words, word_ts = self.run_pyctcdecode(logit_np)
                else:
                    log_prob = torch.from_numpy(logit_np)
                    logits_len = torch.from_numpy(np.array([log_prob.shape[0]]))
                    greedy_predictions = log_prob.argmax(dim=-1, keepdim=False).unsqueeze(0)
                    text, char_ts = wer_ts.ctc_decoder_predictions_tensor_with_ts(
                        greedy_predictions, predictions_len=logits_len
                    )
                    trans, char_ts_in_feature_frame_idx = self.clean_trans_and_TS(text[0], char_ts[0])
                    spaces_in_sec, hyp_words = self._get_spaces(
                        trans, char_ts_in_feature_frame_idx, self.model_stride_in_secs
                    )
                    word_ts = self.get_word_ts_from_spaces(
                        char_ts_in_feature_frame_idx, spaces_in_sec, end_stamp=logit_np.shape[0]
                    )
                word_ts = self.align_decoder_delay(word_ts, self.decoder_delay_in_sec)
                assert len(hyp_words) == len(word_ts), "Words and word timestamp list length does not match."
                words_dict[uniq_id] = hyp_words
                word_ts_dict[uniq_id] = word_ts

        return words_dict, word_ts_dict

    @staticmethod
    def clean_trans_and_TS(trans: str, char_ts: List[str]) -> Tuple[str, List[str]]:
        """
        Remove the spaces in the beginning and the end.
        The char_ts need to be changed and synced accordingly.

        Args:
            trans (list):
                List containing the character output (str).
            char_ts (list):
                List containing the timestamps (int) for each character.

        Returns:
            trans (list):
                List containing the cleaned character output.
            char_ts (list):
                List containing the cleaned timestamps for each character.
        """
        assert (len(trans) > 0) and (len(char_ts) > 0)
        assert len(trans) == len(char_ts)

        trans = trans.lstrip()
        diff_L = len(char_ts) - len(trans)
        char_ts = char_ts[diff_L:]

        trans = trans.rstrip()
        diff_R = len(char_ts) - len(trans)
        if diff_R > 0:
            char_ts = char_ts[: -1 * diff_R]
        return trans, char_ts

    def _get_spaces(self, trans: str, char_ts: List[str], time_stride: float) -> Tuple[float, List[str]]:
        """
        Collect the space symbols with a list of words.

        Args:
            trans (list):
                List containing the character output (str).
            char_ts (list):
                List containing the timestamps of the characters.
            time_stride (float):
                The size of stride of the model in second.

        Returns:
            spaces_in_sec (list):
                List containing the ranges of spaces
            word_list (list):
                List containing the words from ASR inference.
        """
        blank = ' '
        spaces_in_sec, word_list = [], []
        stt_idx = 0
        assert (len(trans) > 0) and (len(char_ts) > 0), "Transcript and char_ts length should not be 0."
        assert len(trans) == len(char_ts), "Transcript and timestamp lengths do not match."

        # If there is a blank, update the time stamps of the space and the word.
        for k, s in enumerate(trans):
            if s == blank:
                spaces_in_sec.append(
                    [round(char_ts[k] * time_stride, 2), round((char_ts[k + 1] - 1) * time_stride, 2)]
                )
                word_list.append(trans[stt_idx:k])
                stt_idx = k + 1

        # Add the last word
        if len(trans) > stt_idx and trans[stt_idx] != blank:
            word_list.append(trans[stt_idx:])
        return spaces_in_sec, word_list

    def run_ASR_CitriNet_CTC(self, asr_model: Type[EncDecCTCModelBPE]) -> Tuple[Dict, Dict]:
        """
        Launch CitriNet ASR model and collect logit, timestamps and text output.

        Args:
            asr_model (class):
                The loaded NeMo ASR model.

        Returns:
            words_dict (dict):
                Dictionary containing the sequence of words from hypothesis.
            word_ts_dict (dict):
                Dictionary containing the timestamps of hypothesis words.
        """
        words_dict, word_ts_dict = {}, {}

        werbpe_ts = WERBPE_TS(
            tokenizer=asr_model.tokenizer,
            batch_dim_index=0,
            use_cer=asr_model._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=asr_model._cfg.get("log_prediction", False),
        )

        with torch.cuda.amp.autocast():
            transcript_logits_list = asr_model.transcribe(
                self.audio_file_list, batch_size=self.asr_batch_size, logprobs=True
            )
            for idx, logit_np in enumerate(transcript_logits_list):
                uniq_id = get_uniqname_from_filepath(self.audio_file_list[idx])
                if self.beam_search_decoder:
                    logging.info(
                        f"Running beam-search decoder with LM {self.ctc_decoder_params['pretrained_language_model']}"
                    )
                    hyp_words, word_ts = self.run_pyctcdecode(logit_np)
                else:
                    log_prob = torch.from_numpy(logit_np)
                    greedy_predictions = log_prob.argmax(dim=-1, keepdim=False).unsqueeze(0)
                    logits_len = torch.from_numpy(np.array([log_prob.shape[0]]))
                    text, char_ts, word_ts = werbpe_ts.ctc_decoder_predictions_tensor_with_ts(
                        self.model_stride_in_secs, greedy_predictions, predictions_len=logits_len
                    )
                    hyp_words, word_ts = text[0].split(), word_ts[0]
                word_ts = self.align_decoder_delay(word_ts, self.decoder_delay_in_sec)
                assert len(hyp_words) == len(word_ts), "Words and word timestamp list length does not match."
                words_dict[uniq_id] = hyp_words
                word_ts_dict[uniq_id] = word_ts

        return words_dict, word_ts_dict

    def set_buffered_infer_params(self, asr_model: Type[EncDecCTCModelBPE]) -> Tuple[float, float, float]:
        """
        Prepare the parameters for the buffered inference.
        """
        cfg = copy.deepcopy(asr_model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)

        # some changes for streaming scenario
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"

        preprocessor = nemo_asr.models.EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        preprocessor.to(asr_model.device)

        # Disable config overwriting
        OmegaConf.set_struct(cfg.preprocessor, True)

        onset_delay = (
            math.ceil(((self.total_buffer_in_secs - self.chunk_len_in_sec) / 2) / self.model_stride_in_secs) + 1
        )
        mid_delay = math.ceil(
            (self.chunk_len_in_sec + (self.total_buffer_in_secs - self.chunk_len_in_sec) / 2)
            / self.model_stride_in_secs
        )
        tokens_per_chunk = math.ceil(self.chunk_len_in_sec / self.model_stride_in_secs)

        return onset_delay, mid_delay, tokens_per_chunk

    def run_ASR_BPE_CTC(self, asr_model: Type[EncDecCTCModelBPE]) -> Tuple[Dict, Dict]:
        """
        Launch CTC-BPE based ASR model and collect logit, timestamps and text output.

        Args:
            asr_model (class):
                The loaded NeMo ASR model.

        Returns:
            words_dict (dict):
                Dictionary containing the sequence of words from hypothesis.
            word_ts_dict (dict):
                Dictionary containing the time-stamps of words.
        """
        torch.manual_seed(0)
        torch.set_grad_enabled(False)
        words_dict, word_ts_dict = {}, {}

        werbpe_ts = WERBPE_TS(
            tokenizer=asr_model.tokenizer,
            batch_dim_index=0,
            use_cer=asr_model._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=asr_model._cfg.get("log_prediction", False),
        )

        frame_asr = FrameBatchASR_Logits(
            asr_model=asr_model,
            frame_len=self.chunk_len_in_sec,
            total_buffer=self.total_buffer_in_secs,
            batch_size=self.asr_batch_size,
        )

        onset_delay, mid_delay, tokens_per_chunk = self.set_buffered_infer_params(asr_model)
        onset_delay_in_sec = round(onset_delay * self.model_stride_in_secs, 2)

        with torch.cuda.amp.autocast():
            logging.info(f"Running ASR model {self.ASR_model_name}")

            for idx, audio_file_path in enumerate(self.audio_file_list):
                uniq_id = get_uniqname_from_filepath(audio_file_path)
                logging.info(f"[{idx+1}/{len(self.audio_file_list)}] FrameBatchASR: {audio_file_path}")
                frame_asr.clear_buffer()

                hyp, greedy_predictions_list, log_prob = get_wer_feat_logit(
                    audio_file_path,
                    frame_asr,
                    self.chunk_len_in_sec,
                    tokens_per_chunk,
                    mid_delay,
                    self.model_stride_in_secs,
                )
                if self.beam_search_decoder:
                    logging.info(
                        f"Running beam-search decoder with LM {self.ctc_decoder_params['pretrained_language_model']}"
                    )
                    log_prob = log_prob.unsqueeze(0).cpu().numpy()[0]
                    hyp_words, word_ts = self.run_pyctcdecode(log_prob, onset_delay_in_sec=onset_delay_in_sec)
                else:
                    logits_len = torch.from_numpy(np.array([len(greedy_predictions_list)]))
                    greedy_predictions_list = greedy_predictions_list[onset_delay:]
                    greedy_predictions = torch.from_numpy(np.array(greedy_predictions_list)).unsqueeze(0)
                    text, char_ts, word_ts = werbpe_ts.ctc_decoder_predictions_tensor_with_ts(
                        self.model_stride_in_secs, greedy_predictions, predictions_len=logits_len
                    )
                    hyp_words, word_ts = text[0].split(), word_ts[0]

                word_ts = self.align_decoder_delay(word_ts, self.decoder_delay_in_sec)
                assert len(hyp_words) == len(word_ts), "Words and word timestamp list length does not match."
                words_dict[uniq_id] = hyp_words
                word_ts_dict[uniq_id] = word_ts

        return words_dict, word_ts_dict

    def get_word_ts_from_spaces(self, char_ts: List[float], spaces_in_sec: List[float], end_stamp: float) -> List[str]:
        """
        Take word timestamps from the spaces from the decoded prediction.

        Args:
            char_ts (list):
                List containing the timestamp for each character.
            spaces_in_sec (list):
                List containing the start and the end time of each space token.
            end_stamp (float):
                The end time of the session in sec.

        Returns:
            word_timestamps (list):
                List containing the timestamps for the resulting words.
        """
        end_stamp = min(end_stamp, (char_ts[-1] + 2))
        start_stamp_in_sec = round(char_ts[0] * self.model_stride_in_secs, 2)
        end_stamp_in_sec = round(end_stamp * self.model_stride_in_secs, 2)

        # In case of one word output with no space information.
        if len(spaces_in_sec) == 0:
            word_timestamps = [[start_stamp_in_sec, end_stamp_in_sec]]
        elif len(spaces_in_sec) > 0:
            # word_timetamps_middle should be an empty list if len(spaces_in_sec) == 1.
            word_timetamps_middle = [
                [round(spaces_in_sec[k][1], 2), round(spaces_in_sec[k + 1][0], 2),]
                for k in range(len(spaces_in_sec) - 1)
            ]
            word_timestamps = (
                [[start_stamp_in_sec, round(spaces_in_sec[0][0], 2)]]
                + word_timetamps_middle
                + [[round(spaces_in_sec[-1][1], 2), end_stamp_in_sec]]
            )
        return word_timestamps

    def run_pyctcdecode(
        self, logprob: np.ndarray, onset_delay_in_sec: float = 0, beam_width: int = 32
    ) -> Tuple[List[str], List[str]]:
        """
        Launch pyctcdecode with the loaded pretrained language model.

        Args:
            logprob (np.ndarray):
                The log probability from the ASR model inference in numpy array format.
            onset_delay_in_sec (float):
                The amount of delay that needs to be compensated for the timestamp outputs froM pyctcdecode.
            beam_width (int):
                The beam width parameter for beam search decodring.
        Returns:
            hyp_words (list):
                List containing the words in the hypothesis.
            word_ts (list):
                List containing the word timestamps from the decoder.
        """
        beams = self.beam_search_decoder.decode_beams(logprob, beam_width=self.ctc_decoder_params['beam_width'])
        word_ts_beam, words_beam = [], []
        for idx, (word, _) in enumerate(beams[0][2]):
            ts = self.get_word_ts_from_wordframes(idx, beams[0][2], self.model_stride_in_secs, onset_delay_in_sec)
            word_ts_beam.append(ts)
            words_beam.append(word)
        hyp_words, word_ts = words_beam, word_ts_beam
        return hyp_words, word_ts

    @staticmethod
    def get_word_ts_from_wordframes(
        idx, word_frames: List[List[float]], frame_duration: float, onset_delay: float, word_block_delay: float = 2.25
    ):
        """
        Extract word timestamps from word frames generated from pyctcdecode.
        """
        offset = -1 * word_block_delay * frame_duration - onset_delay
        frame_begin = word_frames[idx][1][0]
        if frame_begin == -1:
            frame_begin = word_frames[idx - 1][1][1] if idx != 0 else 0
        frame_end = word_frames[idx][1][1]
        return [
            round(max(frame_begin * frame_duration + offset, 0), 2),
            round(max(frame_end * frame_duration + offset, 0), 2),
        ]

    @staticmethod
    def align_decoder_delay(word_ts, decoder_delay_in_sec: float):
        """
        Subtract decoder_delay_in_sec from the word timestamp output.
        """
        for k in range(len(word_ts)):
            word_ts[k] = [
                round(word_ts[k][0] - decoder_delay_in_sec, 2),
                round(word_ts[k][1] - decoder_delay_in_sec, 2),
            ]
        return word_ts
