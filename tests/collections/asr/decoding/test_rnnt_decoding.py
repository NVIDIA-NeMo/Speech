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

import copy
import os
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Optional

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.modules import RNNTDecoder, RNNTJoint
from nemo.collections.asr.parts.context_biasing import BoostingTreeModelConfig, GPUBoostingTreeModel
from nemo.collections.asr.parts.mixins import mixins
from nemo.collections.asr.parts.submodules import rnnt_beam_decoding
from nemo.collections.asr.parts.submodules import rnnt_greedy_decoding as greedy_decode
from nemo.collections.asr.parts.submodules import tdt_beam_decoding
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding, RNNTDecoding, RNNTDecodingConfig
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.core.utils import numba_utils
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__
from tests.collections.asr.decoding.test_timestamps import BaseTimestampsTest

NUMBA_RNNT_LOSS_AVAILABLE = numba_utils.numba_cpu_is_supported(
    __NUMBA_MINIMUM_VERSION__
) or numba_utils.numba_cuda_is_supported(__NUMBA_MINIMUM_VERSION__)


def char_vocabulary():
    return [' ', 'a', 'b', 'c', 'd', 'e', 'f', '.']


@pytest.fixture()
@lru_cache(maxsize=8)
def tmp_tokenizer(test_data_dir):
    cfg = DictConfig({'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_wpe_128"), 'type': 'wpe'})

    class _TmpASRBPE(mixins.ASRBPEMixin):
        def register_artifact(self, _, vocab_path):
            return vocab_path

    asrbpe = _TmpASRBPE()
    asrbpe._setup_tokenizer(cfg)
    return asrbpe.tokenizer


@lru_cache(maxsize=2)
def get_rnnt_decoder(vocab_size, decoder_output_size=4):
    prednet_cfg = {'pred_hidden': decoder_output_size, 'pred_rnn_layers': 1}
    torch.manual_seed(0)
    decoder = RNNTDecoder(prednet=prednet_cfg, vocab_size=vocab_size)
    decoder.freeze()
    return decoder


@lru_cache(maxsize=2)
def get_rnnt_joint(vocab_size, vocabulary=None, encoder_output_size=4, decoder_output_size=4, joint_output_shape=4):
    jointnet_cfg = {
        'encoder_hidden': encoder_output_size,
        'pred_hidden': decoder_output_size,
        'joint_hidden': joint_output_shape,
        'activation': 'relu',
    }
    torch.manual_seed(0)
    joint = RNNTJoint(jointnet_cfg, vocab_size, vocabulary=vocabulary)
    joint.freeze()
    return joint


@lru_cache(maxsize=1)
def get_model_encoder_output(data_dir, model_name):
    # Import inside function to avoid issues with dependencies
    import librosa

    audio_filepath = os.path.join(data_dir, 'asr', 'test', 'an4', 'wav', 'cen3-fjlp-b.wav')

    with torch.no_grad():
        model = ASRModel.from_pretrained(model_name, map_location='cpu')  # type: ASRModel
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
        model.eval()

        audio, sr = librosa.load(path=audio_filepath, sr=16000, mono=True)

        input_signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        input_signal_length = torch.tensor([len(audio)], dtype=torch.int32)

        encoded, encoded_len = model(input_signal=input_signal, input_signal_length=input_signal_length)

    return model, encoded, encoded_len


def decode_text_from_greedy_hypotheses(hyps, decoding):
    decoded_hyps = decoding.decode_hypothesis(hyps)  # type: List[str]

    return decoded_hyps


def decode_text_from_nbest_hypotheses(hyps, decoding):
    hypotheses = []
    all_hypotheses = []

    for nbest_hyp in hyps:  # type: rnnt_utils.NBestHypotheses
        n_hyps = nbest_hyp.n_best_hypotheses  # Extract all hypotheses for this sample
        decoded_hyps = decoding.decode_hypothesis(n_hyps)  # type: List[str]

        hypotheses.append(decoded_hyps[0])  # best hypothesis
        all_hypotheses.append(decoded_hyps)

    return hypotheses, all_hypotheses


def check_beam_decoding(test_data_dir, beam_config):
    beam_size = beam_config.pop("beam_size", 1)
    model, encoded, encoded_len = get_model_encoder_output(test_data_dir, 'nvidia/parakeet-tdt_ctc-110m')

    model_config = model.to_config_dict()
    durations = list(model_config["model_defaults"]["tdt_durations"])

    beam = tdt_beam_decoding.BeamTDTInfer(
        model.decoder,
        model.joint,
        beam_size=beam_size,
        return_best_hypothesis=False,
        durations=durations,
        **beam_config,
    )

    enc_out = encoded
    enc_len = encoded_len

    with torch.no_grad():
        hyps: rnnt_utils.Hypothesis = beam(encoder_output=enc_out, encoded_lengths=enc_len)[0]
        _, all_hyps = decode_text_from_nbest_hypotheses(hyps, model.decoding)
        all_hyps = all_hyps[0]

        print("Beam search algorithm :", beam_config['search_type'])
        for idx, hyp_ in enumerate(all_hyps):
            print("Hyp index", idx + 1, "text :", hyp_.text)

            assert len(hyp_.timestamp) > 0
            print("Timesteps", hyp_.timestamp)
            print()


def check_tdt_greedy_decoding(
    test_data_dir,
    use_cuda_graph_decoder: bool,
    lm_path: Optional[str | Path] = None,
    boosting_tree: Optional[BoostingTreeModelConfig] = None,
):
    model, encoded, encoded_len = get_model_encoder_output(test_data_dir, 'nvidia/parakeet-tdt_ctc-110m')

    model_config = model.to_config_dict()

    fusion_models, fusion_models_alpha = None, None
    if lm_path or boosting_tree:
        fusion_models = []
        fusion_models_alpha = []

    if lm_path:
        fusion_models.append(NGramGPULanguageModel.from_file(lm_path=lm_path, vocab_size=model.decoder.blank_idx))
        fusion_models_alpha.append(0.5)
    if boosting_tree:
        fusion_models.append(GPUBoostingTreeModel.from_config(boosting_tree, tokenizer=model.tokenizer))
        fusion_models_alpha.append(0.5)

    decoding_algo = greedy_decode.GreedyBatchedTDTInfer(
        model.decoder,
        model.joint,
        blank_index=model.decoder.blank_idx,
        durations=list(model_config["model_defaults"]["tdt_durations"]),
        max_symbols_per_step=10,
        preserve_alignments=False,
        preserve_frame_confidence=False,
        use_cuda_graph_decoder=use_cuda_graph_decoder,
        fusion_models=fusion_models,
        fusion_models_alpha=fusion_models_alpha,
    )

    enc_out = encoded
    enc_len = encoded_len

    with torch.no_grad():
        hyps: rnnt_utils.Hypothesis = decoding_algo(encoder_output=enc_out, encoded_lengths=enc_len)[0]
        all_hyps = decode_text_from_greedy_hypotheses(hyps, model.decoding)

        print("Decoding result")
        for idx, hyp_ in enumerate(all_hyps):
            print(f"Hyp index {idx + 1} | text : {hyp_.text}")
            assert len(hyp_.timestamp) > 0
            print("Timesteps", hyp_.timestamp)
            print()


class TestRNNTDecoding:
    @pytest.mark.unit
    def test_constructor(self):
        cfg = RNNTDecodingConfig()
        vocab = char_vocabulary()
        decoder = get_rnnt_decoder(vocab_size=len(vocab))
        joint = get_rnnt_joint(vocab_size=len(vocab))
        decoding = RNNTDecoding(decoding_cfg=cfg, decoder=decoder, joint=joint, vocabulary=vocab)
        assert decoding is not None

    @pytest.mark.unit
    def test_constructor_subword(self, tmp_tokenizer):
        cfg = RNNTDecodingConfig()
        vocab = tmp_tokenizer.vocab
        decoder = get_rnnt_decoder(vocab_size=len(vocab))
        joint = get_rnnt_joint(vocab_size=len(vocab))
        decoding = RNNTBPEDecoding(decoding_cfg=cfg, decoder=decoder, joint=joint, tokenizer=tmp_tokenizer)
        assert decoding is not None

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    def test_greedy_decoding_preserve_alignments(self, test_data_dir):
        model, encoded, encoded_len = get_model_encoder_output(test_data_dir, 'stt_en_conformer_transducer_small')

        beam = greedy_decode.GreedyRNNTInfer(
            model.decoder,
            model.joint,
            blank_index=model.joint.num_classes_with_blank - 1,
            max_symbols_per_step=5,
            preserve_alignments=True,
        )

        enc_out = encoded
        enc_len = encoded_len

        with torch.no_grad():
            hyps = beam(encoder_output=enc_out, encoded_lengths=enc_len)[0]  # type: rnnt_utils.Hypothesis
            hyp = decode_text_from_greedy_hypotheses(hyps, model.decoding)
            hyp = hyp[0]

            assert hyp.alignments is not None

            # Use the following commented print statements to check
            # the alignment of other algorithms compared to the default
            print("Text", hyp.text)
            for t in range(len(hyp.alignments)):
                t_u = []
                for u in range(len(hyp.alignments[t])):
                    logp, label = hyp.alignments[t][u]
                    assert torch.is_tensor(logp)
                    assert torch.is_tensor(label)

                    t_u.append(int(label))

                print(f"Tokens at timestamp {t} = {t_u}")
            print()

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize("loop_labels", [True, False])
    def test_batched_greedy_decoding_preserve_alignments(self, test_data_dir, loop_labels: bool):
        """Test batched greedy decoding using non-batched decoding as a reference"""
        model, encoded, encoded_len = get_model_encoder_output(test_data_dir, 'stt_en_conformer_transducer_small')

        search_algo = greedy_decode.GreedyBatchedRNNTInfer(
            model.decoder,
            model.joint,
            blank_index=model.joint.num_classes_with_blank - 1,
            max_symbols_per_step=5,
            preserve_alignments=True,
            loop_labels=loop_labels,
        )

        etalon_search_algo = greedy_decode.GreedyRNNTInfer(
            model.decoder,
            model.joint,
            blank_index=model.joint.num_classes_with_blank - 1,
            max_symbols_per_step=5,
            preserve_alignments=True,
        )

        enc_out = encoded
        enc_len = encoded_len

        with torch.no_grad():
            hyps: list[rnnt_utils.Hypothesis] = search_algo(encoder_output=enc_out, encoded_lengths=enc_len)[0]
            hyp = decode_text_from_greedy_hypotheses(hyps, model.decoding)[0]
            etalon_hyps: list[rnnt_utils.Hypothesis] = etalon_search_algo(
                encoder_output=enc_out, encoded_lengths=enc_len
            )[0]
            etalon_hyp = decode_text_from_greedy_hypotheses(etalon_hyps, model.decoding)[0]

            assert hyp.alignments is not None
            assert etalon_hyp.alignments is not None

            assert hyp.text == etalon_hyp.text
            assert len(hyp.alignments) == len(etalon_hyp.alignments)

            for t in range(len(hyp.alignments)):
                t_u = []
                for u in range(len(hyp.alignments[t])):
                    logp, label = hyp.alignments[t][u]
                    assert torch.is_tensor(logp)
                    assert torch.is_tensor(label)
                    etalon_logp, etalon_label = etalon_hyp.alignments[t][u]
                    assert label == etalon_label
                    assert torch.allclose(logp, etalon_logp, atol=1e-4, rtol=1e-4)

                    t_u.append(int(label))

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "greedy"},
            {
                "search_type": "default",
                "beam_size": 2,
            },
            {
                "search_type": "alsd",
                "alsd_max_target_len": 0.5,
                "beam_size": 2,
            },
            {
                "search_type": "tsd",
                "tsd_max_sym_exp_per_step": 3,
                "beam_size": 2,
            },
            {"search_type": "maes", "maes_num_steps": 2, "maes_expansion_beta": 2, "beam_size": 2},
            {"search_type": "maes", "maes_num_steps": 3, "maes_expansion_beta": 1, "beam_size": 2},
        ],
    )
    def test_rnnt_beam_decoding_preserve_alignments(self, test_data_dir, beam_config):
        beam_size = beam_config.pop("beam_size", 1)
        model, encoded, encoded_len = get_model_encoder_output(test_data_dir, 'stt_en_conformer_transducer_small')
        beam = rnnt_beam_decoding.BeamRNNTInfer(
            model.decoder,
            model.joint,
            beam_size=beam_size,
            return_best_hypothesis=False,
            preserve_alignments=True,
            **beam_config,
        )

        enc_out = encoded
        enc_len = encoded_len
        blank_id = torch.tensor(model.joint.num_classes_with_blank - 1, dtype=torch.int32)

        with torch.no_grad():
            hyps = beam(encoder_output=enc_out, encoded_lengths=enc_len)[0]  # type: rnnt_utils.Hypothesis
            hyp, all_hyps = decode_text_from_nbest_hypotheses(hyps, model.decoding)
            hyp = hyp[0]  # best hypothesis
            all_hyps = all_hyps[0]

            assert hyp.alignments is not None

            if beam_config['search_type'] == 'alsd':
                assert len(all_hyps) <= int(beam_config['alsd_max_target_len'] * float(enc_len[0]))

            print("Beam search algorithm :", beam_config['search_type'])
            # Use the following commented print statements to check
            # the alignment of other algorithms compared to the default
            for idx, hyp_ in enumerate(all_hyps):  # type: (int, rnnt_utils.Hypothesis)
                print("Hyp index", idx + 1, "text :", hyp_.text)

                # Alignment length (T) must match audio length (T)
                # NOTE: increase length threshold to two to prevent intermittent failures when a word is split into subwords
                assert abs(len(hyp_.alignments) - enc_len[0]) <= 2  # 1

                for t in range(len(hyp_.alignments)):
                    t_u = []
                    for u in range(len(hyp_.alignments[t])):
                        logp, label = hyp_.alignments[t][u]
                        assert torch.is_tensor(logp)
                        assert torch.is_tensor(label)

                        t_u.append(int(label))

                    # Blank token must be the last token in the current
                    if len(t_u) > 1:
                        assert t_u[-1] == blank_id

                        # No blank token should be present in the current timestamp other than at the end
                        for token in t_u[:-1]:
                            assert token != blank_id

                    print(f"Tokens at timestamp {t} = {t_u}")
                print()

                assert len(hyp_.timestamp) > 0
                print("Timesteps", hyp_.timestamp)
                print()

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_name, decoding_strategy",
        [
            ("stt_en_conformer_transducer_small", "greedy"),
            ("stt_en_conformer_transducer_small", "greedy_batch"),
            ("stt_en_conformer_transducer_small", "beam"),
            # ("stt_en_conformer_transducer_small", "tsd"),
            ("stt_en_conformer_transducer_small", "alsd"),
            ("nvidia/parakeet-tdt_ctc-110m", "greedy"),
            ("nvidia/parakeet-tdt_ctc-110m", "greedy_batch"),
        ],
    )
    def test_subword_decoding_compute_timestamps(self, test_data_dir, decoding_strategy, model_name):

        model, encoded, encoded_len = get_model_encoder_output(test_data_dir, model_name)

        cfg = DictConfig(model.cfg.decoding)
        cfg['strategy'] = decoding_strategy
        cfg['preserve_alignments'] = True
        cfg['compute_timestamps'] = True

        decoding = RNNTBPEDecoding(
            decoding_cfg=cfg, decoder=model.decoder, joint=model.joint, tokenizer=model.tokenizer
        )

        hyps = decoding.rnnt_decoder_predictions_tensor(encoded, encoded_len, return_hypotheses=True)
        if isinstance(hyps[0], list):
            BaseTimestampsTest.check_subword_timestamps(hyps[0][0], decoding)
        else:
            BaseTimestampsTest.check_subword_timestamps(hyps[0], decoding)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model_name, decoding_strategy",
        [
            ("stt_en_conformer_transducer_small", "greedy"),
            ("stt_en_conformer_transducer_small", "greedy_batch"),
            ("stt_en_conformer_transducer_small", "beam"),
            # ("stt_en_conformer_transducer_small", "tsd"),
            ("stt_en_conformer_transducer_small", "alsd"),
            ("nvidia/parakeet-tdt_ctc-110m", "greedy"),
            ("nvidia/parakeet-tdt_ctc-110m", "greedy_batch"),
        ],
    )
    def test_char_decoding_compute_timestamps(self, test_data_dir, decoding_strategy, model_name):

        model, encoded, encoded_len = get_model_encoder_output(test_data_dir, model_name)

        cfg = DictConfig(model.cfg.decoding)
        cfg['strategy'] = decoding_strategy
        cfg['preserve_alignments'] = True
        cfg['compute_timestamps'] = True

        vocab = [t[0] for t in model.tokenizer.vocab]

        decoding = RNNTDecoding(decoding_cfg=cfg, decoder=model.decoder, joint=model.joint, vocabulary=vocab)

        hyps = decoding.rnnt_decoder_predictions_tensor(encoded, encoded_len, return_hypotheses=True)

        if isinstance(hyps[0], list):
            BaseTimestampsTest.check_char_timestamps(hyps[0][0], decoding)
        else:
            BaseTimestampsTest.check_char_timestamps(hyps[0], decoding)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize("use_cuda_graph_decoder", [True, False])
    @pytest.mark.parametrize("use_lm", [True, False])
    @pytest.mark.parametrize("use_boosting_tree", [True, False])
    def test_tdt_greedy_decoding(
        self, test_data_dir, use_cuda_graph_decoder: bool, use_lm: bool, use_boosting_tree: bool
    ):
        kenlm_model_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
        boosting_tree = BoostingTreeModelConfig(key_phrases_list=["hello", "nvidia"]) if use_boosting_tree else None
        check_tdt_greedy_decoding(
            test_data_dir,
            use_cuda_graph_decoder=use_cuda_graph_decoder,
            lm_path=kenlm_model_path if use_lm else None,
            boosting_tree=boosting_tree,
        )

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "beam_config",
        [
            {
                "search_type": "default",
                "beam_size": 2,
            },
            {"search_type": "maes", "maes_num_steps": 2, "maes_expansion_beta": 2, "beam_size": 2},
            {"search_type": "maes", "maes_num_steps": 2, "maes_expansion_beta": 1, "beam_size": 4},
        ],
    )
    def test_tdt_beam_decoding(self, test_data_dir, beam_config):
        check_beam_decoding(test_data_dir, beam_config)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "beam_config",
        [
            {
                "search_type": "maes",
                "maes_num_steps": 2,
                "maes_expansion_beta": 1,
                "beam_size": 4,
                "ngram_lm_alpha": 0.3,
            },
        ],
    )
    def test_tdt_beam_decoding_with_kenlm(self, test_data_dir, beam_config):
        # skipping if kenlm is not installed
        pytest.importorskip("kenlm", reason="Skipping test because 'kenlm' is not installed.")

        kenlm_model_path = os.path.join(
            test_data_dir, "asr", "kenlm_ngram_lm", "parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
        )
        beam_config["ngram_lm_model"] = kenlm_model_path
        check_beam_decoding(test_data_dir, beam_config)


class TestRNNTTimestamps(BaseTimestampsTest):
    """RNNT-specific timestamp tests that inherit from BaseTimestampsTest"""

    def _convert_offsets(self, offsets):
        result = copy.deepcopy(offsets)
        for offset in result:
            offset['char'] = [offset['char']]
        return result

    @property
    def char_offsets_chars(self):
        return self._convert_offsets(super().char_offsets_chars)

    @property
    def char_offsets_wpe(self):
        return self._convert_offsets(super().char_offsets_wpe)

    @property
    def char_offsets_bpe(self):
        return self._convert_offsets(super().char_offsets_bpe)

    @cached_property
    def decoding_char(self):
        cfg = RNNTDecodingConfig()
        vocab = char_vocabulary()
        decoder = get_rnnt_decoder(vocab_size=len(vocab))
        joint = get_rnnt_joint(vocab_size=len(vocab))
        decoding = RNNTDecoding(decoding_cfg=cfg, decoder=decoder, joint=joint, vocabulary=vocab)
        return decoding

    @cached_property
    def decoding_subword_wpe(self):
        cfg = RNNTDecodingConfig()
        vocab = self.tmp_tokenizer.vocab
        decoder = get_rnnt_decoder(vocab_size=len(vocab))
        joint = get_rnnt_joint(vocab_size=len(vocab))
        decoding = RNNTBPEDecoding(decoding_cfg=cfg, decoder=decoder, joint=joint, tokenizer=self.tmp_tokenizer)
        return decoding

    @cached_property
    def decoding_subword_bpe(self):
        vocab = self.bpe_tokenizer.vocab
        cfg = RNNTDecodingConfig()
        decoder = get_rnnt_decoder(vocab_size=len(vocab))
        joint = get_rnnt_joint(vocab_size=len(vocab))
        decoding = RNNTBPEDecoding(decoding_cfg=cfg, decoder=decoder, joint=joint, tokenizer=self.bpe_tokenizer)
        return decoding

    @pytest.mark.unit
    def test_word_offsets_subword_wpe(self, tmp_tokenizer):
        self.tmp_tokenizer = tmp_tokenizer
        super().test_word_offsets_subword_wpe()

    @pytest.mark.unit
    def test_word_offsets_subword_wpe_other_delimiter(self, tmp_tokenizer):
        self.tmp_tokenizer = tmp_tokenizer
        super().test_word_offsets_subword_wpe_other_delimiter()
