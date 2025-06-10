# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import glob
import os
from pathlib import Path

import jiwer
import pytest
import torch
from omegaconf import open_dict
from tqdm import tqdm

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.parts.submodules.ctc_beam_decoding import BeamBatchedCTCInfer
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.submodules.rnnt_beam_decoding import BeamBatchedRNNTInfer
from nemo.collections.asr.parts.submodules.tdt_beam_decoding import BeamBatchedTDTInfer
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.core.utils import numba_utils
from nemo.core.utils.cuda_python_utils import skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__

RNNT_MODEL = "stt_en_conformer_transducer_small"
CTC_MODEL = "nvidia/stt_en_conformer_ctc_small"
TDT_MODEL = "nvidia/stt_en_fastconformer_tdt_large"
MAX_SAMPLES = 10

DEVICES = [torch.device("cpu")]

if torch.cuda.is_available():
    DEVICES.append(torch.device('cuda'))

NUMBA_RNNT_LOSS_AVAILABLE = numba_utils.numba_cpu_is_supported(
    __NUMBA_MINIMUM_VERSION__
) or numba_utils.numba_cuda_is_supported(__NUMBA_MINIMUM_VERSION__)


def load_audio(file_path, target_sr=16000):
    import librosa

    audio, sr = librosa.load(file_path, sr=target_sr)
    return torch.tensor(audio, dtype=torch.float32), sr


# available audio filename fixtures
@pytest.fixture(scope="module")
def test_audio_filenames(test_data_dir):
    return tuple(glob.glob(os.path.join(test_data_dir, "asr", "test", "an4", "wav", "*.wav")))


# model fixtures
@pytest.fixture(scope="module")
def rnnt_model():
    model = ASRModel.from_pretrained(model_name=RNNT_MODEL, map_location="cpu")
    model.eval()
    return model


@pytest.fixture(scope="module")
def tdt_model():
    model = ASRModel.from_pretrained(model_name=TDT_MODEL, map_location="cpu")
    model.eval()
    return model


@pytest.fixture(scope="module")
def ctc_model():
    model = ASRModel.from_pretrained(model_name=CTC_MODEL, map_location="cpu")
    model.eval()
    return model


# encoder output fixtures
@pytest.fixture(scope="module")
def get_rnnt_encoder_output(rnnt_model, test_audio_filenames):
    encoder_output, encoded_lengths = get_transducer_model_encoder_output(
        test_audio_filenames, MAX_SAMPLES, rnnt_model
    )
    return encoder_output, encoded_lengths


@pytest.fixture(scope="module")
def get_tdt_encoder_output(tdt_model, test_audio_filenames):
    encoder_output, encoded_lengths = get_transducer_model_encoder_output(test_audio_filenames, MAX_SAMPLES, tdt_model)
    return encoder_output, encoded_lengths


@pytest.fixture(scope="module")
def get_ctc_output(ctc_model, test_audio_filenames):
    encoder_output, encoded_lengths = get_ctc_model_output(test_audio_filenames, MAX_SAMPLES, ctc_model)
    return encoder_output, encoded_lengths


@pytest.fixture(scope="module")
def kenlm_model_path(tmp_path_factory, test_data_dir):
    lm_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
    assert os.path.exists(lm_path), f"LM file not found: {lm_path}"
    lm_nemo_path = tmp_path_factory.mktemp("lm") / f"{lm_path.name}.nemo"
    NGramGPULanguageModel.from_file(lm_path, vocab_size=1024).save_to(f"{lm_nemo_path}")
    return f"{lm_nemo_path}"


def get_transducer_model_encoder_output(
    test_audio_filenames,
    num_samples: int,
    model: ASRModel,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    audio_filepaths = test_audio_filenames[:num_samples]

    with torch.no_grad():
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
        model.eval()

        all_inputs, all_lengths = [], []
        for audio_file in tqdm(audio_filepaths, desc="Loading audio files"):
            audio_tensor, _ = load_audio(audio_file)
            all_inputs.append(audio_tensor)
            all_lengths.append(torch.tensor(audio_tensor.shape[0], dtype=torch.int64))

        input_batch = torch.nn.utils.rnn.pad_sequence(all_inputs, batch_first=True).to(device=device, dtype=dtype)
        length_batch = torch.tensor(all_lengths, dtype=torch.int64).to(device)

        encoded_outputs, encoded_length = model(input_signal=input_batch, input_signal_length=length_batch)

    return encoded_outputs, encoded_length


def get_ctc_model_output(
    test_audio_filenames,
    num_samples: int,
    model: ASRModel,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    audio_filepaths = test_audio_filenames[:num_samples]

    with torch.no_grad():
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
        model.eval()

        all_inputs, all_lengths = [], []
        for audio_file in tqdm(audio_filepaths, desc="Loading audio files"):
            audio_tensor, _ = load_audio(audio_file)
            all_inputs.append(audio_tensor)
            all_lengths.append(torch.tensor(audio_tensor.shape[0], dtype=torch.int64))

        input_batch = torch.nn.utils.rnn.pad_sequence(all_inputs, batch_first=True).to(device=device, dtype=dtype)
        length_batch = torch.tensor(all_lengths, dtype=torch.int64).to(device)

        log_probs, encoded_length, _ = model(input_signal=input_batch, input_signal_length=length_batch)

    return log_probs, encoded_length


def print_unit_test_info(strategy, batch_size, beam_size, allow_cuda_graphs, device):
    print(
        f"""Beam search algorithm: {strategy},
                Batch size: {batch_size},
                Beam size: {beam_size},
                Cuda Graphs: {allow_cuda_graphs},
                Decoding device: {device}
            """
    )


def check_res_best_hyps(num_samples, hyps):
    assert type(hyps) == list
    assert type(hyps[0]) == rnnt_utils.Hypothesis

    assert len(hyps) == num_samples

    assert all(
        [
            hasattr(hyps[hyp_idx], "y_sequence")
            and hasattr(hyps[hyp_idx], "score")
            and hasattr(hyps[hyp_idx], "timestamp")
            for hyp_idx in range(num_samples)
        ]
    )


def print_res_best_hyps(hyps):
    for hyp_idx, hyp in enumerate(hyps):
        print("Sample: ", hyp_idx)
        print("Decoded text: ", hyp.text)
        print("Score: ", hyp.score)
        print("Transcript", hyp.y_sequence)
        print("Timesteps", hyp.timestamp)
        print()


def check_res_nbest_hyps(num_samples, batch_nbest_hyps):
    assert type(batch_nbest_hyps) == list
    assert type(batch_nbest_hyps[0]) == rnnt_utils.NBestHypotheses

    assert len(batch_nbest_hyps) == num_samples

    for idx in range(num_samples):
        assert all(
            [
                hasattr(batch_nbest_hyps[idx].n_best_hypotheses[hyp_idx], "y_sequence")
                and hasattr(batch_nbest_hyps[idx].n_best_hypotheses[hyp_idx], "score")
                and hasattr(batch_nbest_hyps[idx].n_best_hypotheses[hyp_idx], "timestamp")
                for hyp_idx in range(len(batch_nbest_hyps[idx].n_best_hypotheses))
            ]
        )

        assert all(
            [
                len(batch_nbest_hyps[idx].n_best_hypotheses[hyp_idx].y_sequence) > 0
                and len(batch_nbest_hyps[idx].n_best_hypotheses[hyp_idx].timestamp) > 0
                for hyp_idx in range(len(batch_nbest_hyps[idx].n_best_hypotheses))
            ]
        )


def print_res_nbest_hyps(batch_nbest_hyps):
    for batch_idx, nbest_hyps in enumerate(batch_nbest_hyps):
        print(f"Batch idx: {batch_idx}")
        for idx, hyp in enumerate(nbest_hyps):
            print(f"Hyp index: {idx + 1}")
            print("Text: ", hyp.text)
            print("Score: ", hyp.score)
            print("Transcripts: ", hyp.y_sequence)
            print("Timesteps: ", hyp.timestamp)
            print()


def decode_text_from_hypotheses(hyps, model):
    if isinstance(model, EncDecCTCModel):
        return model.decoding.decode_hypothesis(hyps, fold_consecutive=False)
    else:
        return model.decoding.decode_hypothesis(hyps)


def decode_text_from_nbest_hypotheses(hyps, model):
    if isinstance(model, EncDecCTCModel):
        return [
            model.decoding.decode_hypothesis(nbest_hyp.n_best_hypotheses, fold_consecutive=False) for nbest_hyp in hyps
        ]
    else:
        return [model.decoding.decode_hypothesis(nbest_hyp.n_best_hypotheses) for nbest_hyp in hyps]


class TestRNNTDecoding:
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "malsd_batch", "allow_cuda_graphs": False},
            {"search_type": "malsd_batch", "allow_cuda_graphs": True},
            {"search_type": "maes_batch", "allow_cuda_graphs": False},
        ],
    )
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("batch_size", [4, 16])
    @pytest.mark.parametrize("device", DEVICES)
    def test_rnnt_beam_decoding_return_best_hypothesis(
        self, test_audio_filenames, rnnt_model, get_rnnt_encoder_output, beam_config, device, batch_size, beam_size
    ):
        num_samples = min(batch_size, len(test_audio_filenames))
        model = rnnt_model.to(device)
        encoder_output, encoded_lengths = get_rnnt_encoder_output
        encoder_output, encoded_lengths = encoder_output[:num_samples].to(device), encoded_lengths[:num_samples].to(
            device
        )

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedRNNTInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=True,
            **beam_config,
        )

        print_unit_test_info(
            strategy=beam_config['search_type'],
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]

            check_res_best_hyps(num_samples, hyps)
            hyps = decode_text_from_hypotheses(hyps, model)
            print_res_best_hyps(hyps)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test is only GPU-based decoding")
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "malsd_batch", "allow_cuda_graphs": False},
            {"search_type": "malsd_batch", "allow_cuda_graphs": True},
            {"search_type": "maes_batch", "allow_cuda_graphs": False},
        ],
    )
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("batch_size", [4])
    def test_rnnt_beam_decoding_return_nbest(
        self, test_audio_filenames, rnnt_model, get_rnnt_encoder_output, beam_config, device, beam_size, batch_size
    ):
        device = torch.device("cuda")
        num_samples = min(batch_size, len(test_audio_filenames))
        model = rnnt_model.to(device)
        encoder_output, encoded_lengths = get_rnnt_encoder_output
        encoder_output, encoded_lengths = encoder_output[:num_samples].to(device), encoded_lengths[:num_samples].to(
            device
        )

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedRNNTInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=False,
            **beam_config,
        )

        print_unit_test_info(
            strategy=beam_config['search_type'],
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            batch_nbest_hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]

            check_res_nbest_hyps(num_samples, batch_nbest_hyps)
            batch_nbest_hyps = decode_text_from_nbest_hypotheses(batch_nbest_hyps, model)
            print_res_nbest_hyps(batch_nbest_hyps)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test is only GPU-based decoding")
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "malsd_batch", "allow_cuda_graphs": False, "ngram_lm_alpha": 0.3},
            {"search_type": "maes_batch", "allow_cuda_graphs": False, "ngram_lm_alpha": 0.3},
            {"search_type": "malsd_batch", "allow_cuda_graphs": True, "ngram_lm_alpha": 0.3},
        ],
    )
    @pytest.mark.parametrize("batch_size", [4])
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("pruning_mode", ["late", "early"])
    @pytest.mark.parametrize("blank_lm_score_mode", ["no_score", "lm_weighted_full"])
    def test_rnnt_beam_decoding_kenlm(
        self,
        kenlm_model_path,
        test_audio_filenames,
        rnnt_model,
        get_rnnt_encoder_output,
        beam_config,
        device,
        batch_size,
        beam_size,
        pruning_mode,
        blank_lm_score_mode,
    ):
        device = torch.device("cuda")
        beam_config["ngram_lm_model"] = kenlm_model_path

        num_samples = min(batch_size, len(test_audio_filenames))
        model = rnnt_model.to(device)
        encoder_output, encoded_lengths = get_rnnt_encoder_output
        encoder_output, encoded_lengths = encoder_output[:num_samples].to(device), encoded_lengths[:num_samples].to(
            device
        )

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedRNNTInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=True,
            pruning_mode=pruning_mode,
            blank_lm_score_mode=blank_lm_score_mode,
            **beam_config,
        )

        print_unit_test_info(
            strategy=beam_config['search_type'],
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]

            check_res_best_hyps(num_samples, hyps)
            hyps = decode_text_from_hypotheses(hyps, model)
            print_res_best_hyps(hyps)


class TestTDTDecoding:
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "malsd_batch", "allow_cuda_graphs": False},
            {"search_type": "malsd_batch", "allow_cuda_graphs": True},
        ],
    )
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("batch_size", [4, 16])
    @pytest.mark.parametrize("device", DEVICES)
    def test_tdt_beam_decoding_return_best_hypothesis(
        self, test_audio_filenames, tdt_model, get_tdt_encoder_output, beam_config, device, batch_size, beam_size
    ):
        num_samples = min(batch_size, len(test_audio_filenames))
        model = tdt_model.to(device)
        encoder_output, encoded_lengths = get_tdt_encoder_output
        encoder_output, encoded_lengths = encoder_output[:num_samples].to(device), encoded_lengths[:num_samples].to(
            device
        )

        model_config = model.to_config_dict()
        durations = list(model_config["model_defaults"]["tdt_durations"])

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedTDTInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            durations=durations,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=True,
            **beam_config,
        )

        print_unit_test_info(
            strategy=beam_config['search_type'],
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]

            check_res_best_hyps(num_samples, hyps)
            hyps = decode_text_from_hypotheses(hyps, model)
            print_res_best_hyps(hyps)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test is only GPU-based decoding")
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "malsd_batch", "allow_cuda_graphs": False},
            {"search_type": "malsd_batch", "allow_cuda_graphs": True},
        ],
    )
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("batch_size", [4])
    def test_tdt_beam_decoding_return_nbest(
        self, test_audio_filenames, tdt_model, get_tdt_encoder_output, beam_config, device, beam_size, batch_size
    ):
        device = torch.device("cuda")
        num_samples = min(batch_size, len(test_audio_filenames))
        model = tdt_model.to(device)
        encoder_output, encoded_lengths = get_tdt_encoder_output
        encoder_output, encoded_lengths = encoder_output[:num_samples].to(device), encoded_lengths[:num_samples].to(
            device
        )

        model_config = model.to_config_dict()
        durations = list(model_config["model_defaults"]["tdt_durations"])

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedTDTInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            durations=durations,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=False,
            **beam_config,
        )

        print_unit_test_info(
            strategy=beam_config['search_type'],
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            batch_nbest_hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]

            check_res_nbest_hyps(num_samples, batch_nbest_hyps)
            batch_nbest_hyps = decode_text_from_nbest_hypotheses(batch_nbest_hyps, model)
            print_res_nbest_hyps(batch_nbest_hyps)

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test is only GPU-based decoding")
    @pytest.mark.parametrize(
        "beam_config",
        [
            {
                "search_type": "malsd_batch",
                "allow_cuda_graphs": False,
                "ngram_lm_alpha": 0.3,
            },
            {
                "search_type": "malsd_batch",
                "allow_cuda_graphs": True,
                "ngram_lm_alpha": 0.3,
            },
        ],
    )
    @pytest.mark.parametrize("batch_size", [4])
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("pruning_mode", ["late", "early"])
    @pytest.mark.parametrize("blank_lm_score_mode", ["lm_weighted_full", "no_score"])
    def test_tdt_beam_decoding_kenlm(
        self,
        kenlm_model_path,
        test_audio_filenames,
        tdt_model,
        get_tdt_encoder_output,
        beam_config,
        device,
        batch_size,
        beam_size,
        pruning_mode,
        blank_lm_score_mode,
    ):
        device = torch.device("cuda")
        beam_config["ngram_lm_model"] = kenlm_model_path

        num_samples = min(batch_size, len(test_audio_filenames))
        model = tdt_model.to(device)
        encoder_output, encoded_lengths = get_tdt_encoder_output
        encoder_output, encoded_lengths = encoder_output[:num_samples].to(device), encoded_lengths[:num_samples].to(
            device
        )

        model_config = model.to_config_dict()
        durations = list(model_config["model_defaults"]["tdt_durations"])

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedTDTInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            durations=durations,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=True,
            pruning_mode=pruning_mode,
            blank_lm_score_mode=blank_lm_score_mode,
            **beam_config,
        )

        print_unit_test_info(
            strategy=beam_config['search_type'],
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]

            check_res_best_hyps(num_samples, hyps)
            hyps = decode_text_from_hypotheses(hyps, model)
            print_res_best_hyps(hyps)


class TestTransducerCudaGraphBeamDecoding:
    """
    Tests CudaGraphs implementations from Transducer models (RNN-T and TDT)
    """

    @pytest.mark.with_downloads
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decoder can run only on CUDA")
    @pytest.mark.parametrize("force_mode", ["no_graphs", "no_while_loops", "full_graph"])
    @pytest.mark.parametrize("model_type", ["rnnt", "tdt"])
    def test_stated_stateless(self, test_audio_filenames, rnnt_model, tdt_model, model_type, force_mode: str):
        """
        Compares pure Pytorch and with three modes of statefull implementations for double floating point precision.
            1. Pure pytorch, but statefull implementation: no_graphs
            2. With CudaGrpahs: no_while_loops and full_graph.
        """
        if force_mode == "full_graph":
            skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()

        batch_size = 16
        device = torch.device("cuda")
        model = rnnt_model.to(device) if model_type == "rnnt" else tdt_model.to(device)
        decoding_config = copy.deepcopy(model.cfg.decoding)

        with open_dict(decoding_config):
            decoding_config["strategy"] = "malsd_batch"
            decoding_config["beam"]["beam_size"] = 4
            decoding_config["beam"]["return_best_hypothesis"] = False
            decoding_config["beam"]["allow_cuda_graphs"] = False

        model.change_decoding_strategy(decoding_config)

        actual_hypotheses = model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)
        actual_transcripts = [[hyp.text for hyp in actual_beam] for actual_beam in actual_hypotheses]
        actual_scores = [[hyp.score for hyp in actual_beam] for actual_beam in actual_hypotheses]
        actual_timestamps = [[hyp.timestamp for hyp in actual_beam] for actual_beam in actual_hypotheses]

        # transcribe with use implementation with cuda graphs
        decoding_config["beam"]["allow_cuda_graphs"] = True
        model.change_decoding_strategy(decoding_config)
        model.decoding.decoding._decoding_computer.force_cuda_graphs_mode(mode=force_mode)

        cudagraph_hypotheses = model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)
        cudagraph_transcripts = [[hyp.text for hyp in cudagraphs_beam] for cudagraphs_beam in cudagraph_hypotheses]
        cudagraph_scores = [[hyp.score for hyp in cudagraph_beam] for cudagraph_beam in cudagraph_hypotheses]
        cudagraph_timestamps = [[hyp.timestamp for hyp in cudagraph_beam] for cudagraph_beam in cudagraph_hypotheses]

        for batch_idx in range(min(batch_size, len(test_audio_filenames))):
            assert len(actual_transcripts[batch_idx]) == len(cudagraph_transcripts[batch_idx])
            assert cudagraph_scores[batch_idx] == pytest.approx(
                actual_scores[batch_idx], abs=1e-2
            ), f"Scores mismatch for batch_idx {batch_idx}"
            assert (
                cudagraph_timestamps[batch_idx] == actual_timestamps[batch_idx]
            ), f"Timestamps mismatch for batch_idx {batch_idx}"

            wer = jiwer.wer(actual_transcripts[batch_idx], cudagraph_transcripts[batch_idx])

            assert wer <= 1e-3, "Cuda graph greedy decoder should match original decoder implementation."

            for actual, fast in zip(actual_transcripts[batch_idx], cudagraph_transcripts[batch_idx]):
                if actual != fast:
                    print("Erroneous samples in batch:", batch_idx)
                    print("Original transcript:", actual)
                    print("New transcript:", fast)

    @pytest.mark.with_downloads
    @pytest.mark.skipif(
        not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()), reason="CUDA decoder can run only on CUDA"
    )
    @pytest.mark.parametrize("model_type", ["rnnt", "tdt"])
    def test_stated_stateless_bf16(self, test_audio_filenames, rnnt_model, tdt_model, model_type):
        """
        Checks that we are able to run without errors all decodings in bfloat16.
        Computational errors accumulate, so just checking if algorithms run without errors
        """
        batch_size = 16
        device = torch.device("cuda")
        model = rnnt_model.to(device) if model_type == "rnnt" else tdt_model.to(device)
        decoding_config = copy.deepcopy(model.cfg.decoding)

        # checking pytorch implementation
        with open_dict(decoding_config):
            decoding_config["strategy"] = "malsd_batch"
            decoding_config["beam"]["beam_size"] = 4
            decoding_config["beam"]["return_best_hypothesis"] = False
            decoding_config["beam"]["allow_cuda_graphs"] = False

        model.change_decoding_strategy(decoding_config)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)

        modes = ["no_graphs", "no_while_loops", "full_graph"]
        for force_mode in modes:
            if force_mode == "full_graph":
                skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()

            # transcribe with use implementation with cuda graphs
            decoding_config["beam"]["allow_cuda_graphs"] = True
            model.change_decoding_strategy(decoding_config)
            model.decoding.decoding._decoding_computer.force_cuda_graphs_mode(mode=force_mode)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)


class TestCTCDecoding:
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"allow_cuda_graphs": False},
            {"allow_cuda_graphs": True},
        ],
    )
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("batch_size", [4, 16])
    @pytest.mark.parametrize("device", DEVICES)
    def test_ctc_beam_decoding_return_best_hypothesis(
        self, test_audio_filenames, ctc_model, get_ctc_output, beam_config, device, batch_size, beam_size
    ):
        num_samples = min(batch_size, len(test_audio_filenames))
        model = ctc_model.to(device)
        log_probs, encoded_lengths = get_ctc_output
        log_probs, encoded_lengths = log_probs[:num_samples].to(device), encoded_lengths[:num_samples].to(device)

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedCTCInfer(
            blank_index=vocab_size,
            beam_size=beam_size,
            return_best_hypothesis=True,
            **beam_config,
        )

        print_unit_test_info(
            strategy="beam_batch",
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            hyps = decoding(decoder_output=log_probs, decoder_lengths=encoded_lengths)[0]

            check_res_best_hyps(num_samples, hyps)
            hyps = decode_text_from_hypotheses(hyps, model)
            print_res_best_hyps(hyps)

    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test is only GPU-based decoding")
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"allow_cuda_graphs": False},
            {"allow_cuda_graphs": True},
        ],
    )
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("batch_size", [4])
    def test_ctc_beam_decoding_return_nbest(
        self, test_audio_filenames, ctc_model, get_ctc_output, beam_config, device, beam_size, batch_size
    ):
        device = torch.device("cuda")
        num_samples = min(batch_size, len(test_audio_filenames))
        model = ctc_model.to(device)
        log_probs, encoded_lengths = get_ctc_output
        log_probs, encoded_lengths = log_probs[:num_samples].to(device), encoded_lengths[:num_samples].to(device)

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedCTCInfer(
            blank_index=vocab_size,
            beam_size=beam_size,
            return_best_hypothesis=False,
            **beam_config,
        )

        print_unit_test_info(
            strategy="beam_batch",
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            batch_nbest_hyps = decoding(decoder_output=log_probs, decoder_lengths=encoded_lengths)[0]

            check_res_nbest_hyps(num_samples, batch_nbest_hyps)
            batch_nbest_hyps = decode_text_from_nbest_hypotheses(batch_nbest_hyps, model)
            print_res_nbest_hyps(batch_nbest_hyps)

    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test is only GPU-based decoding")
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"allow_cuda_graphs": False, "ngram_lm_alpha": 0.3, "beam_beta": 1.0},
            {"allow_cuda_graphs": False, "ngram_lm_alpha": 0.3, "beam_beta": 1.0},
        ],
    )
    @pytest.mark.parametrize("batch_size", [4])
    @pytest.mark.parametrize("beam_size", [4])
    def test_ctc_beam_decoding_kenlm(
        self,
        kenlm_model_path,
        test_audio_filenames,
        ctc_model,
        get_ctc_output,
        beam_config,
        device,
        batch_size,
        beam_size,
    ):
        device = torch.device("cuda")
        beam_config["ngram_lm_model"] = kenlm_model_path

        num_samples = min(batch_size, len(test_audio_filenames))
        model = ctc_model.to(device)
        decoder_output, decoder_lengths = get_ctc_output
        decoder_output, decoder_lengths = decoder_output[:num_samples].to(device), decoder_lengths[:num_samples].to(
            device
        )

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedCTCInfer(
            blank_index=vocab_size,
            beam_size=beam_size,
            return_best_hypothesis=True,
            **beam_config,
        )

        print_unit_test_info(
            strategy="beam_batch",
            batch_size=batch_size,
            beam_size=beam_size,
            allow_cuda_graphs=beam_config.get('allow_cuda_graphs', True),
            device=device,
        )

        with torch.no_grad():
            hyps = decoding(decoder_output=decoder_output, decoder_lengths=decoder_lengths)[0]

            check_res_best_hyps(num_samples, hyps)
            hyps = decode_text_from_hypotheses(hyps, model)
            print_res_best_hyps(hyps)
