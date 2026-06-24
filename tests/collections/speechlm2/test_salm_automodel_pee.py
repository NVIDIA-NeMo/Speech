# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""SALMAutomodel tests for the Parallel Expert Encoder (PEE) recipe.

Mirrors ``test_salm_automodel.py`` but swaps the plain ConformerEncoder for a
``ParallelExpertEncoder`` (streaming Sortformer diarizer + ASR Conformer, fused).
The tiny-but-real PE encoder is the same dummy bundle built in
``tests/collections/asr/test_parallel_expert_encoder.py`` (``build_toy_pe_encoder``);
this file mounts it onto ``model.perception.encoder`` exactly the way
``nemo.collections.speechlm2.parts.pretrained.setup_parallel_expert_encoder`` does
and exercises the SALM training / validation / generation path end to end, plus the
``spk_targets -> spk_targets`` routing that is unique to the PEE recipe.
"""
import importlib.util
import os
from types import SimpleNamespace

import pytest
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import dummy_cut, dummy_recording
from omegaconf import DictConfig

from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoder
from nemo.collections.common.data.lhotse import NeMoMultimodalConversation
from nemo.collections.common.data.lhotse.text_adapters import AudioTurn, TextTurn
from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.speechlm2.data import SALMDataset
from nemo.collections.speechlm2.models import SALMAutomodel

# Reuse the toy PE encoder (and its dimensions) defined for the standalone
# ParallelExpertEncoder tests, so both suites share one dummy bundle definition.
from tests.collections.asr.test_parallel_expert_encoder import (
    _ASR_D_MODEL,
    _MEL_FEATURES,
    _N_SPK,
    _SUBSAMPLING_FACTOR,
    build_toy_pe_encoder,
)

# SALMAutomodel.configure_model() pulls in the (gitignored) nemo_automodel package,
# so the full-model tests need both CUDA and that dependency to be importable.
# Build a precise skip reason that names only the piece that is actually missing
# (so a CUDA-equipped box without nemo_automodel doesn't get a misleading "requires CUDA").
_HAS_CUDA = torch.cuda.is_available()
_HAS_AUTOMODEL = importlib.util.find_spec("nemo_automodel") is not None
_MISSING = [name for name, ok in (("CUDA", _HAS_CUDA), ("the nemo_automodel package", _HAS_AUTOMODEL)) if not ok]
requires_cuda = pytest.mark.skipif(
    bool(_MISSING),
    reason="SALMAutomodel needs " + " and ".join(_MISSING) if _MISSING else "",
)

# NOTE: deliberately do NOT call torch.set_default_device('cuda') here. It is a
# global, process-wide mutation that leaks into other test modules collected in
# the same pytest session (e.g. the device-agnostic CPU tests in
# tests/collections/asr/test_parallel_expert_encoder.py), causing spurious
# cuda/cpu device-mismatch failures. CUDA tests below place their tensors on
# model.device explicitly instead.


def resolve_pretrained_models():
    if os.path.exists("/home/TestData/speechlm/pretrained_models"):
        # CI pre-cached paths:
        return {
            "pretrained_llm": "/home/TestData/speechlm/pretrained_models/Qwen--Qwen3-1.7B",
            "pretrained_asr": "/home/TestData/speechlm/pretrained_models/canary-1b-flash.nemo",
        }
    else:
        # HF URLs:
        return {
            "pretrained_asr": "nvidia/canary-1b-flash",
            "pretrained_llm": "Qwen/Qwen3-1.7B",
        }


AUDIO_LOCATOR_TAG = "<|audioplaceholder|>"
PROMPT = "qwen"

# Speaker-activity (SOT) target settings; num_speakers matches the PE encoder's n_spk.
SOT_CFG = {
    "num_speakers": _N_SPK,
    "sample_rate": 16000,
    "window_stride": 0.01,
    "subsampling_factor": _SUBSAMPLING_FACTOR,
    "no_rttm_to_ones": True,
}


def mount_dummy_pe_encoder(model: SALMAutomodel) -> ParallelExpertEncoder:
    """Mount a dummy ParallelExpertEncoder onto ``model.perception.encoder``.

    Replicates ``setup_parallel_expert_encoder`` without needing an on-disk
    ``.nemo`` bundle: it swaps the perception encoder for a tiny-but-real PE
    encoder, disables the outer preprocessor normalization (the PE encoder
    re-applies ASR normalization internally), and matches the model dtype/device.
    """
    pe_encoder = build_toy_pe_encoder()
    # The PE encoder consumes un-normalised mels; turn off the perception
    # preprocessor's normalization just like the real mounting helper does.
    model.perception.preprocessor.featurizer.normalize = None
    model.perception.encoder = pe_encoder
    target_dtype = next(model.llm.parameters()).dtype
    model.perception.to(device=model.device, dtype=target_dtype)
    return pe_encoder


@pytest.fixture(scope="session")
def model():
    if _MISSING:
        pytest.skip("SALMAutomodel needs " + " and ".join(_MISSING))
    cfg = {
        **resolve_pretrained_models(),
        "pretrained_weights": False,
        "prompt_format": PROMPT,
        "audio_locator_tag": AUDIO_LOCATOR_TAG,
        "perception": {
            "target": "nemo.collections.speechlm2.modules.perception.AudioPerceptionModule",
            "output_dim": 2048,
            # Placeholder encoder; its only job is to define a d_model that matches
            # the dummy PE encoder we mount in its place (mount validation checks this).
            "encoder": {
                "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                "att_context_size": [-1, -1],
                "causal_downsampling": False,
                "conv_context_size": None,
                "conv_kernel_size": 9,
                "conv_norm_type": "batch_norm",
                "d_model": _ASR_D_MODEL,
                "dropout": 0.0,
                "dropout_att": 0.0,
                "dropout_emb": 0.0,
                "dropout_pre_encoder": 0.0,
                "feat_in": _MEL_FEATURES,
                "feat_out": -1,
                "ff_expansion_factor": 4,
                "n_heads": 4,
                "n_layers": 1,
                "pos_emb_max_len": 5000,
                "self_attention_model": "rel_pos",
                "subsampling": "dw_striding",
                "subsampling_conv_channels": 16,
                "subsampling_factor": _SUBSAMPLING_FACTOR,
            },
            "modality_adapter": {
                "_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector",
                "d_model": _ASR_D_MODEL,
            },
            "preprocessor": {
                "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                "dither": 1e-05,
                "features": _MEL_FEATURES,
                "frame_splicing": 1,
                "log": True,
                "n_fft": 512,
                "normalize": "per_feature",
                "pad_to": 0,
                "pad_value": 0.0,
                "sample_rate": 16000,
                "window": "hann",
                "window_size": 0.025,
                "window_stride": 0.01,
            },
        },
        "optimizer": {"_target_": "torch.optim.AdamW"},
        "torch_dtype": "bfloat16",
    }
    model = SALMAutomodel(cfg)
    model.configure_model()
    model.to("cuda")
    mount_dummy_pe_encoder(model)
    return model


@pytest.fixture(scope="session")
def dataset(model):
    # SOT mode: each batch additionally carries RTTM-derived `spk_targets`.
    return SALMDataset(model.tokenizer, multispeaker_cfg=SOT_CFG)


@pytest.fixture(scope="session")
def prompt_formatter(model):
    return PromptFormatter.resolve(PROMPT)(model.tokenizer)


@pytest.fixture(scope="session")
def training_cutset_batch():
    cut = dummy_cut(0, recording=dummy_recording(0, with_data=True))
    cut.supervisions = [
        SupervisionSegment(
            id=cut.id, recording_id=cut.recording_id, start=0, duration=1.0, text='Some text transcription.'
        )
    ]
    return CutSet(
        [
            NeMoMultimodalConversation(
                id="example-0",
                turns=[
                    TextTurn(role="user", value="Repeat after me:"),
                    AudioTurn(role="user", cut=cut, audio_locator_tag=AUDIO_LOCATOR_TAG),
                    TextTurn(role="assistant", value=cut.supervisions[0].text),
                ],
                token_equivalent_duration=0.08,
            )
        ]
    )


@requires_cuda
def test_salm_automodel_pee_uses_parallel_expert_encoder(model):
    assert isinstance(model.perception.encoder, ParallelExpertEncoder)
    assert model._uses_parallel_expert_encoder()
    # The mounted PE encoder is a drop-in: its d_model drives the perception output projection.
    assert model.perception.encoder.d_model == _ASR_D_MODEL
    assert model.perception.encoder.n_spk == _N_SPK
    # PE encoder consumes un-normalised mels; the outer preprocessor normalization is disabled.
    assert model.perception.preprocessor.featurizer.normalize is None


@requires_cuda
def test_salm_automodel_pee_dataset_emits_speaker_targets(dataset, prompt_formatter, training_cutset_batch):
    training_cutset_batch = training_cutset_batch.map(lambda c: c.apply_prompt_format(prompt_formatter), apply_fn=None)
    batch = dataset[training_cutset_batch]
    for key in ("audios", "audio_lens", "input_ids", "loss_mask", "spk_targets"):
        assert key in batch
        assert torch.is_tensor(batch[key])
    # spk_targets: (B, T_target, num_speakers) -> last dim must match the PE encoder n_spk.
    assert batch["spk_targets"].ndim == 3
    assert batch["spk_targets"].shape[0] == batch["audios"].shape[0]
    assert batch["spk_targets"].shape[-1] == _N_SPK


@requires_cuda
def test_salm_automodel_pee_training_step(model, dataset, prompt_formatter, training_cutset_batch):
    training_cutset_batch = training_cutset_batch.map(lambda c: c.apply_prompt_format(prompt_formatter), apply_fn=None)
    batch = dataset[training_cutset_batch]
    assert "spk_targets" in batch  # injected as spk_targets into the PE encoder during training
    batch = move_data_to_device(batch, device=model.device)
    results = model.training_step(batch, batch_idx=0)
    assert torch.is_tensor(results["loss"])
    assert not torch.isnan(results["loss"])
    assert results["loss"] > 0


@requires_cuda
def test_salm_automodel_pee_validation_step(model, dataset, prompt_formatter, training_cutset_batch):
    model.on_validation_epoch_start()
    training_cutset_batch = training_cutset_batch.map(lambda c: c.apply_prompt_format(prompt_formatter), apply_fn=None)
    batch = dataset[training_cutset_batch]
    batch = move_data_to_device(batch, device=model.device)
    # Validation ignores spk_targets and lets the PE encoder run its embedded Sortformer.
    results = model.validation_step({"dummy_val_set": batch}, batch_idx=0)
    assert results is None


@requires_cuda
def test_salm_automodel_pee_generation(model):
    # No spk_targets at inference -> the PE encoder predicts diarization internally.
    answer = model.generate(
        prompts=[
            [
                {"role": "user", "slots": {"message": f"Repeat after me: {AUDIO_LOCATOR_TAG}"}},
            ]
        ],
        audios=torch.randn(1, 16000, device=model.device),
        audio_lens=torch.tensor([16000], device=model.device),
        max_new_tokens=4,
    )
    assert answer.shape == (1, 4)
    assert answer.dtype == torch.long
    assert (answer >= 0).all()
    assert (answer < model.text_vocab_size).all()


# ----------------------------------------------------------------------------- #
# CPU-only: spk_targets -> spk_targets routing in prepare_inputs.
#
# Uses a bare SALMAutomodel (no LLM/ASR download) with a stub perception whose
# `.encoder` is the real dummy ParallelExpertEncoder, so `_uses_parallel_expert_encoder`
# takes the PEE branch and we can assert how speaker targets are forwarded.
# ----------------------------------------------------------------------------- #
class _PEETestTokenizer:
    pad = 0
    unk_id = None
    bos_id = 1
    eos_id = 2

    def __init__(self, audio_locator_tag):
        self._audio_locator_tag = audio_locator_tag

    def token_to_id(self, token):
        assert token == self._audio_locator_tag
        return 99


class _PEETestLLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(128, 1)
        with torch.no_grad():
            self.model.embed_tokens.weight.zero_()


class _PEETestPerception(torch.nn.Module):
    """Stub perception that exposes a real PE encoder and records the spk_targets it receives."""

    def __init__(self, pe_encoder: ParallelExpertEncoder):
        super().__init__()
        self.encoder = pe_encoder  # real ParallelExpertEncoder -> drives the PEE branch
        self.preprocessor = SimpleNamespace(featurizer=SimpleNamespace(sample_rate=16000, hop_length=160))
        self.spk_targets_calls = []

    def forward(self, input_signal=None, input_signal_length=None, spk_targets=None):
        self.spk_targets_calls.append(spk_targets)
        max_len = int(input_signal_length.max().item())
        return input_signal[:, :max_len].unsqueeze(-1), input_signal_length.clone()


def _make_pee_routing_test_model(pe_encoder, cfg=None):
    model = SALMAutomodel.__new__(SALMAutomodel)
    torch.nn.Module.__init__(model)
    model.cfg = DictConfig(cfg or {"encoder_chunk_size_seconds": None})
    model.audio_locator_tag = AUDIO_LOCATOR_TAG
    model.tokenizer = _PEETestTokenizer(AUDIO_LOCATOR_TAG)
    model.llm = _PEETestLLM()
    model.perception = _PEETestPerception(pe_encoder)
    model._use_tp = False
    return model


@pytest.fixture(scope="module")
def dummy_pe_encoder():
    return build_toy_pe_encoder().eval()


@pytest.mark.unit
def test_pee_prepare_inputs_detects_parallel_expert_encoder(dummy_pe_encoder):
    model = _make_pee_routing_test_model(dummy_pe_encoder)
    assert model._uses_parallel_expert_encoder()


@pytest.mark.unit
def test_pee_prepare_inputs_routes_spk_targets_as_spk_targets(dummy_pe_encoder):
    model = _make_pee_routing_test_model(dummy_pe_encoder)
    spk_targets = torch.rand(1, 4, _N_SPK)
    batch = {
        "audios": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        "audio_lens": torch.tensor([5], dtype=torch.long),
        "input_ids": torch.tensor([[model.audio_locator_tag_id, 10]], dtype=torch.long),
        "loss_mask": torch.tensor([[False, True]], dtype=torch.bool),
        "spk_targets": spk_targets,
    }

    # RTTM-derived spk_targets are forwarded when present in the batch.
    model.prepare_inputs(batch)
    assert model.perception.spk_targets_calls[-1] is spk_targets

    # No spk_targets key means the embedded Sortformer predicts speaker activity.
    batch_without_spk_targets = {k: v for k, v in batch.items() if k != "spk_targets"}
    model.prepare_inputs(batch_without_spk_targets)
    assert model.perception.spk_targets_calls[-1] is None


@pytest.mark.unit
def test_pee_prepare_inputs_warns_for_experimental_inference_options(dummy_pe_encoder):
    model = _make_pee_routing_test_model(
        dummy_pe_encoder,
        cfg={"pe_encoder_path": "/tmp/pee.nemo", "encoder_chunk_size_seconds": 30.0},
    )
    batch = {
        "audios": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        "audio_lens": torch.tensor([5], dtype=torch.long),
        "input_ids": torch.tensor([[model.audio_locator_tag_id, 10]], dtype=torch.long),
        "loss_mask": torch.tensor([[False, True]], dtype=torch.bool),
    }

    with pytest.warns(UserWarning, match="ParallelExpertEncoder inference path.*encoder_chunk_size_seconds"):
        model.prepare_inputs(batch)
    assert model.perception.spk_targets_calls[-1] is None
