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

"""Model-level integration of the Joint turbo kernel: all ASR models with the RNN-T loss."""

import copy
import os
from types import SimpleNamespace
from typing import Optional, Type
from unittest.mock import Mock

import pytest
import torch
from hydra.errors import InstantiationException
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.data.audio_to_eou_label_lhotse import AudioToTextEOUBatch
from nemo.collections.asr.models import (
    EncDecHybridRNNTCTCBPEModel,
    EncDecHybridRNNTCTCModel,
    EncDecRNNTBPEModel,
    EncDecRNNTModel,
)
from nemo.collections.asr.models.asr_eou_models import EncDecHybridRNNTCTCBPEEOUModel, EncDecRNNTBPEEOUModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt import EncDecHybridRNNTCTCBPEModelWithPrompt
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecodingConfig, CTCDecodingConfig
from nemo.collections.asr.parts.turbo_transducer.rnnt_consistency import ConsistencyRNNTLoss
from nemo.collections.asr.parts.turbo_transducer.types import (
    JointLossInputs,
    TrainingKernelType,
    TurboJointPruningType,
)
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

LABELS = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g']
ENCODER_HIDDEN = 16
PREDICTOR_HIDDEN = 8
JOINT_HIDDEN = 16
PROMPT_DICTIONARY = {'en_US': 0, 'es_ES': 1}

requires_triton_cuda = pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()), reason="Triton and CUDA are required"
)

CHAR_MODELS = [EncDecRNNTModel, EncDecHybridRNNTCTCModel]
HYBRID_MODELS = [
    EncDecHybridRNNTCTCModel,
    EncDecHybridRNNTCTCBPEModel,
    EncDecHybridRNNTCTCBPEModelWithPrompt,
    EncDecHybridRNNTCTCBPEEOUModel,
]
PROMPT_MODELS = [EncDecRNNTBPEModelWithPrompt, EncDecHybridRNNTCTCBPEModelWithPrompt]
EOU_MODELS = [EncDecRNNTBPEEOUModel, EncDecHybridRNNTCTCBPEEOUModel]
ALL_MODELS = [
    EncDecRNNTModel,
    EncDecRNNTBPEModel,
    EncDecHybridRNNTCTCModel,
    EncDecHybridRNNTCTCBPEModel,
    EncDecRNNTBPEModelWithPrompt,
    EncDecHybridRNNTCTCBPEModelWithPrompt,
    EncDecRNNTBPEEOUModel,
    EncDecHybridRNNTCTCBPEEOUModel,
]


def _training_kernel_config(pruning: Optional[dict] = None, backend: str = "triton") -> dict:
    return {
        'type': 'fused_vocab',
        'backend': backend,
        # full precision matmuls to compare with the reference computed from the full logits
        'use_high_precision_mm': True,
        'input_grad_full_precision': True,
        'weight_grad_full_precision': True,
        'pruning': pruning,
    }


def _model_config(
    model_class: Type[EncDecRNNTModel],
    test_data_dir: str,
    training_kernel: Optional[dict] = None,
    loss_name: str = "turbo_rnnt",
    loss_kwargs: Optional[dict] = None,
) -> DictConfig:
    """Tiny model config for any ASR model with the RNN-T loss"""
    model_defaults = {'enc_hidden': ENCODER_HIDDEN, 'pred_hidden': PREDICTOR_HIDDEN}
    config = {
        'preprocessor': {
            '_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
            'features': 64,
            'dither': 0.0,
            'pad_to': 0,
        },
        'model_defaults': model_defaults,
        'encoder': {
            '_target_': 'nemo.collections.asr.modules.ConvASREncoder',
            'feat_in': 64,
            'activation': 'relu',
            'conv_mask': True,
            'jasper': [
                {
                    'filters': ENCODER_HIDDEN,
                    'repeat': 1,
                    'kernel': [1],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': True,
                }
            ],
        },
        'decoder': {
            '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
            'prednet': {'pred_hidden': PREDICTOR_HIDDEN, 'pred_rnn_layers': 1},
        },
        'joint': {
            '_target_': 'nemo.collections.asr.modules.RNNTJoint',
            'jointnet': {'joint_hidden': JOINT_HIDDEN, 'activation': 'relu'},
            'log_softmax': False,
            'training_kernel': training_kernel,
        },
        'decoding': {'strategy': 'greedy_batch', 'greedy': {'max_symbols': 5}},
        'loss': {'loss_name': loss_name, f'{loss_name}_kwargs': loss_kwargs or {}},
        'rnnt_reduction': 'mean_volume',
    }
    is_char_model = model_class in CHAR_MODELS
    if is_char_model:
        config['labels'] = LABELS
    else:
        # SentencePiece tokenizer is loaded from the local file (WPE tokenizer sends requests to HuggingFace Hub)
        config['tokenizer'] = {'dir': os.path.join(test_data_dir, "asr", "tokenizers", "an4_spe_128"), 'type': 'bpe'}
    if model_class in HYBRID_MODELS:
        config['aux_ctc'] = {
            'ctc_loss_weight': 0.3,
            'use_cer': False,
            'ctc_reduction': 'mean_batch',
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
                'feat_in': ENCODER_HIDDEN,
                'num_classes': len(LABELS) if is_char_model else -2,
                'vocabulary': LABELS if is_char_model else None,
            },
            'decoding': OmegaConf.structured(CTCDecodingConfig if is_char_model else CTCBPEDecodingConfig),
        }
    if model_class in PROMPT_MODELS:
        model_defaults['initialize_prompt_feature'] = True
        model_defaults['num_prompts'] = 8
        model_defaults['prompt_dictionary'] = PROMPT_DICTIONARY
        config['num_prompts'] = 8
    if model_class in EOU_MODELS:
        # EOU models require the subsampling factor of the encoder and the window stride of the preprocessor
        config['preprocessor']['window_stride'] = 0.01
        config['encoder'] = {
            '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
            'feat_in': 64,
            'n_layers': 1,
            'd_model': ENCODER_HIDDEN,
            'subsampling': 'dw_striding',
            'subsampling_factor': 4,
            'subsampling_conv_channels': 16,
            # fused subsampling kernels are not related to the Joint, their autotuning takes ~30 seconds
            'use_triton': False,
            'n_heads': 2,
            'dropout': 0.0,
            'dropout_pre_encoder': 0.0,
            'dropout_emb': 0.0,
            'dropout_att': 0.0,
        }
    return OmegaConf.create(config)


def _make_batch(model: EncDecRNNTModel, device: torch.device):
    """
    Batch in the format of the model: (signal, signal length, transcript, transcript length[, prompt indices]),
    `AudioToTextEOUBatch` for EOU models
    """
    num_tokens = model.joint.num_classes_with_blank - 1
    signal = torch.randn(2, 8000, device=device)
    signal_len = torch.tensor([8000, 6400], device=device)
    transcript = torch.randint(0, num_tokens, [2, 4], device=device)
    transcript_len = torch.tensor([4, 3], device=device)
    if type(model) in EOU_MODELS:
        return AudioToTextEOUBatch(
            audio_signal=signal,
            audio_lengths=signal_len,
            text_tokens=transcript,
            text_token_lengths=transcript_len,
            eou_targets=torch.zeros([2, 13], dtype=torch.long, device=device),
            eou_target_lengths=torch.tensor([13, 11], device=device),
        )
    batch = (signal, signal_len, transcript, transcript_len)
    if type(model) in PROMPT_MODELS:
        batch += (torch.tensor([0, 1], device=device),)
    return batch


def _prepare_for_direct_steps(model: EncDecRNNTModel, monkeypatch, global_step: int = 0):
    """Allow calling `training_step` / `validation_pass` directly, without the trainer"""
    model._optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # WER is not computed in training: `(global_step + 1) % log_every_n_steps != 0`
    model._trainer = SimpleNamespace(global_step=global_step, log_every_n_steps=10**6)
    monkeypatch.setattr(model, "log_dict", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)


def _make_models(
    model_class: Type[EncDecRNNTModel], test_data_dir: str, device: torch.device, flash=False, **training_kernel_kwargs
):
    """Reference model (standard Joint) and the model with Joint turbo kernel with identical weights"""
    torch.manual_seed(1234)
    # NB: some models (EOU) require the `trainer` argument
    reference_model = model_class(cfg=_model_config(model_class, test_data_dir), trainer=None).to(device)
    kernel = {"type": "flash", "max_joint_rows": 64} if flash else _training_kernel_config(**training_kernel_kwargs)
    turbo_model = model_class(
        cfg=_model_config(model_class, test_data_dir, training_kernel=kernel),
        trainer=None,
    ).to(device)
    turbo_model.load_state_dict(reference_model.state_dict(), strict=True)
    assert not reference_model.joint.uses_training_kernel
    assert turbo_model.joint.training_kernel_type is TrainingKernelType(kernel["type"])
    return reference_model, turbo_model


def _loss_inputs(model: EncDecRNNTModel, device: torch.device, batch_size: int = 2, requires_grad: bool = False):
    """Encoder output [B, D, T], prediction network output [B, D, U+1], targets and lengths"""
    num_frames, num_text_units = 6, 3
    num_tokens = model.joint.num_classes_with_blank - 1
    encoded = torch.randn(batch_size, ENCODER_HIDDEN, num_frames, device=device, requires_grad=requires_grad)
    predictor_output = torch.randn(
        batch_size, PREDICTOR_HIDDEN, num_text_units + 1, device=device, requires_grad=requires_grad
    )
    transcript = torch.randint(0, num_tokens, [batch_size, num_text_units], device=device)
    encoded_len = torch.tensor([num_frames, 4] * (batch_size // 2), device=device)
    target_length = torch.tensor([num_text_units, 2] * (batch_size // 2), device=device)
    return dict(
        encoded=encoded,
        decoder=predictor_output,
        transcript=transcript,
        encoded_len=encoded_len,
        target_length=target_length,
    )


class TestModelTurboKernelConfig:
    @pytest.mark.unit
    @pytest.mark.parametrize("model_class", ALL_MODELS)
    def test_model_with_training_kernel(self, model_class, test_data_dir):
        model = model_class(
            cfg=_model_config(model_class, test_data_dir, training_kernel=_training_kernel_config()), trainer=None
        )
        assert model.joint.uses_training_kernel
        assert model.loss.supports_logprobs_input
        assert model.consistency_loss is None
        # the Joint config is preserved in the model config (e.g., for saving the checkpoint)
        assert model.cfg.joint.training_kernel.type == "fused_vocab"

    @pytest.mark.unit
    @pytest.mark.parametrize("model_class", [EncDecRNNTModel, EncDecHybridRNNTCTCBPEModel])
    @pytest.mark.parametrize("loss_name,loss_kwargs", [("warprnnt_numba", None), ("turbo_rnnt", {"clamp": 1.0})])
    def test_training_kernel_requires_loss_from_logprobs(self, model_class, loss_name, loss_kwargs, test_data_dir):
        config = _model_config(
            model_class,
            test_data_dir,
            training_kernel=_training_kernel_config(),
            loss_name=loss_name,
            loss_kwargs=loss_kwargs,
        )
        with pytest.raises(ValueError, match="training_kernel"):
            model_class(cfg=config)
        # the same loss is valid without the turbo kernel
        config.joint.training_kernel = None
        assert not model_class(cfg=config).joint.uses_training_kernel

    @pytest.mark.unit
    def test_training_kernel_rejects_fuse_loss_wer(self, test_data_dir):
        config = _model_config(EncDecRNNTModel, test_data_dir, training_kernel=_training_kernel_config())
        config.joint.fuse_loss_wer = True
        config.joint.fused_batch_size = 1
        # the Joint is instantiated by Hydra, which wraps the error
        with pytest.raises(InstantiationException, match="fuse_loss_wer"):
            EncDecRNNTModel(cfg=config)

    @pytest.mark.unit
    def test_training_kernel_rejects_tdt(self, test_data_dir):
        config = _model_config(
            EncDecRNNTModel, test_data_dir, training_kernel=_training_kernel_config(), loss_name="tdt"
        )
        config.joint.num_extra_outputs = 5
        with pytest.raises(InstantiationException, match="extra Joint outputs"):
            EncDecRNNTModel(cfg=config)

    @pytest.mark.unit
    def test_training_kernel_is_preserved_on_vocabulary_change(self, test_data_dir):
        model = EncDecRNNTModel(
            cfg=_model_config(EncDecRNNTModel, test_data_dir, training_kernel=_training_kernel_config())
        )
        model.change_vocabulary(new_vocabulary=LABELS + ['h'])
        assert model.joint.num_classes_with_blank == len(LABELS) + 2
        assert model.joint.training_kernel_type is TrainingKernelType.FUSED_VOCAB
        assert model.loss.supports_logprobs_input

    @pytest.mark.unit
    @pytest.mark.parametrize("model_class", [EncDecRNNTBPEModel, EncDecHybridRNNTCTCBPEModel])
    def test_training_kernel_is_preserved_on_tokenizer_change(self, model_class, test_data_dir):
        """Regression: the loss was replaced by the default one, which can't be computed from log-probabilities"""
        config = _model_config(
            model_class,
            test_data_dir,
            training_kernel=_training_kernel_config(),
            loss_kwargs={"fastemit_lambda": 0.01},
        )
        model = model_class(cfg=config)
        model.change_vocabulary(new_tokenizer_dir=config.tokenizer.dir, new_tokenizer_type="bpe")
        assert model.joint.training_kernel_type is TrainingKernelType.FUSED_VOCAB
        assert model.loss.supports_logprobs_input
        # the loss is created from the config of the model, as in the constructor
        assert model.loss._loss.fastemit_lambda == 0.01
        assert model.loss.reduction == "mean_volume"

    @pytest.mark.unit
    def test_save_restore(self, test_data_dir, tmp_path):
        training_kernel = _training_kernel_config(pruning={'type': 'two_pass', 'range_size': 2, 'start_step': 5})
        model = EncDecRNNTModel(cfg=_model_config(EncDecRNNTModel, test_data_dir, training_kernel=training_kernel))
        model.save_to(str(tmp_path / "model.nemo"))
        restored_model = EncDecRNNTModel.restore_from(str(tmp_path / "model.nemo"), map_location="cpu")
        assert restored_model.joint.training_kernel_type is TrainingKernelType.FUSED_VOCAB
        assert restored_model.joint._training_kernel_cfg == model.joint._training_kernel_cfg
        assert restored_model.joint._training_kernel_cfg.pruning.pruning_type is TurboJointPruningType.TWO_PASS
        assert restored_model.joint._training_kernel_cfg.pruning.range_size == 2
        assert restored_model.joint._training_kernel_cfg.pruning.start_step == 5


class TestModelComputeLossesRouting:
    """Routing of `_compute_losses` between the logits and the log-probabilities path (no kernels involved)"""

    @staticmethod
    def _model_stub(uses_training_kernel: bool, loss_supports_logprobs: bool = True, consistency_loss=None):
        joint = Mock()
        joint.uses_training_kernel = uses_training_kernel
        joint.supports_loss_logprobs = True
        joint.num_extra_outputs = 0
        loss = Mock()
        loss.supports_logprobs_input = loss_supports_logprobs
        return SimpleNamespace(joint=joint, loss=loss, consistency_loss=consistency_loss)

    @pytest.mark.unit
    @pytest.mark.parametrize("current_step", [None, 5])
    def test_training_kernel_uses_logprobs_path(self, current_step):
        model = self._model_stub(uses_training_kernel=True)
        target_logprobs, blank_logprobs = torch.randn(2, 6, 4), torch.randn(2, 6, 4)
        model.joint.forward_for_loss.return_value = JointLossInputs(target_logprobs, blank_logprobs)
        inputs = _loss_inputs(SimpleNamespace(joint=SimpleNamespace(num_classes_with_blank=9)), torch.device("cpu"))

        loss_value, consistency_loss_value = EncDecRNNTModel._compute_losses(
            model, **inputs, current_step=current_step
        )

        assert loss_value is model.loss.forward_from_logprobs.return_value
        assert consistency_loss_value is None
        model.joint.forward_for_loss.assert_called_once_with(
            encoder_outputs=inputs["encoded"],
            decoder_outputs=inputs["decoder"],
            encoder_lengths=inputs["encoded_len"],
            transcripts=inputs["transcript"],
            transcript_lengths=inputs["target_length"],
            compute_consistency=False,
            consistency_symmetrical=True,
            current_step=current_step,
        )
        model.loss.forward_from_logprobs.assert_called_once_with(
            target_logprobs=target_logprobs,
            blank_logprobs=blank_logprobs,
            input_lengths=inputs["encoded_len"],
            target_lengths=inputs["target_length"],
        )
        model.joint.assert_not_called()  # the full Joint output is not computed
        model.loss.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.parametrize("with_trainer", [True, False])
    def test_full_lattice_loss_is_logged(self, with_trainer):
        """Two-pass pruning: the loss of the full lattice reported by the Joint is reduced as the loss and logged"""
        model = self._model_stub(uses_training_kernel=True)
        model._trainer = object() if with_trainer else None
        model.log = Mock()
        model._log_full_lattice_loss = lambda *args: EncDecRNNTModel._log_full_lattice_loss(model, *args)
        full_lattice_loss = torch.tensor([3.0, 5.0])
        model.joint.forward_for_loss.return_value = JointLossInputs(
            torch.randn(2, 6, 4), torch.randn(2, 6, 4), full_lattice_loss=full_lattice_loss
        )
        inputs = _loss_inputs(SimpleNamespace(joint=SimpleNamespace(num_classes_with_blank=9)), torch.device("cpu"))

        loss_value, _ = EncDecRNNTModel._compute_losses(model, **inputs, current_step=5)

        # the training loss is the loss of the pruned lattice
        assert loss_value is model.loss.forward_from_logprobs.return_value
        if with_trainer:
            (reduce_args,) = model.loss.reduce.call_args_list
            assert reduce_args.args[0] is full_lattice_loss
            assert torch.equal(reduce_args.args[1], inputs["target_length"])
            model.log.assert_called_once_with("train_loss_full_lattice", model.loss.reduce.return_value)
        else:
            model.log.assert_not_called()

    @pytest.mark.unit
    def test_standard_joint_uses_logits_path(self):
        """Without the turbo kernel the behavior is not changed, even if the loss supports log-probabilities"""
        model = self._model_stub(uses_training_kernel=False)
        inputs = _loss_inputs(SimpleNamespace(joint=SimpleNamespace(num_classes_with_blank=9)), torch.device("cpu"))

        loss_value, consistency_loss_value = EncDecRNNTModel._compute_losses(model, **inputs, current_step=5)

        assert loss_value is model.loss.return_value
        assert consistency_loss_value is None
        model.joint.assert_called_once_with(encoder_outputs=inputs["encoded"], decoder_outputs=inputs["decoder"])
        model.loss.assert_called_once_with(
            log_probs=model.joint.return_value,
            targets=inputs["transcript"],
            input_lengths=inputs["encoded_len"],
            target_lengths=inputs["target_length"],
        )
        model.joint.forward_for_loss.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.parametrize("uses_training_kernel", [False, True])
    def test_consistency_uses_logprobs_path(self, uses_training_kernel: bool):
        consistency_loss = Mock()
        consistency_loss.symmetrical = False
        model = self._model_stub(uses_training_kernel=uses_training_kernel, consistency_loss=consistency_loss)
        consistency_kl = torch.randn(2, 6, 4)
        model.joint.forward_for_loss.return_value = JointLossInputs(
            torch.randn(4, 6, 4), torch.randn(4, 6, 4), consistency_kl
        )
        inputs = _loss_inputs(
            SimpleNamespace(joint=SimpleNamespace(num_classes_with_blank=9)), torch.device("cpu"), batch_size=4
        )

        loss_value, consistency_loss_value = EncDecRNNTModel._compute_losses(model, **inputs, compute_consistency=True)

        assert loss_value is model.loss.forward_from_logprobs.return_value
        assert consistency_loss_value is consistency_loss.forward_from_kl.return_value
        assert model.joint.forward_for_loss.call_args.kwargs["compute_consistency"] is True
        assert model.joint.forward_for_loss.call_args.kwargs["consistency_symmetrical"] is False
        consistency_call = consistency_loss.forward_from_kl.call_args.kwargs
        assert consistency_call["consistency_kl"] is consistency_kl
        # only the teacher (first) half of the lengths is used for the consistency loss
        assert torch.equal(consistency_call["src_lengths"], inputs["encoded_len"][:2])
        assert torch.equal(consistency_call["tgt_lengths"], inputs["target_length"][:2])

    @pytest.mark.unit
    def test_consistency_with_loss_from_logits(self):
        """Loss that requires the full Joint output: both losses are computed from the logits"""
        consistency_loss = Mock()
        model = self._model_stub(
            uses_training_kernel=False, loss_supports_logprobs=False, consistency_loss=consistency_loss
        )
        logits = torch.randn(4, 6, 4, 9)
        model.joint.return_value = logits
        inputs = _loss_inputs(
            SimpleNamespace(joint=SimpleNamespace(num_classes_with_blank=9)), torch.device("cpu"), batch_size=4
        )

        loss_value, consistency_loss_value = EncDecRNNTModel._compute_losses(model, **inputs, compute_consistency=True)

        assert loss_value is model.loss.return_value
        assert consistency_loss_value is consistency_loss.return_value
        model.joint.forward_for_loss.assert_not_called()
        consistency_call = consistency_loss.call_args.kwargs
        assert torch.equal(consistency_call["teacher_logits"], logits[:2])
        assert torch.equal(consistency_call["student_logits"], logits[2:])

    @pytest.mark.unit
    @pytest.mark.parametrize("loss_supports_logprobs", [False, True])
    def test_consistency_rejects_odd_batch(self, loss_supports_logprobs):
        """Regression: with the loss from the logits, the halves of an odd batch were broadcast silently"""
        model = self._model_stub(
            uses_training_kernel=False, loss_supports_logprobs=loss_supports_logprobs, consistency_loss=Mock()
        )
        inputs = _loss_inputs(
            SimpleNamespace(joint=SimpleNamespace(num_classes_with_blank=9)), torch.device("cpu"), batch_size=3
        )
        with pytest.raises(ValueError, match="double batch"):
            EncDecRNNTModel._compute_losses(model, **inputs, compute_consistency=True)

    @pytest.mark.unit
    def test_consistency_requires_consistency_loss(self):
        model = self._model_stub(uses_training_kernel=True)
        inputs = _loss_inputs(
            SimpleNamespace(joint=SimpleNamespace(num_classes_with_blank=9)), torch.device("cpu"), batch_size=4
        )
        with pytest.raises(ValueError, match="consistency_loss"):
            EncDecRNNTModel._compute_losses(model, **inputs, compute_consistency=True)

    @pytest.mark.unit
    def test_consistency_rejects_extra_outputs(self):
        model = self._model_stub(uses_training_kernel=False, loss_supports_logprobs=False, consistency_loss=Mock())
        model.joint.num_extra_outputs = 5
        inputs = _loss_inputs(
            SimpleNamespace(joint=SimpleNamespace(num_classes_with_blank=9)), torch.device("cpu"), batch_size=4
        )
        with pytest.raises(NotImplementedError, match="extra outputs"):
            EncDecRNNTModel._compute_losses(model, **inputs, compute_consistency=True)


@pytest.mark.gpu
@requires_triton_cuda
class TestModelTurboKernel:
    @pytest.mark.unit
    @pytest.mark.parametrize("model_class,flash", [(cls, False) for cls in ALL_MODELS] + [(EncDecRNNTModel, True)])
    def test_training_step_matches_standard_joint(self, model_class, flash, test_data_dir, monkeypatch):
        device = torch.device("cuda")
        reference_model, turbo_model = _make_models(model_class, test_data_dir, device, flash=flash)
        batch = _make_batch(reference_model, device)
        global_step = 3

        forward_for_loss = Mock(wraps=turbo_model.joint.forward_for_loss)
        monkeypatch.setattr(turbo_model.joint, "forward_for_loss", forward_for_loss)

        losses, grads = [], []
        for model in [reference_model, turbo_model]:
            model.train()
            _prepare_for_direct_steps(model, monkeypatch, global_step=global_step)
            loss_value = model.training_step(copy.copy(batch), 0)['loss']
            loss_value.backward()
            losses.append(loss_value.detach())
            grads.append({name: parameter.grad for name, parameter in model.named_parameters()})

        # the model passes the optimizer step to the Joint (schedule of pruning)
        forward_for_loss.assert_called_once()
        assert forward_for_loss.call_args.kwargs["current_step"] == global_step
        assert forward_for_loss.call_args.kwargs["compute_consistency"] is False

        assert torch.isfinite(losses[0])
        assert torch.allclose(losses[1], losses[0], atol=1e-4, rtol=1e-4)
        for name, reference_grad in grads[0].items():
            turbo_grad = grads[1][name]
            assert (reference_grad is None) == (turbo_grad is None), name
            if reference_grad is not None:
                assert torch.allclose(turbo_grad, reference_grad, atol=2e-4, rtol=2e-4), name
        assert turbo_model.joint.joint_net[-1].weight.grad is not None

    @pytest.mark.unit
    @pytest.mark.parametrize("model_class", ALL_MODELS)
    def test_validation_matches_standard_joint(self, model_class, test_data_dir, monkeypatch):
        device = torch.device("cuda")
        # pruning is configured, but must not be applied (and must not require the step) in validation
        reference_model, turbo_model = _make_models(
            model_class, test_data_dir, device, pruning={'type': 'two_pass', 'range_size': 2, 'start_step': 5}
        )
        batch = _make_batch(reference_model, device)

        logs = []
        for model in [reference_model, turbo_model]:
            model.eval()
            _prepare_for_direct_steps(model, monkeypatch)
            assert model.compute_eval_loss
            with torch.no_grad():
                logs.append(model.validation_pass(copy.copy(batch), 0, 0))

        assert torch.isfinite(logs[0]['val_loss'])
        assert torch.allclose(logs[1]['val_loss'], logs[0]['val_loss'], atol=1e-4, rtol=1e-4)
        # decoding is not affected by the turbo kernel
        assert torch.equal(logs[1]['val_wer_num'], logs[0]['val_wer_num'])
        assert torch.equal(logs[1]['val_wer_denom'], logs[0]['val_wer_denom'])

    @pytest.mark.unit
    def test_validation_after_decoding_temperature_change(self, test_data_dir, monkeypatch):
        """
        Regression: the temperature set by the decoding config (`change_decoding_strategy`) was rejected by
        the Joint with the turbo kernel, and validation with the loss failed. The temperature is a parameter of
        decoding: the loss does not depend on it, as for the standard Joint without log-softmax.
        """
        device = torch.device("cuda")
        reference_model, turbo_model = _make_models(EncDecRNNTModel, test_data_dir, device)
        batch = _make_batch(reference_model, device)

        logs = []
        for model in [reference_model, turbo_model]:
            model.joint.log_softmax = None  # as in the example configs: the loss is computed from the raw logits
            model.eval()
            _prepare_for_direct_steps(model, monkeypatch)
            assert model.compute_eval_loss
            for temperature in [1.0, 0.8]:
                model.change_decoding_strategy(
                    OmegaConf.create(
                        {'strategy': 'greedy_batch', 'greedy': {'max_symbols': 5}, 'temperature': temperature}
                    )
                )
                assert model.joint.temperature == temperature
                with torch.no_grad():
                    logs.append(model.validation_pass(copy.copy(batch), 0, 0))

        reference_loss = logs[0]['val_loss']
        assert torch.isfinite(reference_loss)
        for log in logs[1:]:
            assert torch.allclose(log['val_loss'], reference_loss, atol=1e-4, rtol=1e-4)

    @pytest.mark.unit
    def test_pruning_is_scheduled_by_training_step(self, test_data_dir, monkeypatch):
        device = torch.device("cuda")
        _, turbo_model = _make_models(
            EncDecRNNTModel,
            test_data_dir,
            device,
            pruning={'type': 'backward', 'range_size': 2, 'start_step': 5},
        )
        batch = _make_batch(turbo_model, device)
        range_sizes = []
        get_active_pruning = turbo_model.joint._get_active_pruning

        def spy_active_pruning(current_step):
            pruning = get_active_pruning(current_step)
            range_sizes.append(None if pruning is None else pruning.range_size)
            return pruning

        monkeypatch.setattr(turbo_model.joint, "_get_active_pruning", spy_active_pruning)
        turbo_model.train()
        losses = []
        for global_step in [4, 5]:
            _prepare_for_direct_steps(turbo_model, monkeypatch, global_step=global_step)
            turbo_model.zero_grad()
            loss_value = turbo_model.training_step(copy.copy(batch), 0)['loss']
            loss_value.backward()
            losses.append(loss_value.detach())
            assert torch.isfinite(turbo_model.joint.joint_net[-1].weight.grad).all()
        assert range_sizes == [None, 2]
        # backward pruning does not change the loss value
        assert torch.equal(losses[0], losses[1])

    @pytest.mark.unit
    @pytest.mark.parametrize("model_class", [EncDecRNNTModel, EncDecHybridRNNTCTCBPEModel])
    def test_training_step_with_two_pass_pruning(self, model_class, test_data_dir, monkeypatch):
        """Two-pass pruning in `training_step`: scheduled by the step, the loss of the full lattice is logged"""
        device = torch.device("cuda")
        _, turbo_model = _make_models(
            model_class, test_data_dir, device, pruning={'type': 'two_pass', 'range_size': 2, 'start_step': 5}
        )
        batch = _make_batch(turbo_model, device)
        turbo_model.train()
        logged_by_step = {}
        for global_step in [4, 5]:
            _prepare_for_direct_steps(turbo_model, monkeypatch, global_step=global_step)
            logged = logged_by_step.setdefault(global_step, {})
            monkeypatch.setattr(
                turbo_model, "log", lambda name, value, logged=logged, **kwargs: logged.update({name: value})
            )
            turbo_model.zero_grad()
            loss_value = turbo_model.training_step(copy.copy(batch), 0)['loss']
            loss_value.backward()
            logged["loss"] = loss_value.detach()
            assert torch.isfinite(turbo_model.joint.joint_net[-1].weight.grad).all()
        # before `start_step`: the full lattice, nothing is logged by the Joint
        assert "train_loss_full_lattice" not in logged_by_step[4]
        full_lattice_loss = logged_by_step[5]["train_loss_full_lattice"]
        assert torch.isfinite(full_lattice_loss) and not full_lattice_loss.requires_grad
        # the pruned lattice contains a part of the alignments: its loss is not less than the loss of the full lattice
        assert logged_by_step[5]["loss"] > logged_by_step[4]["loss"]
        if model_class is EncDecRNNTModel:
            # the same parameters and batch (no dropout): the loss of the full lattice is the loss of the previous step
            torch.testing.assert_close(full_lattice_loss, logged_by_step[4]["loss"])

    @pytest.mark.unit
    @pytest.mark.parametrize("model_class", [EncDecRNNTModel, EncDecHybridRNNTCTCBPEModel])
    def test_compute_losses_matches_standard_joint(self, model_class, test_data_dir):
        torch.manual_seed(17)
        device = torch.device("cuda")
        reference_model, turbo_model = _make_models(model_class, test_data_dir, device)
        inputs = _loss_inputs(turbo_model, device, requires_grad=True)
        reference_inputs = {
            key: value.detach().clone().requires_grad_(value.requires_grad) for key, value in inputs.items()
        }

        loss_value, consistency_loss_value = turbo_model._compute_losses(**inputs)
        reference_loss_value, _ = reference_model._compute_losses(**reference_inputs)
        assert consistency_loss_value is None
        assert torch.allclose(loss_value, reference_loss_value, atol=1e-4, rtol=1e-4)

        loss_value.backward()
        reference_loss_value.backward()
        assert torch.allclose(inputs["encoded"].grad, reference_inputs["encoded"].grad, atol=2e-4, rtol=2e-4)
        assert torch.allclose(inputs["decoder"].grad, reference_inputs["decoder"].grad, atol=2e-4, rtol=2e-4)

    @pytest.mark.unit
    @pytest.mark.parametrize("symmetrical", [True, False])
    @pytest.mark.parametrize("use_training_kernel", [False, True])
    def test_compute_losses_with_consistency_matches_reference(
        self, symmetrical: bool, use_training_kernel: bool, test_data_dir
    ):
        """
        Consistency loss is not configured by the model, but `_compute_losses` supports it:
        the fused path must match the reference (full logits + two separate losses).
        """
        torch.manual_seed(17)
        device = torch.device("cuda")
        consistency_weight = 0.2
        reference_model, turbo_model = _make_models(EncDecRNNTModel, test_data_dir, device)
        model = turbo_model if use_training_kernel else reference_model
        model.consistency_loss = ConsistencyRNNTLoss(symmetrical=symmetrical, reduction="mean_volume")
        model.train()

        half_batch_size = 2
        inputs = _loss_inputs(model, device, batch_size=2 * half_batch_size, requires_grad=True)
        # double batch: targets of the student (second half) are the same as for the teacher
        inputs["transcript"] = inputs["transcript"][:half_batch_size].repeat(2, 1)

        loss_value, consistency_loss_value = model._compute_losses(**inputs, compute_consistency=True)
        (loss_value + consistency_weight * consistency_loss_value).backward()

        reference_encoded = inputs["encoded"].detach().clone().requires_grad_(True)
        reference_predictor_output = inputs["decoder"].detach().clone().requires_grad_(True)
        reference_logits = reference_model.joint(
            encoder_outputs=reference_encoded, decoder_outputs=reference_predictor_output
        )
        reference_loss_value = reference_model.loss(
            log_probs=reference_logits,
            targets=inputs["transcript"],
            input_lengths=inputs["encoded_len"],
            target_lengths=inputs["target_length"],
        )
        reference_consistency_value = model.consistency_loss(
            teacher_logits=reference_logits[:half_batch_size],
            student_logits=reference_logits[half_batch_size:],
            src_lengths=inputs["encoded_len"][:half_batch_size],
            tgt_lengths=inputs["target_length"][:half_batch_size],
        )
        reference_gradients = torch.autograd.grad(
            reference_loss_value + consistency_weight * reference_consistency_value,
            [reference_encoded, reference_predictor_output],
        )

        tolerance = dict(atol=2e-4, rtol=2e-4) if use_training_kernel else dict(atol=1e-5, rtol=1e-5)
        assert torch.allclose(loss_value, reference_loss_value, **tolerance)
        assert torch.allclose(consistency_loss_value, reference_consistency_value, **tolerance)
        assert torch.allclose(inputs["encoded"].grad, reference_gradients[0], **tolerance)
        assert torch.allclose(inputs["decoder"].grad, reference_gradients[1], **tolerance)

    @pytest.mark.unit
    def test_compute_losses_with_consistency_and_loss_from_logits(self, test_data_dir):
        """The loss requiring full logits (gradient clamping) must still compute both losses"""
        torch.manual_seed(19)
        device = torch.device("cuda")
        model = EncDecRNNTModel(
            cfg=_model_config(EncDecRNNTModel, test_data_dir, loss_kwargs={"clamp": 1.0}),
        ).to(device)
        model.consistency_loss = ConsistencyRNNTLoss()
        model.train()
        assert not model.loss.supports_logprobs_input

        inputs = _loss_inputs(model, device, batch_size=4, requires_grad=True)
        inputs["transcript"] = inputs["transcript"][:2].repeat(2, 1)
        loss_value, consistency_loss_value = model._compute_losses(**inputs, compute_consistency=True)
        assert torch.isfinite(loss_value)
        assert torch.isfinite(consistency_loss_value)
        (loss_value + 0.2 * consistency_loss_value).backward()
        assert torch.isfinite(inputs["encoded"].grad).all()
        assert torch.isfinite(inputs["decoder"].grad).all()
