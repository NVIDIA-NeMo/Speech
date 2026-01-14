# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import os
import tempfile
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from lightning.pytorch.plugins.environments import TorchElasticEnvironment
from megatron.core import parallel_state
from omegaconf import DictConfig, OmegaConf

from nemo.collections.common.parts.nlp_overrides import NLPDDPStrategy
from nemo.collections.multimodal.speech_llm.models import modular_models
from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import shift_tokens_by_multi_audios


class ModularAudioGPTModel(modular_models.ModularAudioGPTModel):
    # disable logging to avoid MisconfigurationException
    def log(self, *args, **kwargs):
        pass


def setup_module():
    pl.seed_everything(1)
    # init model parallel needed for LLM loss
    init_method = 'tcp://'
    master_ip = 'localhost'
    master_port = '6000'
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(backend='gloo', world_size=1, rank=0, init_method=init_method)
    parallel_state.initialize_model_parallel(1, 1)


@pytest.fixture
def llm_model_config():
    this_test_dir = os.path.dirname(os.path.abspath(__file__))
    # Although most of the stuff in model is loaded from ckpt, we need configs
    # for e.g. cfg.model.optim
    config = OmegaConf.load(
        os.path.join(
            this_test_dir,
            "../../../examples/multimodal/speech_llm/conf/modular_audio_gpt_config_peft.yaml",
        )
    )
    # TODO(zhehuai): move the following to Test /home/TestData
    config.model.restore_from_path = "/root/home/works/TestData/pretrained_models/megatron_gpt/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo"
    config.model.micro_batch_size = 2
    config.model.global_batch_size = 2
    config.model.data.validation_ds.manifest_filepath = (
        '/root/home/works/TestData/datasets/LibriSpeech/dev_clean_cleaned.json'
    )
    config.model.data.train_ds.manifest_filepath = (
        '/root/home/works/TestData/datasets/LibriSpeech/dev_clean_cleaned.json'
    )
    return config


@pytest.fixture
def trainer_config():
    config_trainer = DictConfig({})

    if torch.cuda.is_available():
        accelerator = "gpu"
        torch.set_default_device('cuda')
    else:
        accelerator = "cpu"
    config_trainer.accelerator = accelerator
    config_trainer.devices = 1
    config_trainer.num_nodes = 1
    config_trainer.max_epochs = 4
    config_trainer.max_steps = 1
    config_trainer.val_check_interval = 1.0

    # for PyTorch Native AMP set precision=16
    config_trainer.precision = 32

    # setup cluster environment parameters"
    # use torch elastic cluster environment so `create_process_externally` is True
    # the launcher is set to None. It will not try to spawn new processes.
    # It won't create the misconfiguration error because of the `interactive session`
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    strategy = NLPDDPStrategy()
    plugins = [TorchElasticEnvironment()]
    trainer = pl.Trainer(logger=False, plugins=plugins, strategy=strategy, **config_trainer)
    return trainer, config_trainer


@pytest.fixture
def perception_model_config():
    preprocessor = {"_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor"}
    encoder = {
        "_target_": "nemo.collections.asr.modules.ConformerEncoder",
        "feat_in": 64,
        "n_layers": 8,
        "d_model": 64,
        "self_attention_model": "rel_pos_local_attn",
        "att_context_size": [128, 128],
    }

    model_config = DictConfig(
        {
            "_target_": "nemo.collections.multimodal.speechllm.modules.speechllm_perception.AudioPerceptionModule",
            "preprocessor": DictConfig(preprocessor),
            "encoder": DictConfig(encoder),
            "modality_adapter": DictConfig(encoder),
            "output_dim": 1024,
        }
    )
    return model_config


@pytest.fixture
def test_batch():
    signal_len = torch.from_numpy(np.array([64000, 64000]))
    transcript = torch.arange(10).reshape(2, 5).int()
    tokens = transcript[:, :-1]
    labels = transcript[:, 1:]
    transcript_length = torch.Tensor([3, 2]).int()
    # assuming context_lengths = [1, 1]
    loss_mask = torch.Tensor([[0, 1, 1, 0], [0, 1, 0, 0]])
    batch = {
        'audio_signal_length': signal_len,
        'tokens': tokens,
        'tokens_length': transcript_length,
        'contexts': torch.arange(260).reshape(2, 130).int(),
        'context_lengths': torch.Tensor([1, 1]).int(),
        'labels': labels,
        'answers': labels,
        'loss_mask': loss_mask,
    }
    batch['audio_signal'] = torch.randn([2, 64000])
    return batch


@pytest.mark.skip(reason="nedd to move pretrained GPT model to /home/works/TestData first")
class TestModularAudioGPTModel:
    @pytest.mark.unit
    def test_init_and_train(self, llm_model_config, perception_model_config, trainer_config):
        llm_model_config.model.pretrained_audio_model = "stt_en_fastconformer_transducer_large"
        llm_model_config.model.perception = perception_model_config
        trainer, llm_model_config.trainer = trainer_config
        model = ModularAudioGPTModel.restore_from_pretrained_models(llm_model_config, trainer=trainer)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "model.nemo")
            model.train()
            model.save_to(save_path)

    @pytest.mark.unit
    def test_prepare_llm_input(self, llm_model_config, perception_model_config, trainer_config, test_batch):
        llm_model_config.model.pretrained_audio_model = "stt_en_fastconformer_transducer_large"
        llm_model_config.model.perception = perception_model_config
        trainer, llm_model_config.trainer = trainer_config
        model = ModularAudioGPTModel.restore_from_pretrained_models(llm_model_config, trainer=trainer)
        model.cuda()
        model.train()
        batch = {key: val.cuda(non_blocking=True) for key, val in test_batch.items()}
        encoder_input, attention_mask, labels, loss_mask, encoder_length = model.prepare_llm_input(batch)
        assert encoder_input.shape == (17, 2, 768)
        assert np.allclose(encoder_input.sum().cpu().detach().numpy(), 15.783691)
        assert attention_mask.shape == (2, 1, 17, 17)
        assert labels.shape == (2, 17)
        assert np.allclose(loss_mask.sum(axis=1).cpu().numpy(), [2, 1])
        assert np.allclose(encoder_length.cpu().numpy(), (16, 15))

    @pytest.mark.unit
    def test_training_step(self, llm_model_config, perception_model_config, trainer_config, test_batch):
        llm_model_config.model.pretrained_audio_model = "stt_en_fastconformer_transducer_large"
        llm_model_config.model.perception = perception_model_config
        trainer, llm_model_config.trainer = trainer_config
        model = ModularAudioGPTModel.restore_from_pretrained_models(llm_model_config, trainer=trainer)
        model.cuda()
        model.on_train_start()
        model.setup()
        model.train()
        loss_mean = model.training_step(iter([test_batch]), None)
        assert np.allclose(loss_mean.cpu().detach().numpy(), 5.7052)

    @pytest.mark.unit
    def test_validation_step(self, llm_model_config, perception_model_config, trainer_config, test_batch):
        llm_model_config.model.pretrained_audio_model = "stt_en_fastconformer_transducer_large"
        llm_model_config.model.perception = perception_model_config
        trainer, llm_model_config.trainer = trainer_config
        model = ModularAudioGPTModel.restore_from_pretrained_models(llm_model_config, trainer=trainer)
        model.cuda()
        model.train()
        batch = {key: val.cuda(non_blocking=True) for key, val in test_batch.items()}
        loss_mean = model.validation_step(iter([batch]), 0)
        assert np.allclose(loss_mean['loss'].cpu().detach().numpy(), 5.7052)

    @pytest.mark.unit
    def test_predict_step(self, llm_model_config, perception_model_config, trainer_config, test_batch):
        llm_model_config.model.pretrained_audio_model = "stt_en_fastconformer_transducer_large"
        llm_model_config.model.perception = perception_model_config
        trainer, llm_model_config.trainer = trainer_config
        model = ModularAudioGPTModel.restore_from_pretrained_models(llm_model_config, trainer=trainer)
        model.cuda()
        model.train()
        batch = {key: val.cuda(non_blocking=True) for key, val in test_batch.items()}
        response = model.predict_step(batch, 0, 0)
        ground_truth = 'to suit you. Please note these are lecture notes from an alternate presentation. Copyright  ⁇ '
        assert response['sentences'][0] == ground_truth

    @pytest.mark.unit
    def test_concat_multi_features(self, llm_model_config, perception_model_config, trainer_config):
        llm_model_config.model.pretrained_audio_model = "stt_en_fastconformer_transducer_large"
        llm_model_config.model.perception = perception_model_config
        trainer, llm_model_config.trainer = trainer_config
        model = ModularAudioGPTModel.restore_from_pretrained_models(llm_model_config, trainer=trainer)
        model.eval()

        feat_dim = 32
        encoded = [torch.ones([3, 16, feat_dim]), torch.ones([3, 16, feat_dim])]
        encoded_len = [torch.LongTensor([12, 8, 4]), torch.LongTensor([12, 8, 4])]
        input_embeds = torch.zeros([2, 32, feat_dim])
        input_length = torch.LongTensor([32, 28])
        context_start_idx = [[0, 4, 12, 20], [0, 8, 16, 25]]
        encoder_input, encoder_length = model._concat_multi_features(
            encoded, encoded_len, input_embeds, input_length, context_start_idx
        )
        assert encoder_input.shape == (2, 56, feat_dim)  # max audio_len + text_len = (12 + 8 + 4) + 32 = 56
        assert encoder_length.shape == (2,)
        assert np.allclose(encoder_length.cpu().numpy(), (56, 52))
        assert encoder_input[0, : context_start_idx[0][1]].sum() == 0  # first 4 features are text features
        assert np.allclose(
            encoder_input[0, context_start_idx[0][1] : context_start_idx[0][1] + encoded_len[0][0]],
            torch.ones([encoded_len[0][0], feat_dim]),
        )

    @pytest.mark.unit
    def test_shift_tokens_by_multi_audios(self):
        """This test is put here because its functionality is similar to _concat_multi_features()"""
        encoder_max_length = 64
        audio_len = [torch.LongTensor([12, 8, 4]), torch.LongTensor([12, 8, 4])]
        context_tokens = torch.ones([2, 32])
        context_length = torch.LongTensor([32, 28])
        context_start_idx = [[0, 4, 12, 20], [0, 8, 16, 25]]
        new_context_tokens = shift_tokens_by_multi_audios(
            context_tokens, context_length, audio_len, context_start_idx, encoder_max_length
        )
        assert new_context_tokens.shape == (2, 64)
        assert np.allclose(new_context_tokens[0, : context_start_idx[0][1]], torch.ones([context_start_idx[0][1]]))
        assert np.allclose(
            new_context_tokens[0, context_start_idx[0][1] : context_start_idx[0][1] + audio_len[0][0]],
            torch.zeros([audio_len[0][0]]),
        )
