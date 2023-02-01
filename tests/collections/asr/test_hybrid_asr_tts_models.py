# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import tempfile
from pathlib import Path

import pytest
import torch.nn as nn
from omegaconf import DictConfig

from nemo.collections.asr.models import ASRModel, EncDecCTCModelBPE, EncDecRNNTBPEModel
from nemo.collections.asr.models.hybrid_asr_tts_models import ASRWithTTSModel
from nemo.collections.asr.parts.submodules.batchnorm import FusedBatchNorm1d
from nemo.collections.tts.models import FastPitchModel


@pytest.fixture(scope="module")
def fastpitch_model():
    model = FastPitchModel.from_pretrained(model_name="tts_en_fastpitch_multispeaker")
    return model


@pytest.fixture(scope="module")
def fastpitch_model_path(fastpitch_model, tmp_path_factory):
    path = tmp_path_factory.mktemp("tts_models") / "fastpitch.nemo"
    fastpitch_model.save_to(path)
    return path


@pytest.fixture(scope="module")
def conformer_ctc_bpe_bn_model():
    model = EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small")
    return model


@pytest.fixture(scope="module")
def conformer_ctc_bpe_bn_model_path(conformer_ctc_bpe_bn_model, tmp_path_factory):
    path = tmp_path_factory.mktemp("asr_models") / "conformer-ctc-bpe-bn.nemo"
    conformer_ctc_bpe_bn_model.save_to(path)
    return path


@pytest.fixture(scope="module")
def conformer_rnnt_bpe_bn_model():
    model = EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_conformer_transducer_small")
    return model


@pytest.fixture(scope="module")
def conformer_rnnt_bpe_bn_model_path(conformer_rnnt_bpe_bn_model, tmp_path_factory):
    path = tmp_path_factory.mktemp("asr_models") / "conformer-rnnt-bpe.nemo"
    conformer_rnnt_bpe_bn_model.save_to(path)
    return path


@pytest.fixture
def asr_model_ctc_bpe_config(test_data_dir):
    preprocessor = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor'}
    encoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASREncoder',
        'feat_in': 64,
        'activation': 'relu',
        'conv_mask': True,
        'jasper': [
            {
                'filters': 1024,
                'repeat': 1,
                'kernel': [1],
                'stride': [1],
                'dilation': [1],
                'dropout': 0.0,
                'residual': False,
                'separable': True,
                'se': True,
                'se_context_size': -1,
            }
        ],
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
        'feat_in': 1024,
        'num_classes': -1,
        'vocabulary': None,
    }

    tokenizer = {'dir': str(Path(test_data_dir) / "asr/tokenizers/an4_wpe_128"), 'type': 'wpe'}

    model_config = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'tokenizer': DictConfig(tokenizer),
        }
    )
    return model_config


@pytest.fixture
def asr_tts_ctc_bpe_model(asr_model_ctc_bpe_config, fastpitch_model_path):
    model = ASRWithTTSModel.from_asr_config(
        asr_cfg=asr_model_ctc_bpe_config, asr_model_type="ctc_bpe", tts_model_path=fastpitch_model_path,
    )
    return model


class TestASRWithTTSModel:
    @pytest.mark.with_downloads
    @pytest.mark.unit
    def test_from_pretrained_ctc_model(self, fastpitch_model_path, conformer_ctc_bpe_bn_model_path):
        model = ASRWithTTSModel.from_pretrained_models(
            asr_model_path=conformer_ctc_bpe_bn_model_path, tts_model_path=fastpitch_model_path
        )
        assert isinstance(model.tts_model, FastPitchModel)
        assert isinstance(model.asr_model, EncDecCTCModelBPE)

    @pytest.mark.with_downloads
    @pytest.mark.unit
    def test_from_pretrained_rnnt_model(self, fastpitch_model_path, conformer_rnnt_bpe_bn_model_path):
        model = ASRWithTTSModel.from_pretrained_models(
            asr_model_path=conformer_rnnt_bpe_bn_model_path, tts_model_path=fastpitch_model_path
        )
        assert isinstance(model.tts_model, FastPitchModel)
        assert isinstance(model.asr_model, EncDecRNNTBPEModel)

    @pytest.mark.with_downloads
    @pytest.mark.unit
    def test_from_asr_config(self, asr_model_ctc_bpe_config, fastpitch_model_path):
        model = ASRWithTTSModel.from_asr_config(
            asr_cfg=asr_model_ctc_bpe_config, asr_model_type="ctc_bpe", tts_model_path=fastpitch_model_path,
        )
        assert isinstance(model.tts_model, FastPitchModel)
        assert isinstance(model.asr_model, EncDecCTCModelBPE)

    @pytest.mark.with_downloads
    @pytest.mark.unit
    def test_save_restore(self, asr_tts_ctc_bpe_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "model.nemo")
            asr_tts_ctc_bpe_model.train()
            asr_tts_ctc_bpe_model.save_to(save_path)

            restored_model = ASRModel.restore_from(save_path)
            assert isinstance(restored_model, ASRWithTTSModel)
            assert isinstance(restored_model.tts_model, FastPitchModel)
            assert isinstance(restored_model.asr_model, EncDecCTCModelBPE)

    @pytest.mark.with_downloads
    @pytest.mark.unit
    def test_save_restore_asr(self, asr_tts_ctc_bpe_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "asr_model.nemo")
            asr_tts_ctc_bpe_model.save_asr_model_to(save_path)

            restored_model = ASRModel.restore_from(save_path)
            assert isinstance(restored_model, EncDecCTCModelBPE)

    @pytest.mark.with_downloads
    @pytest.mark.unit
    def test_from_pretrained_ctc_model_fused_bn(self, fastpitch_model_path, conformer_ctc_bpe_bn_model_path):
        model = ASRWithTTSModel.from_pretrained_models(
            asr_model_path=conformer_ctc_bpe_bn_model_path,
            tts_model_path=fastpitch_model_path,
            asr_model_fuse_bn=True,
        )
        assert isinstance(model.tts_model, FastPitchModel)
        assert isinstance(model.asr_model, EncDecCTCModelBPE)
        assert model.asr_model.cfg.encoder.conv_norm_type == "fused_batch_norm"

        # test model has fused BatchNorm
        has_fused_bn = False
        for name, module in model.asr_model.named_modules():
            assert not isinstance(module, nn.BatchNorm1d)
            has_fused_bn = has_fused_bn or isinstance(module, FusedBatchNorm1d)
        assert has_fused_bn, "Fused BatchNorm not found model"

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "asr_tts_model.nemo")
            model.save_to(save_path)

            # check restored model has fused batchnorm
            model = ASRWithTTSModel.restore_from(save_path)
            assert model.asr_model.cfg.encoder.conv_norm_type == "fused_batch_norm"

            has_fused_bn = False
            for name, module in model.asr_model.named_modules():
                assert not isinstance(module, nn.BatchNorm1d)
                has_fused_bn = has_fused_bn or isinstance(module, FusedBatchNorm1d)
            assert has_fused_bn, "Fused BatchNorm not found model"
