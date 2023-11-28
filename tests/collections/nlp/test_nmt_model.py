# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import shutil
import tempfile

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import MTEncDecModel
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import AAYNBaseConfig


def export_test(model, suffix):
    with tempfile.TemporaryDirectory() as restore_folder:
        filename = os.path.join(restore_folder, 'nmt' + suffix)
        enc_filename = os.path.join(restore_folder, 'encoder-nmt' + suffix)
        dec_filename = os.path.join(restore_folder, 'decoder-nmt' + suffix)
        model.export(output=filename, check_trace=True)
        assert os.path.exists(enc_filename)
        assert os.path.exists(dec_filename)


def get_cfg():
    cfg = AAYNBaseConfig()
    cfg.encoder_tokenizer.tokenizer_name = 'yttm'
    cfg.encoder_tokenizer.tokenizer_model = 'tests/.data/yttm.4096.en-de.model'
    cfg.decoder_tokenizer.tokenizer_name = 'yttm'
    cfg.decoder_tokenizer.tokenizer_model = 'tests/.data/yttm.4096.en-de.model'
    cfg.train_ds = None
    cfg.validation_ds = None
    cfg.test_ds = None
    return cfg


class TestMTEncDecModel:
    @pytest.mark.unit
    def test_creation_saving_restoring(self):
        model = MTEncDecModel(cfg=get_cfg())
        assert isinstance(model, MTEncDecModel)
        # Create a new temporary directory
        with tempfile.TemporaryDirectory() as restore_folder:
            with tempfile.TemporaryDirectory() as save_folder:
                save_folder_path = save_folder
                # Where model will be saved
                model_save_path = os.path.join(save_folder, f"{model.__class__.__name__}.nemo")
                model.save_to(save_path=model_save_path)
                # Where model will be restored from
                model_restore_path = os.path.join(restore_folder, f"{model.__class__.__name__}.nemo")
                shutil.copy(model_save_path, model_restore_path)
            # at this point save_folder should not exist
            assert save_folder_path is not None and not os.path.exists(save_folder_path)
            assert not os.path.exists(model_save_path)
            assert os.path.exists(model_restore_path)
            # attempt to restore
            model_copy = model.__class__.restore_from(restore_path=model_restore_path)
            assert model.num_weights == model_copy.num_weights

    @pytest.mark.unit
    def test_no_artifact_name_collision(self):
        model = MTEncDecModel(cfg=get_cfg())
        assert isinstance(model, MTEncDecModel)
        with tempfile.TemporaryDirectory() as tmpdir1:
            model.save_to("nmt_model.nemo")
            with tempfile.TemporaryDirectory() as tmpdir:
                model._save_restore_connector._unpack_nemo_file(path2file="nmt_model.nemo", out_folder=tmpdir)
                conf = OmegaConf.load(os.path.join(tmpdir, "model_config.yaml"))
                # Make sure names now differ in saved config
                assert conf.encoder_tokenizer.tokenizer_model != conf.decoder_tokenizer.tokenizer_model
                # Make sure names in config start with "nemo:" prefix
                assert conf.encoder_tokenizer.tokenizer_model.startswith("nemo:")
                assert conf.decoder_tokenizer.tokenizer_model.startswith("nemo:")
                # Check if both tokenizers were included
                assert os.path.exists(os.path.join(tmpdir, conf.encoder_tokenizer.tokenizer_model[5:]))
                assert os.path.exists(os.path.join(tmpdir, conf.decoder_tokenizer.tokenizer_model[5:]))

    @pytest.mark.unit
    def test_train_eval_loss(self):
        cfg = get_cfg()
        cfg.label_smoothing = 0.5
        model = MTEncDecModel(cfg=cfg)
        assert isinstance(model, MTEncDecModel)
        batch_size = 10
        time = 32
        vocab_size = 32000
        torch.manual_seed(42)
        tgt_ids = torch.LongTensor(batch_size, time).random_(1, model.decoder_tokenizer.vocab_size)
        logits = torch.FloatTensor(batch_size, time, vocab_size).random_(-1, 1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        train_loss = model.loss_fn(log_probs=log_probs, labels=tgt_ids)
        eval_loss = model.eval_loss_fn(log_probs=log_probs, labels=tgt_ids)
        assert not torch.allclose(train_loss, eval_loss)  # , (train_loss, eval_loss)

        cfg.label_smoothing = 0
        model = MTEncDecModel(cfg=cfg)
        # Train loss == val loss when label smoothing = 0
        train_loss = model.loss_fn(log_probs=log_probs, labels=tgt_ids)
        eval_loss = model.eval_loss_fn(log_probs=log_probs, labels=tgt_ids)
        assert torch.allclose(train_loss, eval_loss)

    @pytest.mark.skipif(not os.path.exists('/home/TestData'), reason='Not a Jenkins machine')
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_gpu_export_ts(self):
        model = MTEncDecModel(cfg=get_cfg()).cuda()
        assert isinstance(model, MTEncDecModel)
        export_test(model, ".ts")

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_gpu_export_onnx(self):
        model = MTEncDecModel(cfg=get_cfg()).cuda()
        assert isinstance(model, MTEncDecModel)
        export_test(model, ".onnx")


if __name__ == "__main__":
    t = TestMTEncDecModel()
    # t.test_gpu_export_ts()
    t.test_train_eval_loss()
