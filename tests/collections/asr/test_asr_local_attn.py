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
import os
import shutil
import tempfile

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from nemo.collections.asr.models import ASRModel, EncDecCTCModel


def getattr2(object, attr):
    if not '.' in attr:
        return getattr(object, attr)
    else:
        arr = attr.split('.')
        return getattr2(getattr(object, arr[0]), '.'.join(arr[1:]))


class TestASRLocalAttention:
    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_forward(self):
        asr_model = ASRModel.from_pretrained("stt_en_conformer_ctc_small")
        asr_model = asr_model.eval()

        len = 16000 * 60 * 30  # 30 minutes, OOM without local attention
        input_signal_long = torch.randn(size=(1, len), device=asr_model.device)
        length_long = torch.tensor([len], device=asr_model.device)

        # switch to local attn
        asr_model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=(64, 64))
        with torch.no_grad():
            asr_model.forward(input_signal=input_signal_long, input_signal_length=length_long)

        # switch context size only (keep local)
        asr_model.change_attention_model(att_context_size=(192, 192))
        with torch.no_grad():
            asr_model.forward(input_signal=input_signal_long, input_signal_length=length_long)

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_change_save_restore(self):

        model = ASRModel.from_pretrained("stt_en_conformer_ctc_small")
        model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=(64, 64))
        attr_for_eq_check = ["encoder.self_attention_model", "encoder.att_context_size"]

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
            model_copy = model.__class__.restore_from(
                restore_path=model_restore_path,
                map_location=None,
                strict=True,
                return_config=False,
                override_config_path=None,
            )

            assert model.num_weights == model_copy.num_weights
            if attr_for_eq_check is not None and len(attr_for_eq_check) > 0:
                for attr in attr_for_eq_check:
                    assert getattr2(model, attr) == getattr2(model_copy, attr)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "global_tokens", [0, 1, 4],
    )
    @pytest.mark.parametrize(
        "global_tokens_spacing", [1, 4],
    )
    def test_train(self, global_tokens, global_tokens_spacing):
        preprocessor_config = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor'}
        vocabulary = [
            ' ',
            'a',
            'b',
            'c',
            'd',
            'e',
            'f',
            'g',
            'h',
            'i',
            'j',
            'k',
            'l',
            'm',
            'n',
            'o',
            'p',
            'q',
            'r',
            's',
            't',
            'u',
            'v',
            'w',
            'x',
            'y',
            'z',
            "'",
        ]
        encoder_config = {
            '_target_': 'nemo.collections.asr.modules.ConformerEncoder',
            'feat_in': 64,
            'n_layers': 8,
            'd_model': 4,
            'self_attention_model': 'rel_pos_local_attn',
            'att_context_size': [128, 128],
            'global_tokens': global_tokens,
            'global_tokens_spacing': global_tokens_spacing,
        }
        decoder_config = {
            '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
            'feat_in': None,
            'num_classes': len(vocabulary),
            'vocabulary': vocabulary,
        }
        model_config = DictConfig(
            {
                'preprocessor': DictConfig(preprocessor_config),
                'encoder': DictConfig(encoder_config),
                'decoder': DictConfig(decoder_config),
                'optim': {'name': 'adamw'},
            }
        )

        class DummyDataset(torch.utils.data.Dataset):
            """Simply returns a single set of values."""

            def __init__(self, values):
                self.values = values

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return self.values

        input_signal = torch.randn(size=(1, 960000))
        input_length = torch.tensor([960000])
        target = torch.randint(size=(1, 280), low=0, high=28)
        target_length = torch.tensor([280])

        asr_model = EncDecCTCModel(cfg=model_config)
        asr_model.train()
        _ = asr_model.forward(input_signal=input_signal, input_signal_length=input_length)
        ## Explicitly pass acclerator as cpu, since deafult val in PTL >= 2.0 is auto and it picks cuda
        ## which further causes an error in all reduce at: https://github.com/NVIDIA/NeMo/blob/v1.18.1/nemo/collections/asr/modules/conformer_encoder.py#L462
        ## and in ConvASREncoder, SqueezeformerEncoder where device is CPU
        trainer = pl.Trainer(max_epochs=1, accelerator='cpu')
        trainer.fit(
            asr_model,
            train_dataloaders=torch.utils.data.DataLoader(
                DummyDataset([input_signal, input_length, target, target_length]), collate_fn=lambda x: x[0],
            ),
            val_dataloaders=torch.utils.data.DataLoader(
                DummyDataset([input_signal, input_length, target, target_length]), collate_fn=lambda x: x[0],
            ),
        )
        trainer.test(
            asr_model,
            dataloaders=torch.utils.data.DataLoader(
                DummyDataset([input_signal, input_length, target, target_length]), collate_fn=lambda x: x[0],
            ),
        )
