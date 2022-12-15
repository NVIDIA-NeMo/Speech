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
import torch

from nemo.collections.asr.models import ASRModel


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
