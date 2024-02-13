# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import pytest
import torch
from pytorch_lightning import Trainer

from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.collections.asr.parts import context_biasing
from nemo.collections.asr.parts.context_biasing.ctc_based_word_spotter import WSHyp
from nemo.collections.asr.parts.utils import rnnt_utils


@pytest.fixture(scope="module")
def conformer_ctc_bpe_model():
    model = EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small")
    model.set_trainer(Trainer(devices=1, accelerator="cpu"))
    model = model.eval()
    return model


class TestContextGraphCTC:
    @pytest.mark.unit
    def test_graph_building(self):
        context_biasing_list = [["gpu", [['▁g', 'p', 'u'], ['▁g', '▁p', '▁u']]]]
        context_graph = context_biasing.ContextGraphCTC(blank_id=1024)
        context_graph.add_to_graph(context_biasing_list)
        assert context_graph.num_nodes == 8
        assert context_graph.blank_token == 1024
        assert not context_graph.root.next['▁g'].is_end
        assert context_graph.root.next['▁g'].next['p'].next['u'].is_end
        assert context_graph.root.next['▁g'].next['p'].next['u'].word == 'gpu'
        assert context_graph.root.next['▁g'].next['▁p'].next['▁u'].is_end
        assert context_graph.root.next['▁g'].next['▁p'].next['▁u'].word == 'gpu'


class TestCTCWordSpotter:
    @pytest.mark.unit
    @pytest.mark.with_downloads
    def test_run_word_spotter(self, test_data_dir, conformer_ctc_bpe_model):
        asr_model = conformer_ctc_bpe_model
        audio_file_path = os.path.join(test_data_dir, "asr/test/an4/wav/cen3-mjwl-b.wav")
        target_text = "nineteen"
        target_tokenization = asr_model.tokenizer.text_to_ids(target_text)
        ctc_logprobs = (
            asr_model.transcribe([audio_file_path], batch_size=1, return_hypotheses=True)[0].alignments.cpu().numpy()
        )
        context_biasing_list = [[target_text, [target_tokenization]]]
        context_graph = context_biasing.ContextGraphCTC(blank_id=asr_model.decoding.blank_id)
        context_graph.add_to_graph(context_biasing_list)

        # without context biasing
        ws_results = context_biasing.run_word_spotter(
            ctc_logprobs,
            context_graph,
            asr_model,
            blank_idx=asr_model.decoding.blank_id,
            beam_threshold=5.0,
            cb_weight=0.0,
            ctc_ali_token_weight=0.6,
        )
        assert len(ws_results) == 0

        # with context biasing
        ws_results = context_biasing.run_word_spotter(
            ctc_logprobs,
            context_graph,
            asr_model,
            blank_idx=asr_model.decoding.blank_id,
            beam_threshold=5.0,
            cb_weight=3.0,
            ctc_ali_token_weight=0.6,
        )
        assert len(ws_results) == 1
        assert ws_results[0].word == target_text
        assert ws_results[0].start_frame == 9
        assert ws_results[0].end_frame == 19
        assert round(ws_results[0].score, 4) == 8.9967


class TestContextBiasingUtils:
    @pytest.mark.unit
    @pytest.mark.with_downloads
    def test_merge_alignment_with_ws_hyps(self, conformer_ctc_bpe_model):
        asr_model = conformer_ctc_bpe_model
        blank_idx = asr_model.decoding.blank_id
        ws_results = [WSHyp(word="gpu", score=6.0, start_frame=0, end_frame=2)]

        # ctc argmax predictions
        preds = np.array([120, 29, blank_idx, blank_idx])
        pred_text, raw_text = context_biasing.merge_alignment_with_ws_hyps(
            preds, asr_model, ws_results, decoder_type="ctc", blank_idx=blank_idx,
        )
        assert raw_text == "gp"
        assert pred_text == "gpu"

        # rnnt token predictions
        preds = rnnt_utils.Hypothesis(
            y_sequence=torch.tensor([120, 29]), score=0.0, timestep=torch.tensor([0, 1, 2, 3]),
        )
        pred_text, raw_text = context_biasing.merge_alignment_with_ws_hyps(
            preds, asr_model, ws_results, decoder_type="rnnt", blank_idx=blank_idx,
        )
        assert raw_text == "gp"
        assert pred_text == "gpu"

        # rnnt empty token predictions
        preds = rnnt_utils.Hypothesis(y_sequence=[], score=0.0, timestep=[],)
        pred_text, raw_text = context_biasing.merge_alignment_with_ws_hyps(
            preds, asr_model, ws_results, decoder_type="rnnt", blank_idx=blank_idx,
        )
        assert raw_text == ""
        assert pred_text == "gpu"

    @pytest.mark.unit
    def test_compute_fscore(self):
        recog_manifest = """{"audio_filepath": "test.wav", "duration": 1.0, "text": "a new gpu for nvidia", "pred_text": "a new gpu for invidia"}\n"""
        context_words = ["gpu", "cpu", "nvidia"]
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as f:
            f.write(recog_manifest)
            f.seek(0)
            fscore_stats = context_biasing.compute_fscore(f.name, context_words)
            assert (round(fscore_stats[0], 4), round(fscore_stats[1], 4), round(fscore_stats[2], 4)) == (
                1.0,
                0.5,
                0.6667,
            )
