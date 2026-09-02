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

import torch

from tests.collections.speechlm2.test_salm_automodel import _make_chunking_test_model


def test_salm_manual_model_and_optimizer_state_restore_with_runtime_packed_opt_in(tmp_path):
    previous = _make_chunking_test_model(encoder_chunk_size_seconds=1.0, sampling_rate=2, device="cpu")
    optimizer = torch.optim.AdamW(previous.parameters(), lr=1e-3)
    previous.llm.model.embed_tokens.weight.square().sum().backward()
    optimizer.step()
    checkpoint_path = tmp_path / "previous_salm.ckpt"
    torch.save(
        {"state_dict": previous.state_dict(), "optimizer_states": [optimizer.state_dict()]},
        checkpoint_path,
    )

    resumed = _make_chunking_test_model(encoder_chunk_size_seconds=1.0, sampling_rate=2, device="cpu")
    resumed.cfg["packed_encoder_sequences"] = True
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    resumed.load_state_dict(checkpoint["state_dict"], strict=True)
    resumed_optimizer.load_state_dict(checkpoint["optimizer_states"][0])

    assert set(resumed.state_dict()) == set(previous.state_dict())
    assert len(resumed_optimizer.state) == len(optimizer.state)
    device = next(resumed.parameters()).device
    batch = {
        "audios": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], device=device),
        "audio_lens": torch.tensor([5], dtype=torch.long, device=device),
        "input_ids": torch.tensor([[resumed.audio_locator_tag_id, 10]], dtype=torch.long, device=device),
        "loss_mask": torch.tensor([[False, True]], dtype=torch.bool, device=device),
    }
    resumed.prepare_inputs(batch)
    assert resumed.perception.sequence_packed_calls == 1
