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

import torch

from nemo.collections.speechlm2.parts.label_prep import maybe_prepend_prompt_tokens


def test_maybe_prepend_prompt_tokens_with_source_tokens():
    """Test that prompt tokens are correctly prepended to all sequences and lengths are updated."""
    B, T_src, T_tgt, H = 2, 6, 6, 4
    PAD = 0
    prompt_len_0, prompt_len_1 = 3, 2

    # Prompt token IDs (will be passed through embed_fn)
    max_prompt_len = max(prompt_len_0, prompt_len_1)
    prompt_tokens = torch.full((B, max_prompt_len), PAD, dtype=torch.long)
    prompt_tokens[0, :prompt_len_0] = torch.tensor([10, 11, 12])
    prompt_tokens[1, :prompt_len_1] = torch.tensor([20, 21])

    # Use a simple embedding: token_id -> [token_id, token_id, token_id, token_id]
    def embed_fn(token_ids):
        return token_ids.unsqueeze(-1).expand(-1, -1, H).float()

    # Source encoded (audio features)
    source_encoded = torch.arange(B * T_src * H).reshape(B, T_src, H).float()
    source_encoded_lens = torch.tensor([5, 4])

    # Target tokens
    target_tokens = torch.tensor(
        [
            [1, 100, 101, 102, 2, 0],  # BOS, tokens, EOS, PAD
            [1, 200, 201, 2, 0, 0],
        ]
    )
    target_token_lens = torch.tensor([5, 4])

    # Source tokens (for ASR head)
    source_tokens = torch.tensor(
        [
            [1, 50, 51, 52, 2, 0],
            [1, 60, 61, 2, 0, 0],
        ]
    )
    source_token_lens = torch.tensor([5, 4])

    batch = {
        "prompt_tokens": prompt_tokens,
        "prompt_token_lens": torch.tensor([prompt_len_0, prompt_len_1]),
        "target_tokens": target_tokens,
        "target_token_lens": target_token_lens,
        "source_tokens": source_tokens,
        "source_token_lens": source_token_lens,
    }

    new_source_encoded, new_source_encoded_lens, new_target_tokens = maybe_prepend_prompt_tokens(
        batch=batch,
        embed_fn=embed_fn,
        source_encoded=source_encoded,
        source_encoded_lens=source_encoded_lens,
        text_pad_id=PAD,
    )

    # Check output shapes are extended by max_prompt_len
    assert new_source_encoded.shape == (B, max_prompt_len + T_src, H)
    assert new_target_tokens.shape == (B, max_prompt_len + T_tgt)
    assert batch["source_tokens"].shape == (B, max_prompt_len + T_tgt)

    # Check lengths are updated: original_len + prompt_len
    assert new_source_encoded_lens[0].item() == 5 + prompt_len_0
    assert new_source_encoded_lens[1].item() == 4 + prompt_len_1
    assert batch["target_token_lens"][0].item() == 5 + prompt_len_0
    assert batch["target_token_lens"][1].item() == 4 + prompt_len_1
    assert batch["source_token_lens"][0].item() == 5 + prompt_len_0
    assert batch["source_token_lens"][1].item() == 4 + prompt_len_1

    # Check prompt embeddings are at the beginning of source_encoded
    # embed_fn maps token_id -> [token_id]*H, so prompt region should match
    for h in range(H):
        assert new_source_encoded[0, 0, h].item() == 10.0
        assert new_source_encoded[0, 1, h].item() == 11.0
        assert new_source_encoded[0, 2, h].item() == 12.0
        assert new_source_encoded[1, 0, h].item() == 20.0
        assert new_source_encoded[1, 1, h].item() == 21.0

    # Check original audio features follow the prompt
    for t in range(5):  # source_encoded_lens[0] was 5
        assert torch.equal(new_source_encoded[0, prompt_len_0 + t], source_encoded[0, t])
    for t in range(4):  # source_encoded_lens[1] was 4
        assert torch.equal(new_source_encoded[1, prompt_len_1 + t], source_encoded[1, t])

    # Check target tokens are shifted by prompt_len
    assert new_target_tokens[0, :prompt_len_0].tolist() == [PAD] * prompt_len_0
    assert new_target_tokens[0, prompt_len_0 : prompt_len_0 + 5].tolist() == [1, 100, 101, 102, 2]
    assert new_target_tokens[1, :prompt_len_1].tolist() == [PAD] * prompt_len_1
    assert new_target_tokens[1, prompt_len_1 : prompt_len_1 + 4].tolist() == [1, 200, 201, 2]

    # Check source tokens are shifted by prompt_len
    assert batch["source_tokens"][0, :prompt_len_0].tolist() == [PAD] * prompt_len_0
    assert batch["source_tokens"][0, prompt_len_0 : prompt_len_0 + 5].tolist() == [1, 50, 51, 52, 2]
    assert batch["source_tokens"][1, :prompt_len_1].tolist() == [PAD] * prompt_len_1
    assert batch["source_tokens"][1, prompt_len_1 : prompt_len_1 + 4].tolist() == [1, 60, 61, 2]


def test_maybe_prepend_prompt_tokens_no_prompt():
    """Test that without prompt_tokens in batch, inputs are returned unchanged."""
    B, T_src, H = 1, 4, 4
    source_encoded = torch.randn(B, T_src, H)
    source_encoded_lens = torch.tensor([3])
    target_tokens = torch.tensor([[1, 100, 2, 0]])

    batch = {
        "target_tokens": target_tokens,
        "target_token_lens": torch.tensor([3]),
    }

    out_encoded, out_lens, out_tokens = maybe_prepend_prompt_tokens(
        batch=batch,
        embed_fn=lambda x: x.unsqueeze(-1).expand(-1, -1, H).float(),
        source_encoded=source_encoded,
        source_encoded_lens=source_encoded_lens,
        text_pad_id=0,
    )

    assert torch.equal(out_encoded, source_encoded)
    assert torch.equal(out_lens, source_encoded_lens)
    assert torch.equal(out_tokens, target_tokens)
