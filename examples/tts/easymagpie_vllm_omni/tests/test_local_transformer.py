# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""Validity tests for the vLLM-Omni EasyMagpieTTS local transformer.

The headline test is a **numerical parity check against the reference NeMo
implementation** (``transformer_2501.Transformer`` + the projection / embedding
heads, exactly as wired in ``EasyMagpieTTSInferenceModel``): random NeMo weights
are copied 1:1 into the vLLM ``EasyMagpieCodePredictor`` and both stacks are run
teacher-forced on identical inputs; the per-codebook logits must match to fp32
tolerance with identical argmax. This is the pytest port of
``debug_local_transformer.py`` and guards against the re-implementation silently
drifting from the training-time math.

The remaining tests assert the autoregressive sampler's contract (output shape /
dtype / value range, forbidden-token masking, and seeded determinism).

Everything runs as plain PyTorch on CPU via the tiny stand-in config from
``conftest.py`` — no model directory, no vLLM engine, no GPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
transformer_2501 = pytest.importorskip("nemo.collections.tts.modules.transformer_2501")

from conftest import build_vllm_config  # noqa: E402
from easymagpie_vllm_omni.config import EasyMagpieOmniArch  # noqa: E402
from easymagpie_vllm_omni.local_transformer import EasyMagpieCodePredictor  # noqa: E402
from torch import nn  # noqa: E402

# Two arch profiles: one where all widths are equal (in/out projections are
# Identity, matching the real checkpoint) and one where they differ (projections
# are real Linears) — so the weight-copy + parity covers both code paths.
ARCH_PROFILES = {
    "equal_dims": dict(
        hidden_dim=64,
        embedding_dim=64,
        audio_embedding_dim=64,
        local_transformer_hidden_dim=64,
        local_transformer_n_heads=4,
    ),
    "mixed_dims": dict(
        hidden_dim=64,
        embedding_dim=64,
        audio_embedding_dim=48,
        local_transformer_hidden_dim=80,
        local_transformer_n_heads=4,
    ),
}


class NeMoLocalTransformerStack(nn.Module):
    """Reference NeMo local-transformer submodules, named to match the vLLM code predictor.

    Mirrors the wiring in ``EasyMagpieTTSInferenceModel.__init__`` (the
    ``local_transformer*`` / ``audio_*`` heads). Attribute names match
    :class:`EasyMagpieCodePredictor` so a state-dict copy is 1:1.
    """

    def __init__(self, arch: EasyMagpieOmniArch) -> None:
        super().__init__()
        self.n_codebooks = arch.num_stacked_codebooks
        self.num_all_tokens = arch.num_all_tokens_per_codebook
        embedding_dim = arch.embedding_dim
        audio_dim = arch.audio_embedding_dim
        lt_hidden = arch.local_transformer_hidden_dim

        self.audio_embeddings = nn.ModuleList(
            [nn.Embedding(self.num_all_tokens, audio_dim) for _ in range(self.n_codebooks)]
        )
        self.audio_in_projection = nn.Linear(audio_dim, embedding_dim) if audio_dim != embedding_dim else nn.Identity()
        self.local_transformer_in_projection = (
            nn.Linear(embedding_dim, lt_hidden) if lt_hidden != embedding_dim else nn.Identity()
        )
        self.local_transformer = transformer_2501.Transformer(
            n_layers=arch.local_transformer_n_layers,
            d_model=lt_hidden,
            d_ffn=lt_hidden * 4,
            sa_n_heads=arch.local_transformer_n_heads,
            kernel_size=1,
            is_causal=True,
            max_length_causal_mask=self.n_codebooks + 2,
            use_learnable_pos_emb=True,
        )
        self.local_transformer_audio_out_projection = (
            nn.Linear(lt_hidden, audio_dim) if audio_dim != lt_hidden else nn.Identity()
        )
        self.local_transformer_out_projections = nn.ModuleList(
            [nn.Linear(audio_dim, self.num_all_tokens) for _ in range(self.n_codebooks)]
        )

    @torch.no_grad()
    def teacher_forced_logits(self, dec_hidden: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Per-codebook logits given a hidden state and teacher-forced previous codes.

        Replicates ``LocalTransformerHelper.compute_logits`` (AR layout): the input
        sequence is ``[dec_hidden, emb(code_0), ..., emb(code_{N-1})]``; row ``k`` of
        the causal output predicts codebook ``k``, and the trailing row is dropped.
        """
        seq = [dec_hidden]
        for k in range(self.n_codebooks):
            seq.append(self.audio_in_projection(self.audio_embeddings[k](codes[:, k])))
        x = torch.stack(seq, dim=1)  # (T, N+1, embedding_dim)
        x = self.local_transformer_in_projection(x)  # (T, N+1, lt_hidden)
        mask = torch.ones(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
        out = self.local_transformer(x, mask)["output"][:, :-1, :]  # (T, N, lt_hidden)
        out = self.local_transformer_audio_out_projection(out)  # (T, N, audio_dim)
        logits = [self.local_transformer_out_projections[k](out[:, k, :]) for k in range(self.n_codebooks)]
        return torch.stack(logits, dim=1)  # (T, N, vocab)


@torch.no_grad()
def _vllm_teacher_forced_logits(
    cp: EasyMagpieCodePredictor, dec_hidden: torch.Tensor, codes: torch.Tensor
) -> torch.Tensor:
    """Per-codebook logits from the vLLM code predictor, teacher-forced.

    Mirrors :meth:`EasyMagpieCodePredictor.generate_codes` buffer layout (``N``
    rows; row 0 = ``in_proj(dec_hidden)``, row ``k+1`` = projected embedding of
    ``codes[:, k]``), but reads the logits for every row instead of sampling.
    """
    num_tokens = dec_hidden.shape[0]
    n = cp.num_codebooks
    lt_hidden = cp.lt_hidden
    buf = torch.zeros(num_tokens, n, lt_hidden, dtype=dec_hidden.dtype, device=dec_hidden.device)
    buf[:, 0, :] = cp.local_transformer_in_projection(dec_hidden)
    for k in range(n - 1):
        emb = cp.audio_in_projection(cp.audio_embeddings[k](codes[:, k]))
        buf[:, k + 1, :] = cp.local_transformer_in_projection(emb)
    hidden = cp.local_transformer(buf)  # (T, N, lt_hidden)
    logits = []
    for k in range(n):
        row = cp.local_transformer_audio_out_projection(hidden[:, k, :])
        logits.append(cp.local_transformer_out_projections[k](row))
    return torch.stack(logits, dim=1)  # (T, N, vocab)


def _copy_nemo_into_vllm(nemo: NeMoLocalTransformerStack, cp: EasyMagpieCodePredictor) -> None:
    """Copy every vLLM code-predictor parameter from the matching NeMo parameter (names align 1:1)."""
    nemo_sd = nemo.state_dict()
    missing = []
    for name, param in cp.named_parameters():
        if name in nemo_sd:
            src = nemo_sd[name]
            # The FFN ships as kernel-1 Conv1d (``[out, in, 1]``) in NeMo but is a
            # plain ``nn.Linear`` (``[out, in]``) here; squeeze the conv dim to
            # match (mirrors ``EasyMagpieTTS.load_weights``).
            if src.ndim == param.ndim + 1 and src.shape[-1] == 1:
                src = src.squeeze(-1)
            assert param.shape == src.shape, f"shape mismatch {name}"
            param.data.copy_(src.to(param.dtype))
        else:
            missing.append(name)
    assert not missing, f"vLLM params with no NeMo counterpart: {missing}"


def _build_pair(profile_kwargs: dict, seed: int = 0):
    """Build a (code_predictor, nemo_stack, arch) triple with NeMo weights copied in."""
    cfg = build_vllm_config(**profile_kwargs)
    arch = EasyMagpieOmniArch.from_hf_config(cfg.model_config.hf_config)

    cp = EasyMagpieCodePredictor(vllm_config=cfg, prefix="code_predictor").eval()
    cp.init_forbidden_mask()

    gen = torch.Generator().manual_seed(seed)
    nemo = NeMoLocalTransformerStack(arch).float().eval()
    with torch.no_grad():
        for prm in nemo.parameters():
            prm.copy_(torch.empty(prm.shape).normal_(0.0, 0.02, generator=gen))
    _copy_nemo_into_vllm(nemo, cp)
    return cp, nemo, arch


@pytest.mark.unit
@pytest.mark.parametrize("profile", list(ARCH_PROFILES), ids=list(ARCH_PROFILES))
def test_local_transformer_matches_nemo(profile):
    """vLLM re-implementation must equal the NeMo reference in fp32 (teacher-forced)."""
    cp, nemo, arch = _build_pair(ARCH_PROFILES[profile])

    torch.manual_seed(1234)
    num_tokens = 6
    dec_hidden = torch.randn(num_tokens, arch.hidden_dim)
    codes = torch.randint(0, arch.codebook_size, (num_tokens, arch.num_stacked_codebooks))

    nemo_logits = nemo.teacher_forced_logits(dec_hidden, codes)
    vllm_logits = _vllm_teacher_forced_logits(cp, dec_hidden, codes)

    assert vllm_logits.shape == nemo_logits.shape
    max_abs_diff = (vllm_logits - nemo_logits).abs().max().item()
    argmax_mismatch = (vllm_logits.argmax(-1) != nemo_logits.argmax(-1)).sum().item()
    assert max_abs_diff < 1e-3, f"max abs diff too large: {max_abs_diff:.3e}"
    assert argmax_mismatch == 0, f"{argmax_mismatch} argmax mismatches"


@pytest.mark.unit
def test_generate_codes_shape_dtype_and_range():
    """``generate_codes`` returns valid (num_tokens, num_codebooks) int64 codes within vocab."""
    cp, _, arch = _build_pair(ARCH_PROFILES["equal_dims"])
    num_tokens = 5

    torch.manual_seed(0)
    codes = cp.generate_codes(torch.randn(num_tokens, arch.hidden_dim))

    assert codes.shape == (num_tokens, arch.num_stacked_codebooks)
    assert codes.dtype == torch.long
    assert codes.min().item() >= 0
    assert codes.max().item() < arch.num_all_tokens_per_codebook


@pytest.mark.unit
def test_generate_codes_respects_forbidden_mask():
    """With argmax sampling, forbidden special tokens are never emitted (only EOS stays reachable)."""
    cp, _, arch = _build_pair(ARCH_PROFILES["equal_dims"])
    cp.temperature = 0.0  # argmax over masked logits

    torch.manual_seed(0)
    codes = cp.generate_codes(torch.randn(7, arch.hidden_dim))

    # Allowed = real codebook tokens [0, codebook_size) plus the audio EOS id.
    allowed = (codes < arch.codebook_size) | (codes == arch.audio_eos_id)
    assert allowed.all(), f"sampled forbidden tokens: {sorted(set(codes[~allowed].tolist()))}"


@pytest.mark.unit
def test_generate_codes_deterministic_with_seed():
    """Same seed + same input ⇒ identical sampled codes (sampler is RNG-driven, no host state)."""
    cp, _, arch = _build_pair(ARCH_PROFILES["equal_dims"])
    dec_hidden = torch.randn(4, arch.hidden_dim)

    torch.manual_seed(7)
    first = cp.generate_codes(dec_hidden)
    torch.manual_seed(7)
    second = cp.generate_codes(dec_hidden)

    assert torch.equal(first, second)
