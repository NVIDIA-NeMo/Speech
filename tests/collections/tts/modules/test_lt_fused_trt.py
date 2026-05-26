import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Minimal fake model that mirrors the sub-module structure of LocalTransformerHelper
# ---------------------------------------------------------------------------

class FakeTransformer(nn.Module):
    """2-layer-equivalent single Linear to stand in for the causal Transformer."""
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def reset_cache(self, use_cache=False):
        pass

    def forward(self, x, x_mask):
        return {'output': self.linear(x)}

def make_fake_lt_helper(n_codebooks=4, d_model=64, vocab_size=256, topk=10, temperature=0.7):
    """Return a namespace that matches the attributes LocalTransformerFusedModule reads."""
    from types import SimpleNamespace
    ns = SimpleNamespace()
    ns.local_transformer_in_projection = nn.Linear(d_model, d_model)
    ns.local_transformer = FakeTransformer(d_model)
    ns.local_transformer_audio_out_projection = nn.Linear(d_model, d_model)
    ns.local_transformer_out_projections = nn.ModuleList(
        [nn.Linear(d_model, vocab_size) for _ in range(n_codebooks)]
    )
    ns.audio_embeddings = nn.ModuleList(
        [nn.Embedding(vocab_size, d_model) for _ in range(n_codebooks)]
    )
    ns.audio_in_projection = nn.Linear(d_model, d_model)
    ns.n_codebooks = n_codebooks
    ns.d_model = d_model
    ns.vocab_size = vocab_size
    ns.topk = topk
    ns.temperature = temperature
    return ns

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shape_b1():
    from nemo.collections.tts.modules.magpietts_lt_fused import LocalTransformerFusedModule
    ns = make_fake_lt_helper(n_codebooks=4, d_model=64, vocab_size=256)
    module = LocalTransformerFusedModule(ns, temperature=0.7, topk=10)
    module.eval()
    dec_output = torch.randn(1, 64)
    with torch.no_grad():
        tokens = module(dec_output)
    assert tokens.shape == (1, 4), f"Expected (1, 4), got {tokens.shape}"

def test_output_shape_b2():
    from nemo.collections.tts.modules.magpietts_lt_fused import LocalTransformerFusedModule
    ns = make_fake_lt_helper(n_codebooks=4, d_model=64, vocab_size=256)
    module = LocalTransformerFusedModule(ns, temperature=0.7, topk=10)
    module.eval()
    dec_output = torch.randn(2, 64)
    with torch.no_grad():
        tokens = module(dec_output)
    assert tokens.shape == (2, 4), f"Expected (2, 4), got {tokens.shape}"

def test_token_range():
    from nemo.collections.tts.modules.magpietts_lt_fused import LocalTransformerFusedModule
    ns = make_fake_lt_helper(n_codebooks=4, d_model=64, vocab_size=256)
    module = LocalTransformerFusedModule(ns, temperature=0.7, topk=10)
    module.eval()
    dec_output = torch.randn(2, 64)
    with torch.no_grad():
        tokens = module(dec_output)
    assert tokens.min().item() >= 0, "Token below 0"
    assert tokens.max().item() < 256, "Token >= vocab_size"

def test_output_is_integer_dtype():
    from nemo.collections.tts.modules.magpietts_lt_fused import LocalTransformerFusedModule
    ns = make_fake_lt_helper(n_codebooks=4, d_model=64, vocab_size=256)
    module = LocalTransformerFusedModule(ns, temperature=0.7, topk=10)
    module.eval()
    dec_output = torch.randn(1, 64)
    with torch.no_grad():
        tokens = module(dec_output)
    assert tokens.dtype in (torch.int32, torch.int64, torch.long), \
        f"Expected integer dtype, got {tokens.dtype}"
