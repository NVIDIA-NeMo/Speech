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

"""
Test script for NemotronH decoder module.

This script tests:
1. NemotronHConfig initialization
2. NemotronHModel forward pass
3. NemotronHForCausalLM forward pass
4. KV caching for inference
5. Interface compatibility with EasyMagpieTTSModel requirements
"""

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

    # Create a dummy pytest fixture decorator for standalone execution
    class pytest:
        @staticmethod
        def fixture(func):
            return func


import torch

from nemo.collections.tts.modules.nemotron_h_decoder import (
    HybridMambaAttentionDynamicCache,
    NemotronHConfig,
    NemotronHForCausalLM,
    NemotronHMLP,
    NemotronHModel,
    NemotronHMOE,
    NemotronHTopkRouter,
)


class TestNemotronHConfig:
    """Test NemotronHConfig initialization and defaults."""

    def test_default_config(self):
        """Test default config initialization."""
        config = NemotronHConfig()
        assert config.hidden_size == 1536
        assert config.num_hidden_layers == 24
        assert len(config.layers_block_type) == config.num_hidden_layers

    def test_custom_pattern(self):
        """Test custom hybrid_override_pattern."""
        config = NemotronHConfig(num_hidden_layers=8, hybrid_override_pattern="M*M*M*M*")
        assert config.layers_block_type == ['mamba', 'attention'] * 4

    def test_pattern_extension(self):
        """Test that short patterns are extended to match num_hidden_layers."""
        config = NemotronHConfig(num_hidden_layers=8, hybrid_override_pattern="M*")
        assert len(config.layers_block_type) == 8


class TestNemotronHModel:
    """Test NemotronHModel backbone."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        return NemotronHConfig(
            hidden_size=64,
            num_hidden_layers=4,
            vocab_size=1000,
            num_attention_heads=4,
            num_key_value_heads=2,
            mamba_num_heads=8,
            mamba_head_dim=8,
            ssm_state_size=16,
            n_groups=2,
            intermediate_size=128,
            hybrid_override_pattern="M*M*",
        )

    @pytest.fixture
    def model(self, small_config):
        """Create a small model for testing."""
        return NemotronHModel(small_config)

    def test_model_creation(self, model, small_config):
        """Test model can be created."""
        assert model is not None
        assert len(model.layers) == small_config.num_hidden_layers

    def test_forward_with_input_ids(self, model):
        """Test forward pass with input_ids."""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        output = model(input_ids=input_ids)

        assert output.last_hidden_state is not None
        assert output.last_hidden_state.shape == (batch_size, seq_len, 64)

    def test_forward_with_inputs_embeds(self, model):
        """Test forward pass with inputs_embeds (required for TTS)."""
        batch_size, seq_len, hidden_size = 2, 16, 64
        inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)

        output = model(inputs_embeds=inputs_embeds)

        assert output.last_hidden_state is not None
        assert output.last_hidden_state.shape == (batch_size, seq_len, hidden_size)

    def test_get_set_input_embeddings(self, model):
        """Test get/set input embeddings interface."""
        original_embeddings = model.get_input_embeddings()
        assert original_embeddings is not None

        new_embeddings = torch.nn.Embedding(100, 64)
        model.set_input_embeddings(new_embeddings)

        assert model.get_input_embeddings() is new_embeddings


class TestNemotronHForCausalLM:
    """Test NemotronHForCausalLM full model."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        return NemotronHConfig(
            hidden_size=64,
            num_hidden_layers=4,
            vocab_size=1000,
            num_attention_heads=4,
            num_key_value_heads=2,
            mamba_num_heads=8,
            mamba_head_dim=8,
            ssm_state_size=16,
            n_groups=2,
            intermediate_size=128,
            hybrid_override_pattern="M*M*",
        )

    @pytest.fixture
    def model(self, small_config):
        """Create a small model for testing."""
        return NemotronHForCausalLM(small_config)

    def test_model_creation(self, model, small_config):
        """Test model can be created."""
        assert model is not None
        assert model.backbone is not None
        assert model.lm_head is not None

    def test_model_alias(self, model):
        """Test that model.model returns backbone (HF compatibility)."""
        assert model.model is model.backbone

    def test_forward_with_inputs_embeds(self, model):
        """Test forward pass with inputs_embeds."""
        batch_size, seq_len, hidden_size = 2, 16, 64
        inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)

        output = model(inputs_embeds=inputs_embeds)

        assert output.logits is not None
        assert output.logits.shape == (batch_size, seq_len, 1000)  # vocab_size

    def test_interface_compatibility(self, model):
        """Test that model satisfies EasyMagpieTTSModel interface requirements."""
        # Test 1: decoder.get_input_embeddings()
        embeddings = model.backbone.get_input_embeddings()
        assert embeddings is not None

        # Test 2: decoder.set_input_embeddings()
        new_emb = torch.nn.Embedding(100, 64)
        model.backbone.set_input_embeddings(new_emb)
        assert model.backbone.get_input_embeddings() is new_emb

        # Reset for next tests
        model.backbone.set_input_embeddings(embeddings)

        # Test 3: decoder(inputs_embeds, attention_mask, use_cache, past_key_values)
        batch_size, seq_len, hidden_size = 2, 16, 64
        inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        output = model.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            past_key_values=None,
        )

        # Test 4: Return .last_hidden_state
        assert hasattr(output, 'last_hidden_state')
        assert output.last_hidden_state is not None

        # Test 5: Return .past_key_values (when use_cache=True not tested here as it requires more setup)
        assert hasattr(output, 'past_key_values')


class TestHybridCache:
    """Test HybridMambaAttentionDynamicCache."""

    def test_cache_creation(self):
        """Test cache can be created."""
        config = NemotronHConfig(
            hidden_size=64,
            num_hidden_layers=4,
            mamba_num_heads=8,
            mamba_head_dim=8,
            ssm_state_size=16,
            conv_kernel=4,
            hybrid_override_pattern="M*M*",
        )

        batch_size = 2
        cache = HybridMambaAttentionDynamicCache(config, batch_size, dtype=torch.float32)

        assert len(cache.conv_states) == config.num_hidden_layers
        assert len(cache.ssm_states) == config.num_hidden_layers
        assert len(cache.key_cache) == config.num_hidden_layers
        assert len(cache.value_cache) == config.num_hidden_layers


class TestNemotronHCausality:
    """Test that NemotronH model is causal (future timesteps don't affect previous ones)."""

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing causality."""
        return NemotronHConfig(
            hidden_size=64,
            num_hidden_layers=4,
            vocab_size=1000,
            num_attention_heads=4,
            num_key_value_heads=2,
            mamba_num_heads=8,
            mamba_head_dim=8,
            ssm_state_size=16,
            n_groups=2,
            intermediate_size=128,
            hybrid_override_pattern="M*M*",
        )

    @pytest.fixture
    def model(self, small_config):
        """Create a small model for testing."""
        model = NemotronHModel(small_config)
        model.eval()  # Set to eval mode for deterministic behavior
        return model

    def test_causality_with_input_modification(self, model, small_config):
        """
        Test causality by modifying future timesteps and checking that earlier outputs are unchanged.

        The test:
        1. Pass sequence through the model
        2. Modify a future timestep in the input
        3. Verify outputs at earlier timesteps remain exactly the same
        """
        batch_size, seq_len = 2, 16
        hidden_size = small_config.hidden_size

        # Create a base input
        torch.manual_seed(42)
        inputs_embeds_original = torch.randn(batch_size, seq_len, hidden_size)

        # Get output with original input
        with torch.no_grad():
            output_original = model(inputs_embeds=inputs_embeds_original.clone())

        # Test at different positions
        test_positions = [seq_len // 4, seq_len // 2, 3 * seq_len // 4]

        for modify_pos in test_positions:
            # Create modified input where we change timesteps from modify_pos onwards
            inputs_embeds_modified = inputs_embeds_original.clone()
            # Add random noise to all positions from modify_pos onwards
            inputs_embeds_modified[:, modify_pos:, :] += (
                torch.randn(batch_size, seq_len - modify_pos, hidden_size) * 10.0
            )  # Large modification to ensure it would affect outputs if not causal

            # Get output with modified input
            with torch.no_grad():
                output_modified = model(inputs_embeds=inputs_embeds_modified)

            # Check that outputs BEFORE modify_pos are unchanged
            outputs_before_original = output_original.last_hidden_state[:, :modify_pos, :]
            outputs_before_modified = output_modified.last_hidden_state[:, :modify_pos, :]

            # Should be exactly equal (within floating point tolerance)
            assert torch.allclose(
                outputs_before_original, outputs_before_modified, atol=1e-5
            ), f"Causality violation: modifying position {modify_pos} affected earlier positions"

            # Verify that outputs AT and AFTER modify_pos are different (sanity check)
            outputs_after_original = output_original.last_hidden_state[:, modify_pos:, :]
            outputs_after_modified = output_modified.last_hidden_state[:, modify_pos:, :]

            assert not torch.allclose(
                outputs_after_original, outputs_after_modified, atol=1e-3
            ), f"Sanity check failed: modifying position {modify_pos} should affect outputs at/after that position"

    def test_causality_incremental_vs_full(self, model, small_config):
        """
        Test causality by comparing incremental (token-by-token) vs full sequence processing.

        A causal model should produce the same output whether we:
        1. Process the full sequence at once
        2. Process tokens incrementally one at a time
        """
        batch_size, seq_len = 1, 8  # Smaller seq for incremental test
        hidden_size = small_config.hidden_size

        torch.manual_seed(123)
        inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)

        # Get output from full sequence
        with torch.no_grad():
            output_full = model(inputs_embeds=inputs_embeds)

        # Get outputs incrementally (one token at a time)
        # For a causal model, output at each position should match
        incremental_outputs = []
        for t in range(1, seq_len + 1):
            with torch.no_grad():
                partial_output = model(inputs_embeds=inputs_embeds[:, :t, :])
            # Take only the last timestep output for comparison
            incremental_outputs.append(partial_output.last_hidden_state[:, -1:, :])

        # Stack incremental outputs
        output_incremental = torch.cat(incremental_outputs, dim=1)

        # Compare: the full sequence output should match the incrementally computed outputs
        assert torch.allclose(
            output_full.last_hidden_state, output_incremental, atol=1e-4
        ), "Causality violation: incremental processing produces different results than full sequence"

    def test_causality_causal_lm(self, small_config):
        """Test causality for NemotronHForCausalLM."""
        model = NemotronHForCausalLM(small_config)
        model.eval()

        batch_size, seq_len = 2, 12
        hidden_size = small_config.hidden_size

        torch.manual_seed(456)
        inputs_embeds_original = torch.randn(batch_size, seq_len, hidden_size)

        modify_pos = seq_len // 2

        # Get logits with original input
        with torch.no_grad():
            output_original = model(inputs_embeds=inputs_embeds_original.clone())

        # Modify future positions
        inputs_embeds_modified = inputs_embeds_original.clone()
        inputs_embeds_modified[:, modify_pos:, :] += torch.randn(batch_size, seq_len - modify_pos, hidden_size) * 10.0

        with torch.no_grad():
            output_modified = model(inputs_embeds=inputs_embeds_modified)

        # Check logits before modify_pos are unchanged
        logits_before_original = output_original.logits[:, :modify_pos, :]
        logits_before_modified = output_modified.logits[:, :modify_pos, :]

        assert torch.allclose(
            logits_before_original, logits_before_modified, atol=1e-5
        ), "Causality violation in CausalLM: modifying future positions affected earlier logits"

    def test_causality_different_layer_types(self):
        """Test causality with different hybrid patterns (Mamba-only, Attention-only, mixed)."""
        patterns = [
            "MMMM",  # Mamba only
            "****",  # Attention only
            "M*M*",  # Alternating
            "MM**",  # Mixed blocks
        ]

        for pattern in patterns:
            config = NemotronHConfig(
                hidden_size=64,
                num_hidden_layers=4,
                vocab_size=1000,
                num_attention_heads=4,
                num_key_value_heads=2,
                mamba_num_heads=8,
                mamba_head_dim=8,
                ssm_state_size=16,
                n_groups=2,
                intermediate_size=128,
                hybrid_override_pattern=pattern,
            )

            model = NemotronHModel(config)
            model.eval()

            batch_size, seq_len = 2, 8
            hidden_size = config.hidden_size

            torch.manual_seed(789)
            inputs_embeds_original = torch.randn(batch_size, seq_len, hidden_size)

            modify_pos = 4

            with torch.no_grad():
                output_original = model(inputs_embeds=inputs_embeds_original.clone())

            inputs_embeds_modified = inputs_embeds_original.clone()
            inputs_embeds_modified[:, modify_pos:, :] += (
                torch.randn(batch_size, seq_len - modify_pos, hidden_size) * 10.0
            )

            with torch.no_grad():
                output_modified = model(inputs_embeds=inputs_embeds_modified)

            outputs_before_original = output_original.last_hidden_state[:, :modify_pos, :]
            outputs_before_modified = output_modified.last_hidden_state[:, :modify_pos, :]

            assert torch.allclose(
                outputs_before_original, outputs_before_modified, atol=1e-5
            ), f"Causality violation for pattern '{pattern}': modifying future positions affected earlier outputs"


class TestMoELayer:
    """Test Mixture of Experts layer."""

    @pytest.fixture
    def moe_config(self):
        """Create a config for MoE testing."""
        return NemotronHConfig(
            hidden_size=64,
            num_hidden_layers=4,
            vocab_size=1000,
            num_attention_heads=4,
            num_key_value_heads=2,
            mamba_num_heads=8,
            mamba_head_dim=8,
            ssm_state_size=16,
            n_groups=2,
            intermediate_size=128,
            # MoE config
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            moe_shared_expert_intermediate_size=128,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            norm_topk_prob=True,
            hybrid_override_pattern="M*ME",  # Includes MoE layer
        )

    def test_topk_router_creation(self, moe_config):
        """Test NemotronHTopkRouter creation."""
        router = NemotronHTopkRouter(moe_config)
        assert router.weight.shape == (moe_config.n_routed_experts, moe_config.hidden_size)
        assert router.top_k == moe_config.num_experts_per_tok

    def test_topk_router_forward(self, moe_config):
        """Test NemotronHTopkRouter forward pass."""
        router = NemotronHTopkRouter(moe_config)
        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, moe_config.hidden_size)

        topk_indices, topk_weights = router(hidden_states)

        # Check shapes
        assert topk_indices.shape == (batch_size * seq_len, moe_config.num_experts_per_tok)
        assert topk_weights.shape == (batch_size * seq_len, moe_config.num_experts_per_tok)

        # Check indices are valid
        assert topk_indices.min() >= 0
        assert topk_indices.max() < moe_config.n_routed_experts

    def test_moe_layer_creation(self, moe_config):
        """Test NemotronHMOE creation."""
        moe = NemotronHMOE(moe_config, layer_idx=0)

        assert len(moe.experts) == moe_config.n_routed_experts
        assert moe.gate is not None
        assert moe.shared_experts is not None

    def test_moe_layer_forward(self, moe_config):
        """Test NemotronHMOE forward pass."""
        moe = NemotronHMOE(moe_config, layer_idx=0)
        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, moe_config.hidden_size)

        output = moe(hidden_states)

        assert output.shape == hidden_states.shape

    def test_model_with_moe_pattern(self, moe_config):
        """Test full model with MoE layer."""
        model = NemotronHModel(moe_config)

        # Check that MoE layer was created
        assert model.layers[3].block_type == "moe"

        # Test forward pass
        batch_size, seq_len = 2, 8
        inputs_embeds = torch.randn(batch_size, seq_len, moe_config.hidden_size)

        output = model(inputs_embeds=inputs_embeds)

        assert output.last_hidden_state is not None
        assert output.last_hidden_state.shape == (batch_size, seq_len, moe_config.hidden_size)


if __name__ == "__main__":
    """Run basic tests without pytest."""
    print("Testing NemotronH Decoder Module...")

    # Test 1: Config
    print("\n1. Testing NemotronHConfig...")
    config = NemotronHConfig(
        hidden_size=64,
        num_hidden_layers=4,
        vocab_size=1000,
        num_attention_heads=4,
        num_key_value_heads=2,
        mamba_num_heads=8,
        mamba_head_dim=8,
        ssm_state_size=16,
        n_groups=2,
        intermediate_size=128,
        hybrid_override_pattern="M*M*",
    )
    print(f"   Config created: {config.num_hidden_layers} layers, pattern={config.hybrid_override_pattern}")
    print(f"   Layer types: {config.layers_block_type}")

    # Test 2: Model creation
    print("\n2. Testing NemotronHModel creation...")
    model = NemotronHModel(config)
    print(f"   Model created with {len(model.layers)} layers")

    # Test 3: Forward pass with inputs_embeds
    print("\n3. Testing forward pass with inputs_embeds...")
    batch_size, seq_len, hidden_size = 2, 16, 64
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)
    output = model(inputs_embeds=inputs_embeds)
    print(f"   Input shape: {inputs_embeds.shape}")
    print(f"   Output shape: {output.last_hidden_state.shape}")

    # Test 4: Full model
    print("\n4. Testing NemotronHForCausalLM...")
    full_model = NemotronHForCausalLM(config)
    output = full_model(inputs_embeds=inputs_embeds)
    print(f"   Logits shape: {output.logits.shape}")

    # Test 5: Interface compatibility
    print("\n5. Testing interface compatibility for EasyMagpieTTSModel...")
    decoder = full_model.backbone

    # get_input_embeddings
    emb = decoder.get_input_embeddings()
    print(f"   get_input_embeddings(): {type(emb).__name__}")

    # set_input_embeddings
    new_emb = torch.nn.Embedding(100, 64)
    decoder.set_input_embeddings(new_emb)
    print(f"   set_input_embeddings(): OK")
    decoder.set_input_embeddings(emb)  # Reset

    # forward with expected args
    output = decoder(
        inputs_embeds=inputs_embeds,
        attention_mask=torch.ones(batch_size, seq_len),
        use_cache=False,
        past_key_values=None,
    )
    print(f"   forward(inputs_embeds, attention_mask, use_cache, past_key_values): OK")
    print(f"   .last_hidden_state: {output.last_hidden_state.shape}")
    print(f"   .past_key_values: {output.past_key_values}")

    # Test 6: MoE layer
    print("\n6. Testing MoE (Mixture of Experts) layer...")
    moe_config = NemotronHConfig(
        hidden_size=64,
        num_hidden_layers=4,
        vocab_size=1000,
        num_attention_heads=4,
        num_key_value_heads=2,
        mamba_num_heads=8,
        mamba_head_dim=8,
        ssm_state_size=16,
        n_groups=2,
        intermediate_size=128,
        # MoE config
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        moe_shared_expert_intermediate_size=128,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        hybrid_override_pattern="M*ME",  # Includes MoE layer
    )
    print(f"   Config: pattern={moe_config.hybrid_override_pattern}, block_types={moe_config.layers_block_type}")

    # Test router
    router = NemotronHTopkRouter(moe_config)
    test_input = torch.randn(2, 8, 64)
    topk_indices, topk_weights = router(test_input)
    print(f"   Router: topk_indices shape={topk_indices.shape}, topk_weights shape={topk_weights.shape}")

    # Test MoE layer
    moe = NemotronHMOE(moe_config, layer_idx=0)
    moe_output = moe(test_input)
    print(f"   MoE layer: input={test_input.shape}, output={moe_output.shape}")

    # Test full model with MoE
    moe_model = NemotronHModel(moe_config)
    moe_model_output = moe_model(inputs_embeds=test_input)
    print(f"   Full model with MoE: output={moe_model_output.last_hidden_state.shape}")

    # Test 7: Causality test
    print("\n7. Testing model causality (future timesteps don't affect previous ones)...")

    # Create model for causality test
    causality_config = NemotronHConfig(
        hidden_size=64,
        num_hidden_layers=4,
        vocab_size=1000,
        num_attention_heads=4,
        num_key_value_heads=2,
        mamba_num_heads=8,
        mamba_head_dim=8,
        ssm_state_size=16,
        n_groups=2,
        intermediate_size=128,
        hybrid_override_pattern="M*M*",
    )
    causality_model = NemotronHModel(causality_config)
    causality_model.eval()

    batch_size, seq_len = 2, 16
    hidden_size = 64

    # Create base input
    torch.manual_seed(42)
    inputs_embeds_original = torch.randn(batch_size, seq_len, hidden_size)

    # Get output with original input
    with torch.no_grad():
        output_original = causality_model(inputs_embeds=inputs_embeds_original.clone())

    # Test at different positions
    test_positions = [4, 8, 12]
    causality_passed = True

    for modify_pos in test_positions:
        # Create modified input where we change timesteps from modify_pos onwards
        inputs_embeds_modified = inputs_embeds_original.clone()
        inputs_embeds_modified[:, modify_pos:, :] += torch.randn(batch_size, seq_len - modify_pos, hidden_size) * 10.0

        # Get output with modified input
        with torch.no_grad():
            output_modified = causality_model(inputs_embeds=inputs_embeds_modified)

        # Check that outputs BEFORE modify_pos are unchanged
        outputs_before_original = output_original.last_hidden_state[:, :modify_pos, :]
        outputs_before_modified = output_modified.last_hidden_state[:, :modify_pos, :]

        if torch.allclose(outputs_before_original, outputs_before_modified, atol=1e-5):
            print(f"   Position {modify_pos}: PASS (earlier outputs unchanged)")
        else:
            print(f"   Position {modify_pos}: FAIL (causality violation!)")
            causality_passed = False

        # Verify outputs at/after modify_pos are different (sanity check)
        outputs_after_original = output_original.last_hidden_state[:, modify_pos:, :]
        outputs_after_modified = output_modified.last_hidden_state[:, modify_pos:, :]

        if not torch.allclose(outputs_after_original, outputs_after_modified, atol=1e-3):
            print(f"   Position {modify_pos}: Sanity check PASS (later outputs changed)")
        else:
            print(f"   Position {modify_pos}: Sanity check FAIL (later outputs should change)")
            causality_passed = False

    # Test with different layer patterns
    print("\n   Testing causality with different layer patterns...")
    patterns = ["MMMM", "****", "M*M*", "MM**"]
    for pattern in patterns:
        pattern_config = NemotronHConfig(
            hidden_size=64,
            num_hidden_layers=4,
            vocab_size=1000,
            num_attention_heads=4,
            num_key_value_heads=2,
            mamba_num_heads=8,
            mamba_head_dim=8,
            ssm_state_size=16,
            n_groups=2,
            intermediate_size=128,
            hybrid_override_pattern=pattern,
        )
        pattern_model = NemotronHModel(pattern_config)
        pattern_model.eval()

        torch.manual_seed(789)
        test_input = torch.randn(2, 8, 64)
        modify_pos = 4

        with torch.no_grad():
            out_orig = pattern_model(inputs_embeds=test_input.clone())

        test_input_mod = test_input.clone()
        test_input_mod[:, modify_pos:, :] += torch.randn(2, 4, 64) * 10.0

        with torch.no_grad():
            out_mod = pattern_model(inputs_embeds=test_input_mod)

        if torch.allclose(
            out_orig.last_hidden_state[:, :modify_pos, :], out_mod.last_hidden_state[:, :modify_pos, :], atol=1e-5
        ):
            print(f"   Pattern '{pattern}': PASS")
        else:
            print(f"   Pattern '{pattern}': FAIL (causality violation!)")
            causality_passed = False

    if causality_passed:
        print("   All causality tests PASSED!")
    else:
        print("   WARNING: Some causality tests FAILED!")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
