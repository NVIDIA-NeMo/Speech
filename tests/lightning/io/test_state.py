import pytest
from torch import nn

from nemo.lightning.io.state import StateDictTransform, TransformCTX, state_transform


class TestStateDictTransform:
    """
    Tests for the StateDictTransform functionality.
    """

    @pytest.fixture
    def mock_ctx(self):
        """
        Provides a mock transformation context with predefined source and target states.

        Returns
        -------
            TransformCTX: A context object with source and target states.
        """
        source_state = {
            'model.layers.0.self_attn.q_proj.weight': 1,
            'model.layers.0.self_attn.k_proj.weight': 2,
            'model.layers.0.self_attn.v_proj.weight': 3,
            'model.layers.1.self_attn.q_proj.weight': 1,
            'model.layers.1.self_attn.k_proj.weight': 2,
            'model.layers.1.self_attn.v_proj.weight': 3,
        }
        target_state = {
            "decoder.layers.0.self_attention.linear_qkv.weight": 10,
            "decoder.layers.1.self_attention.linear_qkv.weight": 10,
        }
        ctx = TransformCTX(
            source=nn.Module(), source_state=source_state, target=nn.Module(), target_state=target_state
        )
        return ctx

    @pytest.fixture
    def mock_multi_target_ctx(self):
        """
        Provides a mock transformation context with a source state that matches the expected source_key
        and a target state prepared with initial values for the expected target_keys.
        """
        source_state = {'model.layers.1.self_attn.q_proj.weight': 1}
        # Populate target_state with initial placeholder values for keys expected to be matched and updated
        target_state = {
            'decoder.layers.1.self_attention.linear_q.weight': 0,
            'decoder.layers.1.self_attention.linear_k.weight': 0,
        }
        ctx = TransformCTX(
            source=nn.Module(), source_state=source_state, target=nn.Module(), target_state=target_state
        )
        return ctx

    def test_transform_with_multiple_source_keys(self, mock_ctx):
        """
        Test transformation when multiple source keys are specified.
        """
        transform = StateDictTransform(
            source_key=(
                "model.layers.*.self_attn.q_proj.weight",
                "model.layers.*.self_attn.k_proj.weight",
                "model.layers.*.self_attn.v_proj.weight",
            ),
            target_key="decoder.layers.*.self_attention.linear_qkv.weight",
            transform=lambda ctx, k, q, v: q + k + v,
        )
        transform(mock_ctx)
        assert mock_ctx.target_state["decoder.layers.0.self_attention.linear_qkv.weight"] == 6
        assert mock_ctx.target_state["decoder.layers.1.self_attention.linear_qkv.weight"] == 6

    def test_transform_with_wildcard_in_source_keys(self, mock_ctx):
        """
        Test transformation using a wildcard pattern in source keys.
        """
        transform = StateDictTransform(
            source_key="model.layers.*.self_attn.*_proj.weight",
            target_key="decoder.layers.*.self_attention.linear_qkv.weight",
            transform=lambda ctx, k, q, v: q + k + v,
        )
        transform(mock_ctx)
        assert mock_ctx.target_state["decoder.layers.0.self_attention.linear_qkv.weight"] == 6
        assert mock_ctx.target_state["decoder.layers.1.self_attention.linear_qkv.weight"] == 6

    def test_transform_with_mapped_source_keys(self, mock_ctx):
        """
        Test transformation with a dictionary mapping for source keys.
        """
        transform = StateDictTransform(
            source_key={
                "k": "model.layers.*.self_attn.k_proj.weight",
                "q": "model.layers.*.self_attn.q_proj.weight",
                "v": "model.layers.*.self_attn.v_proj.weight",
            },
            target_key="decoder.layers.*.self_attention.linear_qkv.weight",
            transform=lambda ctx, k, q, v: q + k + v,
        )
        transform(mock_ctx)
        assert mock_ctx.target_state["decoder.layers.0.self_attention.linear_qkv.weight"] == 6
        assert mock_ctx.target_state["decoder.layers.1.self_attention.linear_qkv.weight"] == 6

    def test_transform_with_variable_arguments(self, mock_ctx):
        """
        Test transformation with a wildcard pattern and variable arguments.
        """
        transform = StateDictTransform(
            source_key="model.layers.*.self_attn.*_proj.weight",
            target_key="decoder.layers.*.self_attention.linear_qkv.weight",
            transform=lambda ctx, *args: sum(args),
        )
        transform(mock_ctx)
        assert mock_ctx.target_state["decoder.layers.0.self_attention.linear_qkv.weight"] == 6
        assert mock_ctx.target_state["decoder.layers.1.self_attention.linear_qkv.weight"] == 6

    def test_transform_with_no_matching_source_keys(self, mock_ctx):
        """
        Test transformation when no source keys match the pattern.
        """
        transform = StateDictTransform(
            source_key="non.existent.pattern",
            target_key="decoder.layers.*.self_attention.linear_qkv.weight",
            transform=lambda ctx, *args: sum(args),
        )
        with pytest.raises(ValueError):
            transform(mock_ctx)

    def test_transform_with_invalid_transform_function(self, mock_ctx):
        """
        Test transformation with a transform function that does not match expected signature.
        """
        transform = StateDictTransform(
            source_key="model.layers.*.self_attn.q_proj.weight",
            target_key="decoder.layers.*.self_attention.linear_qkv.weight",
            transform=lambda ctx: 0,  # Invalid signature
        )
        with pytest.raises(ValueError):
            transform(mock_ctx)

    def test_transform_with_tuple_target_key_and_multiple_outputs(self, mock_multi_target_ctx):
        """
        Test transformation where the target_key is a tuple and the transform function
        returns multiple values that are then unrolled to these target keys.
        """

        # Define a transformation that splits the input into two parts
        def split_transform(ctx, x):
            return x - 1, x + 1

        # Apply the transformation
        transform = StateDictTransform(
            source_key="model.layers.1.self_attn.q_proj.weight",
            target_key=(
                "decoder.layers.1.self_attention.linear_q.weight",
                "decoder.layers.1.self_attention.linear_k.weight",
            ),
            transform=split_transform,
        )
        transform(mock_multi_target_ctx)

        # Check that the target state has been updated correctly
        assert mock_multi_target_ctx.target_state["decoder.layers.1.self_attention.linear_q.weight"] == 0
        assert mock_multi_target_ctx.target_state["decoder.layers.1.self_attention.linear_k.weight"] == 2


class TestStateTransformDecorator:
    """
    Tests for the @state_transform decorator functionality.
    """

    @pytest.fixture
    def mock_ctx(self):
        """
        Provides a mock transformation context with predefined source and target states.
        """
        source_state = {
            'model.layers.1.self_attn.q_proj.weight': 1,
            'model.layers.1.self_attn.k_proj.weight': 2,
            'model.layers.1.self_attn.v_proj.weight': 3,
        }
        # Pre-populate target_state with initial values or placeholders
        target_state = {
            "decoder.layers.1.self_attention.linear_q.weight": 0,
            "decoder.layers.1.self_attention.linear_k.weight": 0,
            "decoder.layers.1.self_attention.linear_v.weight": 0,
        }
        ctx = TransformCTX(
            source=nn.Module(), source_state=source_state, target=nn.Module(), target_state=target_state
        )
        return ctx

    def test_single_transform(self, mock_ctx):
        """
        Test the @state_transform decorator with a single source and target key.
        """
        # Apply the transformation
        single_transform(mock_ctx)
        # Verify the target state is updated correctly
        assert mock_ctx.target_state["decoder.layers.1.self_attention.linear_q.weight"] == 11

    def test_multiple_outputs_transform(self, mock_ctx):
        """
        Test the @state_transform decorator with a single source key and multiple target keys.
        """
        # Apply the transformation
        multiple_outputs_transform(mock_ctx)
        # Verify the target state is updated correctly for each key
        assert mock_ctx.target_state["decoder.layers.1.self_attention.linear_q.weight"] == 2
        assert mock_ctx.target_state["decoder.layers.1.self_attention.linear_k.weight"] == 1
        assert mock_ctx.target_state["decoder.layers.1.self_attention.linear_v.weight"] == 3


@state_transform(
    source_key="model.layers.*.self_attn.q_proj.weight", target_key="decoder.layers.1.self_attention.linear_q.weight"
)
def single_transform(ctx, x):
    """
    A single transformation function that adds 10 to the input value.
    """
    return x + 10


@state_transform(
    source_key="model.layers.1.self_attn.*_proj.weight",
    target_key=(
        "decoder.layers.1.self_attention.linear_q.weight",
        "decoder.layers.1.self_attention.linear_k.weight",
        "decoder.layers.1.self_attention.linear_v.weight",
    ),
)
def multiple_outputs_transform(ctx, *args):
    """
    A transformation function that returns multiple values for multiple target keys.
    """
    return args
