from dataclasses import dataclass, field
from typing import List, Literal

from megatron.core import parallel_state
from torch import nn

from nemo.lightning.pytorch.callbacks.peft import PEFT, AdapterWrapper
from nemo.utils import logging


class AdapterParallelAdd(AdapterWrapper):
    """An adapter wrapper that adds the output of the adapter to the output of the wrapped module.

    This class is designed to be used with LoRA (Low-Rank Adaptation) and similar techniques
    where the adapter's output is added to the main module's output. It extends the AdapterWrapper
    class to provide a specific implementation of the forward method.
    """

    def forward(self, x):
        linear_output = self.to_wrap(x)
        assert isinstance(
            linear_output, tuple
        ), f"{self.to_wrap} should return a tuple but instead returns {linear_output}"
        """ Four cases for the wrapped module's return values
        1. nothing: (out, None)
        2. return_bias: (out, bias)
        2. return_layernorm_output: ((out, ln_out), None)
        3. both: (out, bias, ln_out)
        """
        if len(linear_output) == 2:
            linear_output, bias = linear_output
            if isinstance(linear_output, tuple) and len(linear_output) == 2:
                linear_output, layernorm_output = linear_output
                x = layernorm_output
        elif len(linear_output) == 3:
            linear_output, bias, layernorm_output = linear_output
            x = layernorm_output

        adapter_output = self.adapter(x)
        return linear_output + adapter_output, bias


@dataclass
class LoRA(PEFT):
    """
    Implements the LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.

    LoRA uses a low-rank projection to adapt the weights of a pre-trained model to a new downstream task.
    This class facilitates the application of LoRA to specific modules within the model architecture.

    Args:
        target_modules (List[str], optional): A list of module names to apply LoRA to.
            Defaults to all linear layers ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
                - 'linear_qkv': Apply LoRA to the fused linear layer used for query, key, and value projections
                                in self-attention modules.
                - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention modules.
                - 'linear_fc1': Apply LoRA to the first fully-connected layer in MLP.
                - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
        dim (int): Dimension of the low-rank projection space. Defaults to 32.
        alpha (int): Weighting factor for the low-rank projection. Defaults to 32.
        dropout (float): Dropout rate for the low-rank projection. Defaults to 0.0.
        dropout_position (Literal['pre', 'post'], optional): Position for applying dropout.
            Can be 'pre' (before the low-rank projection) or 'post' (after). Defaults to 'post'.

    Example:
    --------
        >>> from nemo.collections import llm
        >>> lora = llm.peft.LoRA(target_modules=['linear_qkv', 'linear_proj'], dim=32)
        >>> model = llm.Mistral7BModel(model_transform=lora)
        >>> # (set up trainer and data)
        >>> trainer.fit(model, data)

    References:
    -----------
        Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021).
        LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685.
        https://arxiv.org/abs/2106.09685

    )
    """

    target_modules: List[str] = field(
        default_factory=lambda: ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']
    )
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal['pre', 'post'] = 'post'
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"

    def transform(self, m: nn.Module, name=None, prefix=None):
        """
        Applies LoRA to a specific module within the model architecture.

        Args:
            m (nn.Module): The module to apply LoRA to.
            name (str, optional): Name of the module (if applicable). Defaults to None.
            prefix (str, optional): Prefix for the module name (if applicable). Defaults to None.

        Returns:
            nn.Module: The modified module with LoRA applied, or the original module if not a target.
        """
        from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import ParallelLinearAdapter

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if name in self.target_modules:
            # m.in_features and m.out_features are divided by tp_size already,
            # but in_features and out_features passed to ParallelLinearAdapter are not.
            if name in ['linear_qkv', 'linear_fc1']:
                # Column Parallel Linear
                input_is_parallel = False
                in_features = m.in_features
                out_features = m.out_features * tp_size
                # LoRA is applied after layernorm, so layernorm output must be returned
                m.return_layernorm_output = True
                # perf optimization for LoRA + SP
                if m.config.sequence_parallel and not m.ub_overlap_ag:
                    m.return_layernorm_output_gathered = True
            else:  # name in ['linear_proj', 'linear_fc2']
                # Row Parallel Linear
                input_is_parallel = True
                in_features = m.in_features * tp_size
                out_features = m.out_features

            logging.info(f"Adding lora to: {prefix}.{name}")
            adapter = ParallelLinearAdapter(
                in_features,
                out_features,
                self.dim,
                activation='identity',
                norm_position=None,
                norm_type=None,
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                gather_output=False,
                input_is_parallel=input_is_parallel,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                model_parallel_config=getattr(m, "config", None),
                alpha=self.alpha,
            )
            return AdapterParallelAdd(m, adapter)
        return m
