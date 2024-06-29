import torch

from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import logging, logging_mode


class AttentionAdapterModuleMixin(adapter_mixins.AdapterModuleMixin):
    """
    Utility class that implements a custom forward method for Modules that are attention based.
    Attention based adapters can support either linear adapters, and Multi-Head Attention adapters.

    However, Multi Head Attention adapters require additional arguments, such as `att_mask` and `pos_emb`.
    This utility class unifies the adapter forward pass for both types of adapters.

    .. Usage:

        To use this class, inherit from this class, and when calling self.foward_enabled_adapters() pass the following:

    .. code-block:: python

            if self.is_adapter_available():
                # Call the MHA adapters
                pack_ip = {
                    'x': residual,
                    'loc': 'mha',
                    'att_mask': att_mask,
                    'pos_emb': pos_emb,
                }
                pack_ip = self.forward_enabled_adapters(pack_ip)
                residual = pack_ip['x']

            if self.is_adapter_available():
                # Call the Linear adapters
                pack_ip = {
                    'x': x,
                    'loc': 'post',
                }
                pack_ip = self.forward_enabled_adapters(pack_ip)
                x = pack_ip['x']
    """

    def forward_single_enabled_adapter_(
        self,
        input: dict,
        adapter_module: torch.nn.Module,
        *,
        adapter_name: str,
        adapter_strategy: 'nemo.core.classes.mixins.adapter_mixin_strategies.AbstractAdapterStrategy',
    ):
        """
        Perform the forward step of a single adapter module on some input data.

        **Note**: Subclasses can override this method to accommodate more complicate adapter forward steps.

        Args:
            input: Dictionary of packed tensors. The dict should contain at least
                `x`: output tensor
                `loc`: Semantic location in module where this adapter was called. Can be 'mha' or 'post'.
                `att_mask`: Optional, Attention mask
                `pos_emb`: Optional, Positional Embedding for Relative Positional Encoding.
                The output tensor of the calling module is the input to the first adapter, whose output
                is then chained to the next adapter until all adapters are consumed.
            adapter_module: The adapter module that is currently required to perform the forward pass.
            adapter_name: The resolved name of the adapter that is undergoing the current forward pass.
            adapter_strategy: A subclass of `AbstractAdapterStrategy`, that determines how the
                output of the adapter should be merged with the input, or if it should be merged at all.

        Returns:
            The result tensor, after the current active adapter has finished its forward pass.
        """
        if not hasattr(self, 'self_attention_model'):
            raise RuntimeError(
                "self_attention_model attribute not found in the module! Please set in the module "
                "a string attribute 'self_attention_model' with value 'abs_pos', 'rel_pos' or "
                "other supported self-attention model types."
            )

        # Collect imports to prevent circular imports
        from nemo.collections.asr.modules.transformer import transformer_modules as transformer_mha
        from nemo.collections.asr.parts.submodules import multi_head_attention as conformer_mha

        # (input: torch.Tensor, adapter: torch.nn.Module, *, module: 'AdapterModuleMixin')
        x = input['x']
        loc = input['loc']
        att_mask = input.get('att_mask', None)
        pos_emb = input.get('pos_emb', None)

        from nemo.collections.common.parts import adapter_modules

        if isinstance(adapter_module, adapter_modules.LinearAdapter) and loc == 'post':
            output = adapter_strategy(x, adapter_module, module=self)

        elif isinstance(adapter_module, conformer_mha.MultiHeadAttention) and loc == 'mha':
            if self.self_attention_model == 'rel_pos':
                x = dict(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb)
                output = adapter_strategy(x, adapter_module, module=self)

            elif self.self_attention_model == 'abs_pos':
                x = dict(query=x, key=x, value=x, mask=att_mask)
                output = adapter_strategy(x, adapter_module, module=self)

            else:
                raise ValueError(f"Unsupported value of self_attention_model , provided {self.self_attention_model}!")

        elif isinstance(adapter_module, transformer_mha.MultiHeadAttention) and loc == 'mha':
            x = dict(queries=x, keys=x, values=x, attention_mask=att_mask)
            output = adapter_strategy(x, adapter_module, module=self)

        else:
            # No adapter compatible, skip
            logging.warning(
                "No adapter compatible with the current module. Skipping adapter forward pass.", mode=logging_mode.ONCE
            )

            output = x

        input['x'] = output

        return input
