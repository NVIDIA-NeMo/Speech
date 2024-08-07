Mixture of Experts
==================

Overview
--------

NeMo supports Mixture of Experts (MoE) in the transformer layer for NLP models.

MoE is a machine learning technique where multiple specialized models (experts,
usually multi-layer perceptrons) are combined to solve a complex task. Each expert
focuses on a specific subtask or domain, while a gating network dynamically activates
the most appropriate expert based on the current input.


To use MoE  in the NeMo Framework, adjust the ``num_moe_experts`` parameter in the model configuration:

1. Set ``num_moe_experts`` to `8` to leverage 8 experts in the MoE module.

   .. code-block:: yaml

       num_moe_experts: 8  # Set MoE to use 8 experts

2. Set ``moe_router_topk`` to the number of experts you want activated. For example, if you want to process each input with two experts:

   .. code-block:: yaml

       moe_router_topk: 2  # Processes each token using 2 experts.

In addition, NeMo provides options to configure MoE-specific loss function.
To balance token distribution across experts:

1. Set ``moe_router_load_balancing_type`` to specify the load balancing method:

   .. code-block:: yaml

      moe_router_load_balancing_type: aux_loss  # to use the auxilary loss, other options include "sinkhorn".

2. Set ``moe_aux_loss_coeff`` to specify the weight of the auxilary loss. Values in the 1e-2 range are a good start, as follows:

   .. code-block:: yaml

      moe_aux_loss_coeff: 1e-2  # set the aux-loss weight to 1e-2

3. Set ``moe_z_loss_coeff`` to specify the weight of the z-loss. A starting value of 1e-3 is recommended, as follows:

   .. code-block:: yaml

      moe_z_loss_coeff: 1e-3

Other options include:

1. ``moe_input_jitter_eps`` adds noise to the input tensor by applying jitter with a specified epsilon value.

2. ``moe_token_dropping`` enables selectively dropping and padding tokens for each expert to achieve
   a specified capacity.

3. ``moe_token_dropping`` specifies the token dispatcher type, options include 'allgather' and 'alltoall'.

4. ``moe_per_layer_logging`` enables per-layer logging for MoE, currently support aux-loss and z-loss.

5. ``moe_expert_capacity_factor`` the capacity factor for each expert, None means no token will be dropped. The default is None.

6. ``moe_pad_expert_input_to_capacity`` if True, pads the input for each expert to match the expert capacity length, effective only after the moe_expert_capacity_factor is set. The default setting is False.

7. ``moe_token_drop_policy`` the policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped. Default value is "probs".

8. ``moe_layer_recompute`` if True, checkpointing moe_layer to save activation memory, default is False.