# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import amp_C
import torch

from nemo.utils.model_utils import param_is_not_shared

try:
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.layers import param_is_not_tensor_parallel_duplicate

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

try:
    from apex.multi_tensor_apply import multi_tensor_applier
    from apex.optimizers import FusedAdam

    HAVE_APEX = True

except ModuleNotFoundError:
    HAVE_APEX = False


class MegatronFusedAdam(FusedAdam):
    """Wrapper class that supports NeMo-Megatron optimizations

    Performs gradient clipping, unscaling, and optimizer step.
    """

    def __init__(self, *args, max_norm=0, norm_type=2, **kwargs):
        super().__init__(*args, **kwargs)

        assert norm_type == 2, "Currently only norm_type=2 is supported for MegatronFusedAdam"

        # Gradient clipping parameters
        self.max_norm = float(max_norm)
        self.norm_type = float(norm_type)

    def step(self, closure=None, grad_scaler=None):
        # Code path below assumes capturable=True and master_weights=True
        if not (self.capturable and self.master_weights):
            return super().step(closure=closure, grad_scaler=grad_scaler)

        loss = None
        if closure is not None:
            loss = closure()

        for group, group_master in zip(self.param_groups, self.param_groups_master):
            if len(group['params']) == 0:
                continue
            device = group['params'][0].device
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']

            # Assume same step per parameter group for simplicity
            if 'step' in group:
                group['step'] += 1 if not self.capturable else (self._dummy_overflow_buf != 1).to(torch.int)
            else:
                group['step'] = 1 if not self.capturable else torch.tensor([1], dtype=torch.int, device=device)

            # Check for overflow in gradients
            found_inf = (
                grad_scaler._check_inf_per_device(self)[device]
                if grad_scaler is not None
                else torch.zeros((1,), device=device)
            )
            self._dummy_overflow_buf.copy_(found_inf)

            # Get gradient scaling/unscaling factors
            scale, inv_scale = None, None
            if grad_scaler:
                scale = grad_scaler._get_scale_async()
                inv_scale = scale.double().reciprocal().float()
            else:
                scale = torch.ones((1,), device=device)
                inv_scale = torch.ones((1,), device=device)
            combined_scale = inv_scale

            # Gradient clipping
            if self.max_norm > 0:
                # Unscale gradients and find L2 norm
                fp32_grads_for_norm = []
                fp16_grads_for_norm = []
                for p in group['params']:
                    if p.grad is None:
                        continue
                    assert p.dtype in [torch.float32, torch.float16], 'Only FP32/FP16 model parameters are supported'

                    is_not_shared = param_is_not_shared(p)
                    is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(p)
                    if is_not_shared and is_not_tp_duplicate:
                        if p.dtype == torch.float32:
                            fp32_grads_for_norm.append(p.grad.detach())
                        else:
                            fp16_grads_for_norm.append(p.grad.detach())

                if fp32_grads_for_norm:
                    fp32_grad_norm, _ = multi_tensor_applier(
                        amp_C.multi_tensor_unscale_l2norm,
                        self._dummy_overflow_buf,
                        [fp32_grads_for_norm],
                        inv_scale,
                        False,
                    )
                else:
                    fp32_grad_norm = torch.zeros(1, dtype=torch.float32, device=device)

                if fp16_grads_for_norm:
                    fp16_grad_norm, _ = multi_tensor_applier(
                        amp_C.multi_tensor_unscale_l2norm,
                        self._dummy_overflow_buf,
                        [fp16_grads_for_norm],
                        inv_scale,
                        False,
                    )
                else:
                    fp16_grad_norm = torch.zeros(1, dtype=torch.float32, device=device)

                # Prep L2 norm for allreduce
                total_norm = (fp32_grad_norm ** self.norm_type + fp16_grad_norm ** self.norm_type).squeeze()

                # Allreduce L2 norm across model-parallel GPUs
                torch.distributed.all_reduce(
                    total_norm, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_model_parallel_group()
                )
                total_norm = total_norm ** (1.0 / self.norm_type)

                # Combine unscaling factor with clip coefficient
                clip_coeff = self.max_norm / (total_norm + 1.0e-6)
                clip_coeff_clamped = torch.clamp(clip_coeff, max=1.0)
                combined_scale = clip_coeff_clamped * combined_scale  # Potential issue with associativity?

            # Create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []
            p_16_master = []
            p_32_master = []

            for p, p_master in zip(group['params'], group_master['params']):
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError(
                        'MegatronFusedAdam does not support sparse gradients, please consider SparseAdam instead'
                    )

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()

                if p.dtype == torch.float16:
                    p_16_master.append(p_master.data)
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(state['exp_avg'])
                    v_16.append(state['exp_avg_sq'])
                elif p.dtype == torch.float32:
                    p_32_master.append(p_master.data)
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state['exp_avg'])
                    v_32.append(state['exp_avg_sq'])
                else:
                    raise RuntimeError('MegatronFusedAdam only supports fp16 and fp32.')

            if len(g_16) > 0:
                multi_tensor_applier(
                    self.multi_tensor_adam_capturable_master,
                    self._dummy_overflow_buf,
                    [g_16, p_16, m_16, v_16, p_16_master],
                    group['lr'],
                    beta1,
                    beta2,
                    group['eps'],
                    group['step'],
                    self.adam_w_mode,
                    bias_correction,
                    group['weight_decay'],
                    combined_scale,
                )

            if len(g_32) > 0:
                multi_tensor_applier(
                    self.multi_tensor_adam_capturable_master,
                    self._dummy_overflow_buf,
                    [g_32, p_32, m_32, v_32, p_32_master],
                    group['lr'],
                    beta1,
                    beta2,
                    group['eps'],
                    group['step'],
                    self.adam_w_mode,
                    bias_correction,
                    group['weight_decay'],
                    combined_scale,
                )

        return loss
