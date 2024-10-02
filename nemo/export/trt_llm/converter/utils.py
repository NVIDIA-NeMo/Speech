# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


from typing import List, Optional, Tuple, Union
import numpy as np
import tensorrt_llm
import torch
from tensorrt_llm._utils import mpi_comm, torch_to_numpy

# A global dicts to store exported weights.
# This is set to be a global variable to avoid extra code modification from tensorrt_llm.
weights_dict = {}


DECODER_MODEL_TYPE = {
    "gptj": 'GPTForCausalLM',
    "gptnext": 'GPTForCausalLM',
    "llama": 'LlamaForCausalLM',
    "gemma": 'GemmaForCausalLM',
    "falcon": 'FalconForCausalLM',
}

post_layernorm_keys = [
    "post_attention_layernorm.weight",
    "post_attention_layernorm.bias",
    "post_self_attn_layernorm.weight",
]
mlp_proj_bias_keys = ["mlp.linear_fc2.bias", "mlp.dense_4h_to_h.bias"]
attention_dense_bias_keys = ["attention.linear_proj.bias", "attention.dense.bias"]
input_layernorm_keys = ["input_layernorm.weight", "input_layernorm.bias"]
pre_layernorm_keys = ["pre_mlp_layernorm.weight", "pre_mlp_layernorm.bias"]
attention_dense_weight_keys = ["attention.linear_proj.weight", "attention.dense.weight"]
mlp_proj_weight_keys = ["mlp.linear_fc2.weight", "mlp.dense_4h_to_h.weight"]
mlp_fc_keys = ["mlp.dense_h_to_4h.weight", "mlp.dense_h_to_4h.bias", "mlp.linear_fc1.weight", "mlp.linear_fc1.bias"]
attention_qkv_bias_keys = ["attention.query_key_value.bias", "attention.linear_qkv.bias"]
attention_qkv_weight_keys = ["attention.query_key_value.weight", "attention.linear_qkv.weight"]
mlp_router_keys = ["mlp.router.weight"]
mlp_fc_expert_keys = ["experts.linear_fc1.weight"]
mlp_proj_experts_keys = ["experts.linear_fc2.weight"]
final_layernorm_keys = ["final_layernorm.weight", "final_layernorm.bias"]
mlp_dense_2_keys = ["mlp.dense_h_to_4h_2.weight", "mlp.dense_h_to_4h_2.bias"]
attention_not_mapped_keys = [
    "attention.query.weight",
    "attention.query.bias",
    "attention.key_value.weight",
    "attention.key_value.bias",
]

weight_scaling_suffix = '.weights_scaling_factor'
activation_scaling_suffix = '.activation_scaling_factor'


def save_val(val, dir, key, tp_num=None):
    suffix = "" if tp_num is None else f".{tp_num}.bin"
    global weights_dict

    # Transpose linear layer weights to the correct shape.
    if torch.is_tensor(val):
        val = val.detach().contiguous()
        if len(val.shape) >= 2:
            val = val.reshape(val.shape[0], -1)
            val = torch.transpose(val, 0, 1)
        if key not in weights_dict:
            weights_dict[f"{key}{suffix}"] = torch.empty(
                val.size(), dtype=val.dtype, layout=val.layout, device="cpu", pin_memory=True
            )
        weights_dict[f"{key}{suffix}"].copy_(val, non_blocking=True)
    else:
        if len(val.shape) >= 2:
            val = np.ascontiguousarray(np.transpose(val.reshape(val.shape[0], -1), [1, 0]))
        weights_dict[f"{key}{suffix}"] = val


def save_split(split_vals, dir, key, i, split_factor):
    for j, val in enumerate(split_vals):
        save_val(val, dir, key, i * split_factor + j)


def save_expert_split(split_vals, dir, key, i, split_factor):
    for j, val in enumerate(split_vals):
        tp_num = i * split_factor + j
        suffix = "" if tp_num is None else f".{tp_num}.bin"

        global weights_dict
        weights_dict[f"{key}{suffix}"] = val


def generate_int8(weights, act_range, is_qkv=False, multi_query_mode=False):
    """This function has two purposes:
    - compute quantized weights, scaled either per-tensor or per-column
    - compute scaling factors.

    Depending on the GEMM API (CUTLASS/CUBLAS) the required scaling factors differ.
    CUTLASS uses two sets of scaling factors. One for the activation X, one for the weight W.
    CUBLAS only has one (we can't do per-row scaling). So we must provide pre-multiplied scaling factor.

    Here is the list of what we need (T means per-tensor, C per-column):
    - scale_x_orig_quant puts fp activation into the quantized range (i.e. [-128, 127], for int8).
    Used before the GEMM. (T)
    - scale_y_quant_orig puts quantized activation into the fp range. Used if the GEMM outputs int8. (T)
    - scale_w_quant_orig puts weights from quant range to fp range (used with CUTLASS) (T, C)
    - scale_y_accum_quant puts the GEMM result (XW) from accumulation range (int32)
    to quant range (int8) (used for CUBLAS) (T, C)

    Note that we don't do anything special about row-parallel GEMM.
    Theoretically, we could have per-GPU scaling factors too,
    but then the model would change depending on the number of GPUs used.

    For QKV projection, the behavior is special. Even if we have a single matrix to perform QKV projection,
    we consider it
    as three different matrices: Q, K, and V. So per-tensor actually means one scaling factor for each Q, K and V.
    """
    # compute weight scaling factors for fp->int8 and int8->fp
    if is_qkv and not multi_query_mode:
        scale_w_orig_quant_t = 127.0 / act_range["w"].reshape(3, -1).max(dim=-1, keepdims=True)[0].cpu().numpy()
        scale_w_orig_quant_c = 127.0 / act_range["w"].reshape(3, -1).cpu().numpy()
    elif is_qkv and multi_query_mode:
        raise ValueError("Multi-query w/ int8 quant has not been supported yet")
    else:
        scale_w_orig_quant_t = 127.0 / act_range["w"].max().cpu().numpy()
        scale_w_orig_quant_c = 127.0 / act_range["w"].cpu().numpy()
    scale_w_quant_orig_t = 1.0 / scale_w_orig_quant_t
    scale_w_quant_orig_c = 1.0 / scale_w_orig_quant_c

    # compute the rest of needed scaling factors
    scale_x_orig_quant_t = np.array(127.0 / act_range["x"].max().item())
    scale_y_orig_quant_t = np.array(127.0 / act_range["y"].max().item())
    scale_y_quant_orig_t = np.array(act_range["y"].max().item() / 127.0)
    scale_y_accum_quant_t = scale_y_orig_quant_t / (scale_x_orig_quant_t * scale_w_orig_quant_t)
    scale_y_accum_quant_c = scale_y_orig_quant_t / (scale_x_orig_quant_t * scale_w_orig_quant_c)
    if is_qkv:
        scale_y_accum_quant_t = np.broadcast_to(scale_y_accum_quant_t, scale_w_orig_quant_c.shape)
        scale_w_quant_orig_t = np.broadcast_to(scale_w_quant_orig_t, scale_w_orig_quant_c.shape)

    def to_i8(x):
        return x.round().clip(-127, 127).astype(np.int8)

    return {
        "weight.int8": to_i8(weights * scale_w_orig_quant_t),
        "weight.int8.col": to_i8(weights * scale_w_orig_quant_c),
        "scale_x_orig_quant": scale_x_orig_quant_t.astype(np.float32),
        "scale_w_quant_orig": scale_w_quant_orig_t.astype(np.float32),
        "scale_w_quant_orig.col": scale_w_quant_orig_c.astype(np.float32),
        "scale_y_accum_quant": scale_y_accum_quant_t.astype(np.float32),
        "scale_y_accum_quant.col": scale_y_accum_quant_c.astype(np.float32),
        "scale_y_quant_orig": scale_y_quant_orig_t.astype(np.float32),
    }


def write_int8(vals, dir, base_key, split_dim, tp_rank, split_factor, kv_cache_only=False):
    if not kv_cache_only:
        save_split(
            np.split(vals["weight.int8"], split_factor, axis=split_dim),
            dir,
            f"{base_key}.weight.int8",
            tp_rank,
            split_factor,
        )
        save_split(
            np.split(vals["weight.int8.col"], split_factor, axis=split_dim),
            dir,
            f"{base_key}.weight.int8.col",
            tp_rank,
            split_factor,
        )

    saved_keys_once = ["scale_y_quant_orig"]
    if not kv_cache_only:
        saved_keys_once += ["scale_x_orig_quant", "scale_w_quant_orig", "scale_y_accum_quant"]
    # per-column scaling factors are loaded per-gpu for ColumnParallel GEMMs (QKV, FC1)
    if not kv_cache_only:
        if split_dim == -1:
            save_split(
                np.split(vals["scale_w_quant_orig.col"], split_factor, axis=split_dim),
                dir,
                f"{base_key}.scale_w_quant_orig.col",
                tp_rank,
                split_factor,
            )
            save_split(
                np.split(vals["scale_y_accum_quant.col"], split_factor, axis=split_dim),
                dir,
                f"{base_key}.scale_y_accum_quant.col",
                tp_rank,
                split_factor,
            )
        else:
            saved_keys_once += ["scale_w_quant_orig.col", "scale_y_accum_quant.col"]

    if tp_rank == 0:
        for save_key in saved_keys_once:
            save_val(vals[save_key], dir, f"{base_key}.{save_key}")


def get_suffix(key: str) -> str:
    return '.' + key.split('.')[-1]


def get_trt_llm_prefix(key: str) -> str:
    layer_num = key.split(".")[1]
    return f'transformer.layers.{layer_num}'


def any_word_in_key(key: str, words: List[str]) -> bool:
    return any([word in key for word in words])


def sequential_key_map(key: str, mapping: List[Tuple[List[str], str]]) -> Optional[str]:
    for keywords, mapped in mapping:
        if any_word_in_key(key, keywords):
            return mapped

    return None


def get_trt_llm_infix(key: str) -> Optional[str]:
    mapping = [
        (post_layernorm_keys, '.post_layernorm'),
        (mlp_proj_bias_keys, '.mlp.proj'),
        (attention_dense_bias_keys, '.attention.dense'),
        (input_layernorm_keys, '.input_layernorm'),
        (pre_layernorm_keys, '.post_layernorm'),
        (attention_dense_weight_keys, '.attention.dense'),
        (mlp_proj_weight_keys, '.mlp.proj'),
        (mlp_fc_keys, '.mlp.fc'),
        (attention_qkv_bias_keys + attention_qkv_weight_keys, '.attention.qkv'),
        (mlp_router_keys, '.mlp.router'),
        (mlp_fc_expert_keys, '.mlp.fc'),
        (mlp_proj_experts_keys, '.mlp.proj'),
    ]
    return sequential_key_map(key, mapping)


def get_trt_llm_keyname(key: str) -> str:
    if any_word_in_key(key, final_layernorm_keys):
        return key.replace("final_layernorm", "transformer.ln_f")

    if infix := get_trt_llm_infix(key):
        return get_trt_llm_prefix(key) + infix + get_suffix(key)

    return key


def is_scaling_factor(key: str) -> bool:
    return "scale_fwd" in key


def get_scaling_factor_keys(key: str) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    # Reuses existing mapping of NeMo -> TRT LLM weights key via swapping suffixes
    corresponding_weight_key = '.'.join(key.split('.')[:-2]) + '.weight'
    corresponding_trt_llm_weight_key = get_trt_llm_keyname(corresponding_weight_key)
    base_key = '.'.join(corresponding_trt_llm_weight_key.split('.')[:-1])

    weight_scale = base_key + weight_scaling_suffix
    activation_scale = base_key + activation_scaling_suffix
    keys = (weight_scale, activation_scale)

    layer_prefix = get_trt_llm_prefix(key)
    mapped_key = layer_prefix + '.mlp.gate'
    gate_activation = mapped_key + activation_scaling_suffix
    gate_weight = mapped_key + weight_scaling_suffix
    gate_keys = (gate_activation, gate_weight)

    return keys, gate_keys


def save_scaling_factor(scaling_factors: dict, key: str, val: torch.Tensor, config: dict):
    if not is_scaling_factor(key):
        return scaling_factors

    activation_factor = torch_to_numpy(1 / val[0].view(1))
    weights_factor = torch_to_numpy(1 / val[1].view(1))

    (weights_key, activation_key), gate_keys = get_scaling_factor_keys(key)
    scaling_factors[activation_key] = activation_factor
    scaling_factors[weights_key] = weights_factor

    split_gated_activation = config.get("split_gated_activation", False)
    if split_gated_activation and any_word_in_key(key, ["mlp.dense_h_to_4h", "mlp.linear_fc1"]):
        (gate_activation_key, gate_weight_key) = gate_keys
        scaling_factors[gate_activation_key] = activation_factor
        scaling_factors[gate_weight_key] = weights_factor

    return scaling_factors


def cast_val_datatype(vals, trt_llm_key, storage_type, is_fp8_model, scaling_factors):
    if not is_fp8_model:
        return [val.to(storage_type) for val in vals]

    fp8_storage_type = torch.float8_e4m3fn
    quantized_keys = [
        k.split(weight_scaling_suffix)[0] for k in scaling_factors.keys() if k.endswith(weight_scaling_suffix)
    ]
    for k in quantized_keys:
        if k in trt_llm_key:
            storage_type = fp8_storage_type
            scale = scaling_factors[k + weight_scaling_suffix]
            vals = [val.to(torch.float32) / scale for val in vals]
            break

    return [val.to(storage_type) for val in vals]


def split_val_gate(vals: List[np.ndarray], convert_on_device: bool):
    if convert_on_device:
        return [[n] for n in torch.chunk(vals[0], 2, axis=-1)]

    splits = [np.split(val, 2, axis=-1) for val in vals]
    return list(zip(*splits))


# Note: in multi_query_mode, only query heads are split between multiple GPUs, while key/value head
# are not split as there is only one head per key/value.
@torch.no_grad()
def split_and_save_weight(
    tp_rank, saved_dir, split_factor, key, vals, storage_type, act_range, config, scaling_factors={}
):
    use_attention_nemo_shape = config.get("use_attention_nemo_shape", False)
    split_gated_activation = config.get("split_gated_activation", False)
    num_attention_heads = config.get("num_attention_heads", 0)
    tp_size = config.get("tp_size", 1)
    int8_outputs = config.get("int8_outputs", None)
    multi_query_mode = config.get("multi_query_mode", False)
    num_kv_heads = config.get("num_kv_heads", num_attention_heads)
    size_per_head = config.get("kv_channels", None)
    convert_on_device = config.get("convert_on_device", False)
    is_fp8_model = config.get("fp8_quantized", False)
    use_fp8_kv_cache = config.get("fp8_kvcache", False)
    save_int8 = int8_outputs == "all" or int8_outputs == "kv_cache_only"

    trt_llm_key = get_trt_llm_keyname(key)
    if not isinstance(vals, list):
        vals = [vals]

    if config.get("transpose_weights", False) and vals[0].ndim == 2:
        vals = [val.T for val in vals]
    if "layernorm.weight" in key and config.get("apply_layernorm_1p", False):
        vals = [val.float() + 1.0 for val in vals]

    vals = cast_val_datatype(vals, trt_llm_key, storage_type, is_fp8_model, scaling_factors)
    if convert_on_device:
        assert len(vals) == 1  # Should only convert a single device param per call
        assert torch.is_tensor(vals[0])
    elif torch.is_tensor(vals[0]):
        vals = [torch_to_numpy(val.cpu()) for val in vals]

    if any_word_in_key(
        key,
        input_layernorm_keys
        + pre_layernorm_keys
        + attention_dense_bias_keys
        + post_layernorm_keys
        + mlp_proj_bias_keys
        + final_layernorm_keys,
    ) and (tp_rank == 0 or convert_on_device):
        # shared weights, only need to convert the weights of rank 0
        save_val(vals[0], saved_dir, trt_llm_key)

    elif any_word_in_key(key, attention_dense_weight_keys + mlp_proj_weight_keys):
        if convert_on_device:
            save_val(vals[0], saved_dir, trt_llm_key)
        else:
            cat_dim = 0
            val = np.concatenate(vals, axis=cat_dim)
            split_vals = np.split(val, split_factor, axis=cat_dim)
            save_split(split_vals, saved_dir, trt_llm_key, tp_rank, split_factor)

        if act_range is not None and int8_outputs == "all":
            base_key = trt_llm_key.replace(".weight", "")
            vals_i8 = generate_int8(val, act_range, multi_query_mode=multi_query_mode)
            write_int8(vals_i8, saved_dir, base_key, cat_dim, tp_rank, split_factor)

    elif any_word_in_key(key, mlp_fc_keys):
        if split_gated_activation:
            vals, gates = split_val_gate(vals, convert_on_device)

        if convert_on_device:
            save_val(vals[0], saved_dir, trt_llm_key)
        else:
            cat_dim = -1
            val = np.concatenate(vals, axis=cat_dim)
            split_vals = np.split(val, split_factor, axis=cat_dim)
            save_split(split_vals, saved_dir, trt_llm_key, tp_rank, split_factor)

        if act_range is not None and int8_outputs == "all":
            base_key = trt_llm_key.replace(".weight", "")
            vals_i8 = generate_int8(val, act_range, multi_query_mode=multi_query_mode)
            write_int8(vals_i8, saved_dir, base_key, cat_dim, tp_rank, split_factor)

        if split_gated_activation:
            assert not save_int8
            layer_prefix = get_trt_llm_prefix(key)
            gate_key = layer_prefix + '.mlp.gate' + get_suffix(trt_llm_key)
            if convert_on_device:
                save_val(gates[0], saved_dir, gate_key)
            else:
                gate = np.concatenate(gates, axis=cat_dim)
                split_vals = np.split(gate, split_factor, axis=cat_dim)
                save_split(split_vals, saved_dir, gate_key, tp_rank, split_factor)

    elif any_word_in_key(key, mlp_dense_2_keys):
        if convert_on_device:
            save_val(vals[0], saved_dir, trt_llm_key)
        else:
            cat_dim = -1
            val = np.concatenate(vals, axis=cat_dim)
            split_vals = np.split(val, split_factor, axis=cat_dim)
            save_split(split_vals, saved_dir, trt_llm_key, tp_rank, split_factor)

        if act_range is not None and int8_outputs == "all":
            base_key = trt_llm_key.replace(".weight", "")
            vals_i8 = generate_int8(val, act_range, multi_query_mode=multi_query_mode)
            write_int8(vals_i8, saved_dir, base_key, cat_dim, tp_rank, split_factor)

    elif any_word_in_key(key, attention_qkv_bias_keys):
        qkv_hidden_dim = vals[0].shape[0]
        size_per_head = qkv_hidden_dim // (num_attention_heads + 2 * num_kv_heads)
        q_num = num_attention_heads // num_kv_heads

        # We first concat all sub weights per tp rank together.
        len_vals = len(vals)
        if convert_on_device:
            val = vals[0]
        else:
            val = np.concatenate(vals, axis=0)
        val = val.reshape(num_kv_heads * len_vals // tp_size, q_num + 2, size_per_head)

        # Split the QKV to separate variables.
        if convert_on_device:
            qkv = torch.split(val, [q_num, 1, 1], dim=1)
            split_vals = torch.concatenate([qkv[0].reshape(-1), qkv[1].reshape(-1), qkv[2].reshape(-1)], dim=1)
            save_val(split_vals, saved_dir, trt_llm_key)
        else:
            qkv = np.split(val, [q_num, q_num + 1], axis=1)
            q_split = np.split(qkv[0], split_factor, axis=0)
            k_split = np.split(qkv[1], split_factor, axis=0)
            v_split = np.split(qkv[2], split_factor, axis=0)

            # Concatenate Q, K, and V together
            split_vals = [
                np.concatenate([q_split[i].reshape(-1), k_split[i].reshape(-1), v_split[i].reshape(-1)], axis=0)
                for i in range(split_factor)
            ]
            save_split(split_vals, saved_dir, trt_llm_key, tp_rank, split_factor)

    elif any_word_in_key(key, attention_qkv_weight_keys):
        assert use_attention_nemo_shape, "Only support NEMO shape for QKV weights"
        hidden_dim = vals[0].shape[0]
        if size_per_head is None:
            size_per_head = hidden_dim // num_attention_heads
        q_num = num_attention_heads // num_kv_heads

        # When the merge factor exceeds 1, the 'vals' list will have multiple entries.
        # Depending on the format, 'vals' can look like either [QQQQ..KV, QQQQ..KV, ...](for GQA) or [QKV, QKV, ...](for MHA).
        # We first concat all sub weights per tp rank together.
        if convert_on_device:
            val = vals[0].reshape(hidden_dim, num_kv_heads // tp_size, q_num + 2, size_per_head)
            qkv = torch.split(val, [q_num, 1, 1], dim=2)
            split_vals = torch.concatenate(
                [qkv[0].reshape(hidden_dim, -1), qkv[1].reshape(hidden_dim, -1), qkv[2].reshape(hidden_dim, -1)], dim=1
            )
            save_val(split_vals, saved_dir, trt_llm_key)
        else:
            len_vals = len(vals)
            val = np.concatenate(vals, axis=1)
            val = val.reshape(hidden_dim, num_kv_heads * len_vals // tp_size, q_num + 2, size_per_head)

            # Split the QKV to separate variables.
            qkv = np.split(val, [q_num, q_num + 1], axis=2)

            query_groups_shape = qkv[0].shape
            if len(query_groups_shape) > 1:
                if (query_groups_shape[1] % split_factor) != 0:
                    raise Exception(
                        "Number of query groups of the models is {0}. Please select tensor parallelism size "
                        "that can split the number of query groups to equal number of query matrices in the "
                        "each GPU.".format(query_groups_shape[1])
                    )

            q_split = np.split(qkv[0], split_factor, axis=1)
            k_split = np.split(qkv[1], split_factor, axis=1)
            v_split = np.split(qkv[2], split_factor, axis=1)

            # Concatenate Q, K, and V together
            split_vals = [
                np.concatenate(
                    [
                        q_split[i].reshape(hidden_dim, -1),
                        k_split[i].reshape(hidden_dim, -1),
                        v_split[i].reshape(hidden_dim, -1),
                    ],
                    axis=1,
                )
                for i in range(split_factor)
            ]
            save_split(split_vals, saved_dir, trt_llm_key, tp_rank, split_factor)

        if save_int8:
            base_key = trt_llm_key.replace(".weight", "")
            vals_i8 = generate_int8(val, act_range, is_qkv=True, multi_query_mode=multi_query_mode)
            write_int8(
                vals_i8,
                saved_dir,
                base_key,
                cat_dim,
                tp_rank,
                split_factor,
                kv_cache_only=int8_outputs == "kv_cache_only",
            )

        if use_fp8_kv_cache:
            base_key = trt_llm_key.replace('.qkv.weight', '')
            scaling_factor = np.array([1.0], dtype=np.float32)
            save_val(scaling_factor, dir, base_key + '.kv_cache_scaling_factor')

    elif any_word_in_key(key, attention_not_mapped_keys):
        pass

    elif any_word_in_key(key, mlp_router_keys):
        val = np.concatenate(vals, axis=1)
        save_val(val, saved_dir, trt_llm_key)

    elif any_word_in_key(key, mlp_fc_expert_keys):
        cat_dim = -1
        val = np.concatenate(vals, axis=cat_dim)
        w1, w3 = np.split(val, 2, axis=1)
        # w1 splits
        split_w1s = np.split(w1, split_factor, axis=1)
        # w3 splits
        split_w3s = np.split(w3, split_factor, axis=1)

        split_vals = [np.concatenate(item, axis=1) for item in zip(split_w3s, split_w1s)]
        save_expert_split(split_vals, saved_dir, trt_llm_key, tp_rank, split_factor)

    elif any_word_in_key(key, mlp_proj_experts_keys):
        cat_dim = -1
        val = np.concatenate(vals, axis=cat_dim)
        split_vals = np.split(val, split_factor, axis=cat_dim)
        save_expert_split(split_vals, saved_dir, trt_llm_key, tp_rank, split_factor)
    else:
        print(f"[WARNING] {key} not handled by converter")

    global weights_dict
    return weights_dict


def split(v: Union[np.ndarray, torch.Tensor], tp_size: int, idx: int, dim: int = 0):
    """Splits the np tensor v on dim and return the idx's slice."""
    if tp_size == 1:
        return v

    dim = dim if len(v.shape) != 1 else 0
    if torch.is_tensor(v):
        return torch.split(v, v.size(dim) // tp_size, dim=dim)[idx].contiguous()

    return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])


def init_model_parallel_from_nemo(reshard_model):
    from megatron.core import parallel_state

    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    dp_size = parallel_state.get_data_parallel_world_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    dp_rank = parallel_state.get_data_parallel_rank()

    if reshard_model and pp_size > 1:
        dp_size = dp_size * pp_size
        dp_rank = torch.distributed.get_rank() // tp_size
        pp_rank = 0
        pp_size = 1

    mp_rank = tp_size * pp_rank + tp_rank
    # Need to split cpp MPI World Comm because TensorRT-LLM NCCL plugins refer to the locally split comm.
    # High level call structure is: MpiComm::split -> MpiComm::setSession -> LOCAL_COMM_SESSION (used in allReducePlugin.cpp)
    tensorrt_llm.bindings.MpiComm.split(dp_rank, mp_rank)
    # Also split the python mpi communicator and set the global world one to the local split one
    new_comm = mpi_comm().Split(color=dp_rank, key=mp_rank)
    from mpi4py import MPI

    MPI.COMM_WORLD = new_comm

    return mp_rank, dp_rank, tp_size, pp_size, dp_size
