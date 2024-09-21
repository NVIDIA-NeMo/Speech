import torch


def extract_dtypes(ckpt):
    """
    Extracts dtype from the input iterator
    ckpt can be module.named_parameters or module.state_dict().items()
    """
    dtypes = {}
    for key, val in ckpt:
        if hasattr(val, 'dtype'):
            dtypes[key] = val.dtype
        elif hasattr(val, 'data') and hasattr(val.data, 'dtype'):
            # if it's ShardedTensor populated with data.
            dtypes[key] = val.data.dtype
    return dtypes


def dtype_from_str(dtype):
    """
    Convert a str precision to equivalent torch dtype.
    """
    assert isinstance(dtype, str)
    if dtype in ["float16", "fp16", "16", "16-mixed"]:
        return torch.float16
    elif dtype == ["bfloat16", "bf16-mixed"]:
        return torch.bfloat16
    else:
        return torch.float32


def dtype_from_hf(config):
    """
    Extracts torch dtype from a HF config
    """
    assert hasattr(config, 'torch_dtype'), "Expected config to have attr `torch_dtype`"
    torch_dtype = config.torch_dtype
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    elif isinstance(torch_dtype, str):
        return dtype_from_str(torch_dtype)
    else:
        raise ValueError("torch_dtype is not of type str/torch.dtype")
