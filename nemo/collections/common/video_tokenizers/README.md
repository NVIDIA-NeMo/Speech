# Cosmos Tokenizer

This directory contains the NeMo implementation of the [NVIDIA Cosmos Tokenizers](https://github.com/NVIDIA/Cosmos-Tokenizer)
that are hosted on the [Huggingface Hub (HF-Hub)](https://huggingface.co/nvidia/)

## Usage

### Basic usage
The encoder, decoder and autoencoder models can be loaded directly from the HF-Hub using the `from_pretrained` class method
of the `CausalVideoTokenizer` class:

```python
from nemo.collections.common.video_tokenizers.cosmos_tokenizer import CausalVideoTokenizer

model = CausalVideoTokenizer.from_pretrained("Cosmos-Tokenizer-DV4x8x8")
```
By default, this will download all three (`{encoder, decoder, autoencoder}.jit`) models from `nvidia/Cosmos-Tokenizer-DV4x8x8`
and will only load the encoder and decoder models.

To encode an input tensor, users can run the following:
```python
import torch
input_tensor = torch.randn(1, 3, 9, 512, 512).to('cuda').to(torch.bfloat16)
(indices, codes) = model.encode(input_tensor)
```

Please see the official [NVIDIA Cosmos repository](https://github.com/NVIDIA/Cosmos-Tokenizer)
for the complete list of supported tokenizers.

### Acceleration with TensorRT
To use these tokenizers with TensorRT and acheive up to 3X speedup during tokenization,
users can define a lightweight wrapper model and then pass this wrapper model to `trt_compile`
```python
import torch
from nemo.collections.common.video_tokenizers.cosmos_tokenizer import CausalVideoTokenizer
from nemo.export.tensorrt_lazy_compiler import trt_compile

class VaeWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, input_tensor):
        output_tensor = self.vae.autoencode(input_tensor)
        return output_tensor

model = CausalVideoTokenizer.from_pretrained(
    "Cosmos-Tokenizer-DV4x8x8", 
    use_pytorch=True, 
    dtype="float"
)
model_wrapper = VaeWrapper(model)

input_tensor = torch.randn(1, 3, 9, 512, 512).to('cuda').to(torch.float)
opt_shape = min_shape = max_shape = input_tensor.shape

path_to_engine_outputs="./trt_outputs"
trt_compile(
    model_wrapper,
    path_to_engine_outputs,
    args={
        "precision": "bf16",
        "input_profiles": [
            {"input_tensor": [min_shape, opt_shape, max_shape]},
        ],
    },
)

output = model_wrapper(input_tensor)
```
Note that the `trt_compile` function requires 
providing `min_shape`, `opt_shape` and `max_shape`
as arguments (in this example all are set to the input tensor shape for simplicity) which enables inputs with dynamic shapes after compilation.
For more information about TensorRT and dynamic shapes please review the [Torch-Tensorrt documentation](https://pytorch.org/TensorRT/user_guide/dynamic_shapes.html)

The file `cosmos_trt_run.py` provides a stand-alone script to tokenize tensors with a TensorRT-accelerated
Cosmos tokenizer.

# Examples
1. Multimodal autoregressive model dataset preparation using the [discrete cosmos tokenizer](../../../../nemo/collections/multimodal_autoregressive/data/README.md)
2. Diffusion model dataset preparation using the [continuous cosmos tokenizer](../../diffusion/data/readme.rst)
