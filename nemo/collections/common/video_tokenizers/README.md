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

**Note:** TensorRT acceleration functionality has been moved to the Export-Deploy repository:
```bash
pip install git+https://github.com/NVIDIA-NeMo/Export-Deploy.git
```

For TensorRT acceleration examples and documentation, please refer to:
https://github.com/NVIDIA-NeMo/Export-Deploy

# Examples
1. Multimodal autoregressive model dataset preparation using the [discrete cosmos tokenizer](../../../../nemo/collections/multimodal_autoregressive/data/README.md)
2. Diffusion model dataset preparation using the [continuous cosmos tokenizer](../../diffusion/data/readme.rst)
