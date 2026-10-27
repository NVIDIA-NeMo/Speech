# Serve a TDT decoder with Triton

This example decodes encoder outputs into token IDs, timestamps and durations in one Triton request.
It uses NeMo's PyTorch prediction/joint networks and greedy decoding loop, preserving cached computations and optional CUDA graphs.

## Package and run

```bash
python examples/asr/export/transducer/export_tdt_decoder.py \
    --model model.nemo --output model_repository/tdt_decoder --max-batch-size 16
tritonserver --model-repository=model_repository
```

The Triton Python backend needs this Speech checkout, its inference dependencies, compatible PyTorch/cuda-python, and a matching Python stub.
See [Triton's environment guide](https://github.com/triton-inference-server/python_backend#managing-python-runtime-and-libraries).

The export creates `config.pbtxt`, `1/model.py` and `1/decoder.pt`.
Load only trusted artifacts; their configuration instantiates NeMo classes.
Parameters use FP32 with TF32 disabled, preserving the checkpoint's `decoding.greedy.max_symbols` limit.

## Inputs and outputs

| Tensor | Type | Shape |
| --- | --- | --- |
| Input `ENCODER_OUTPUT` | FP32 | `[B, C, T]` |
| Input `ENCODED_LENGTH` | INT64 | `[B, 1]` |
| Output `TOKEN_LENGTH` | INT64 | `[B, 1]` |
| Outputs `TOKENS`, `TIMESTAMPS`, `TOKEN_DURATIONS` | INT64 | `[B, U]` |

Here, `B` is batch size, `C` encoder channels, `T` padded frames and `U` padded tokens.
Only entries before each token length are valid.
Token/timestamp padding is -1; duration padding is zero.
Empty utterances have zero token lengths and at least one padding column.
Timestamps and durations use encoder frames; convert token IDs to text with the checkpoint's tokenizer.

Triton batches requests with matching padded frame counts; retain each utterance's true encoded length.
Invalid lengths are rejected per request.
Each model instance owns separate weights/buffers and handles calls serially.
Warm up intended batch/frame sizes before timing.

For direct Python use:

```python
from nemo.collections.asr.parts.submodules.tdt_decoder import TDTDecoder

decoder = TDTDecoder.load("model_repository/tdt_decoder/1/decoder.pt", device="cuda")
result = decoder(encoded, encoded_lengths)  # Shapes: [B, C, T] and [B].
```

Preprocessing, the encoder and tokenization remain separate.
Beam search, language-model fusion, confidence outputs and streaming state are unsupported.
See [TDTDecoder](../../../../nemo/collections/asr/parts/submodules/tdt_decoder.py) and the [Triton adapter](tdt_triton_model.py) for execution details.

## Check a running server

```bash
NEMO_TDT_TRITON_URL=http://localhost:8000 \
NEMO_TDT_DECODER_ARTIFACT=model_repository/tdt_decoder/1/decoder.pt \
pytest tests/collections/asr/test_tdt_triton.py
```

The CUDA test compares all outputs with direct decoding and checks changing shapes, empty utterances, concurrent requests and recovery from invalid lengths.
Set `NEMO_TDT_TRITON_MODEL` for another model name.
To require observed batching, set a nonzero `max_queue_delay_microseconds` in Triton's `dynamic_batching` and `NEMO_TDT_TRITON_REQUIRE_BATCHING=1`.
The test leaves the server running.
