DFlash speculative decoding with vLLM
======================================

The NeMo SpeechLM vLLM plugin supports checkpoint-backed DFlash and DFlash2
speculative decoding. Both use intermediate hidden states from the SpeechLM
language tower to condition a separate draft model; generated draft tokens are
verified by the target model, so accepted output remains lossless relative to
the target.

Requirements
------------

* A vLLM-ready NeMo SpeechLM checkpoint whose language backbone is compatible
  with the DFlash draft.
* vLLM 0.27.1 or later for the published Nemotron 3.5 Lightning recipe.
* An attention backend that supports the draft model's non-causal attention.

The following example uses the published NVFP4 DFlash draft for the Nemotron
3.5 Lightning 30B-A3B backbone and proposes six tokens per decoding step:

.. code-block:: bash

   vllm serve /path/to/vllm-ready-speechlm-checkpoint \
     --trust-remote-code \
     --speculative-config '{
       "method": "dflash",
       "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash",
       "num_speculative_tokens": 6
     }'

The target SpeechLM checkpoint must use
``nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`` as its language backbone.
The draft checkpoint provides the auxiliary target-layer selection and mask
token configuration consumed by vLLM; no draft weights are bundled with NeMo.

DFlash2 inference
-----------------

DFlash2 adds dynamic convolutions and a candidate-path selector to the draft
model. Its checkpoint must be trained or fine-tuned separately for the target
language backbone; NeMo's vLLM inference plugin does not create or convert
DFlash2 weights.

At the time of writing, DFlash2 requires the vLLM implementation from pull
request 52816. It uses the same ``method`` value as DFlash; vLLM selects the
DFlash2 runtime from the trained draft checkpoint's architecture:

.. code-block:: bash

   pip install -U "vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/52816/head"

   vllm serve /path/to/vllm-ready-speechlm-checkpoint \
     --trust-remote-code \
     --speculative-config '{
       "method": "dflash",
       "model": "/path/to/trained-lightning-dflash2-checkpoint",
       "num_speculative_tokens": 6
     }'

The trained draft config must declare ``DFlash2DraftModel``. Its
``dflash_config`` must include ``target_layer_ids``, ``conv_group_size``,
``conv_kernel_size``, ``selector_rank``, and ``selector_top_k``. Set
``num_speculative_tokens`` to one less than the convolution block size used to
train the draft: vLLM constructs each runtime block from one anchor plus the
configured number of draft tokens and does not reject a training/inference
block-size mismatch.

The DFlash2 architecture forces vLLM's V2 model runner, including for hybrid
NemotronH targets that would otherwise use V1. vLLM raises an error for features
it knows are incompatible with V2, but the target and serving configuration
should still be qualified on that runner. The SpeechLM target uses the same
``SupportsEagle3`` hidden-state contract for DFlash and DFlash2, so no draft
weights or training logic are bundled with NeMo.

Validation
----------

Compare greedy generation with and without ``--speculative-config`` using the
same text and audio prompts. The generated token IDs must match. Also inspect
vLLM's speculative-decoding metrics to confirm that draft tokens are proposed
and accepted; matching output alone does not prove that DFlash was active.
