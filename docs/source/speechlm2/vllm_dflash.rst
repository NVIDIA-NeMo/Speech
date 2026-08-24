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

DFlash2 bootstrap
-----------------

DFlash2 adds two-tap dynamic convolutions and a candidate-path selector. Until
a trained Lightning DFlash2 checkpoint is published, the existing trained
Lightning DFlash checkpoint can be converted into a functional DFlash2
bootstrap:

.. code-block:: bash

   python scripts/speechlm2/convert_dflash_to_dflash2.py \
     nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash \
     /path/to/lightning-dflash2-bootstrap

The converter preserves the trained draft backbone, initializes both
convolutions as exact identities, and initializes the selector as a no-op. It
also preserves an optional ``mask_embedding.pt`` and excludes the new BF16
modules from ModelOpt quantization metadata. The source must store its weights
in a single safetensors file. Its rank-256, top-k-16 selector defaults keep the
bootstrap memory-representative rather than minimizing its footprint. Output
files use container-readable model-artifact
permissions (``0755`` directory and ``0644`` files). The bootstrap therefore
validates the DFlash2 runtime integration but does not claim the acceptance
improvement of a checkpoint whose DFlash2 parameters were trained.

At the time of writing, DFlash2 requires the vLLM implementation from pull
request 52816. It uses the same ``method`` value as DFlash; vLLM selects the
DFlash2 runtime from the draft checkpoint architecture:

.. code-block:: bash

   pip install -U "vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/52816/head"

   vllm serve /path/to/vllm-ready-speechlm-checkpoint \
     --trust-remote-code \
     --speculative-config '{
       "method": "dflash",
       "model": "/path/to/lightning-dflash2-bootstrap",
       "num_speculative_tokens": 6
     }'

The generated config declares ``DFlash2DraftModel``. That architecture forces
vLLM's V2 model runner; vLLM raises an error if another requested feature is
incompatible with that runner. The runtime derives its convolution block size
from ``num_speculative_tokens`` (seven positions in the example: one anchor plus
six draft tokens). The SpeechLM target uses the same ``SupportsEagle3``
hidden-state contract for both DFlash versions.

Validation
----------

Compare greedy generation with and without ``--speculative-config`` using the
same text and audio prompts. The generated token IDs must match. Also inspect
vLLM's speculative-decoding metrics to confirm that draft tokens are proposed
and accepted; matching output alone does not prove that DFlash was active.
