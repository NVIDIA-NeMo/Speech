DFlash speculative decoding with vLLM
======================================

The NeMo SpeechLM vLLM plugin supports checkpoint-backed DFlash speculative
decoding. DFlash uses intermediate hidden states from the SpeechLM language
tower to condition a separate draft model; generated draft tokens are verified
by the target model, so accepted output remains lossless relative to the target.

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

Validation
----------

Compare greedy generation with and without ``--speculative-config`` using the
same text and audio prompts. The generated token IDs must match. Also inspect
vLLM's speculative-decoding metrics to confirm that draft tokens are proposed
and accepted; matching output alone does not prove that DFlash was active.
