Speculative decoding
====================

A draft model proposes tokens that the target model verifies. For SpeechLM,
DFlash and DFlash2 drafts use intermediate audio-conditioned target features.
Start with inference using a compatible exported draft, then see training to
adapt a draft to a different target checkpoint.

Inference
---------

.. toctree::
   :maxdepth: 2

   vllm_dflash

Training
--------

.. toctree::
   :maxdepth: 2

   dflash_training
