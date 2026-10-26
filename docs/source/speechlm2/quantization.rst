Quantization for vLLM (FP8 and NVFP4)
=====================================

The scripts in ``scripts/speechlm2/quantization/`` quantize a NeMo SpeechLM
checkpoint whose language model is a NemotronH decoder (hybrid Mamba and MoE),
for serving with vLLM and the NeMo SpeechLM plugin. The result is one
checkpoint whose ``config.json`` carries a ModelOpt ``quantization_config``.
vLLM builds the quantized decoder layers from it, and the plugin builds the FP8
audio encoder from the same config.

What the Recipes Quantize
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Part
     - ``fp8``
     - ``nvfp4``
   * - Decoder: routed and shared MoE experts, attention projections
     - FP8
     - NVFP4
   * - Decoder: Mamba ``in_proj`` and ``out_proj``
     - FP8
     - FP8
   * - Decoder: ``lm_head``, Mamba ``conv1d``, MoE routers
     - BF16
     - BF16
   * - KV cache
     - FP8 with calibrated scales
     - FP8 at serving time (``--kv-cache-dtype fp8``)
   * - Audio encoder: ``attn.w_qkv``, ``attn.out_proj``, ``ffn.net.0`` and
       ``ffn.net.3`` of each layer
     - FP8
     - FP8
   * - MTP draft head (its router stays BF16)
     - FP8
     - FP8
   * - CTC timestamp head
     - BF16
     - BF16

FP8 layers have a per-tensor weight scale and a static per-tensor input scale.
NVFP4 layers use ModelOpt's NVFP4 format: 4-bit values in blocks of 16, with an
FP8 scale per block and a per-tensor scale. Mamba's ``A_log``, ``D`` and
``dt_bias`` keep the FP32 values of the source checkpoint.

Expect FP8 to match BF16 within run-to-run noise. NVFP4 trades a small accuracy
cost, larger for some languages than for English, for more throughput. Measure
both on your own test sets (see `Validation`_).

How the Pipeline Works
----------------------

The decoder, the audio encoder and the MTP draft head are quantized in separate
stages, because each one needs a different tool:

* ModelOpt quantizes Hugging Face models. Step 1 exports the decoder as a
  Hugging Face NemotronH checkpoint, and step 2 quantizes it with ModelOpt,
  calibrated on text.
* Hugging Face's NemotronH has no MTP layers and discards their weights at
  load. Step 3 copies the MTP head, the audio encoder and the CTC timestamp
  head byte for byte from the source checkpoint and puts them next to the
  quantized decoder. It also restores the decoder's FP32 Mamba parameters,
  which the BF16 export rounds. vLLM can already serve the result.
* The audio encoder and the MTP head need activation ranges from real audio,
  and the draft head reads the quantized decoder's hidden states. Step 4
  records these ranges while vLLM serves the step 3 checkpoint with MTP
  speculative decoding on.
* Step 5 writes the encoder and the MTP head in FP8 from those ranges and adds
  them to the checkpoint's quantization config.

Requirements
------------

* A GPU with at least 80 GB of memory to quantize the decoder, which loads in
  BF16. Serving FP8 needs SM89 or newer; serving NVFP4 needs Blackwell (SM100
  or SM120).
* A ModelOpt build that includes
  `NVIDIA/Model-Optimizer#2508 <https://github.com/NVIDIA/Model-Optimizer/pull/2508>`__
  (tested with ``main`` at ``54d44161e``). Earlier builds write module names
  that vLLM cannot resolve, and ``assemble_checkpoint.py`` refuses their
  exports.
* ``transformers`` with NemotronH support, ``torch``, ``safetensors`` and
  ``datasets`` for step 2.
* The default FP8 calibration reads the gated
  ``nvidia/Nemotron-Post-Training-Dataset-v2`` dataset. Log in to Hugging Face
  with access to it, or pass ``--calib-dataset cnn_dailymail``.
* For calibration (step 4) and serving: a NeMo build with the SpeechLM plugin's
  quantized serving
  (`NVIDIA-NeMo/Speech#16341 <https://github.com/NVIDIA-NeMo/Speech/pull/16341>`__),
  installed with its ``vllm`` extra (vLLM 0.28.0):
  ``pip install -e ".[asr,vllm]"``. See :doc:`vllm_dflash` for the extras that
  must not be combined with it.
* Serving the CTC timestamp head needs the plugin's CTC timestamp support
  (`NVIDIA-NeMo/Speech#16344 <https://github.com/NVIDIA-NeMo/Speech/pull/16344>`__).
  Without it, assemble with ``--skip-ctc-head``.
* Working space on top of the source checkpoint. Keeping the BF16 decoder
  export, the quantized decoder and the assembled checkpoint at the same time
  takes about 2 times the size of the BF16 checkpoint for FP8 and 1.5 times for
  NVFP4. Step 3 rewrites every decoder shard that holds Mamba parameters (for
  the decoders we tested, all of them), so ``--hardlink`` saves little. The
  BF16 decoder export can be deleted after step 2, and the quantized decoder
  after step 3. Step 5 hard-links unchanged files into ``$OUT``; if ``$OUT`` is
  on another filesystem than ``$WORK``, it copies them instead, so budget one
  more copy of the final checkpoint.

Steps 1 and 5 need only ``torch`` and ``safetensors``, and step 3 only the
Python standard library. Step 2 (ModelOpt) and step 4 (vLLM) can therefore run
in separate environments.

Steps
-----

Run the scripts from ``scripts/speechlm2/quantization/``:

.. code-block:: bash

   SRC=/models/speechlm-bf16        # BF16 SpeechLM checkpoint
   WORK=/work/quant
   RECIPE=nvfp4                     # or fp8
   OUT=/models/speechlm-$RECIPE

1. Export the decoder as a Hugging Face NemotronH checkpoint:

   .. code-block:: bash

      python export_llm_to_hf.py --checkpoint $SRC --output $WORK/llm_bf16

2. Quantize the decoder with ModelOpt (128 calibration samples of up to 512
   tokens):

   .. code-block:: bash

      python quantize_decoder.py --llm $WORK/llm_bf16 --recipe $RECIPE \
         --output $WORK/llm_$RECIPE

3. Assemble a SpeechLM checkpoint with the quantized decoder. Its audio encoder
   and MTP head are still BF16:

   .. code-block:: bash

      python assemble_checkpoint.py --source $SRC \
         --quantized-llm $WORK/llm_$RECIPE \
         --output $WORK/ckpt_bf16_heads --hardlink

4. Calibrate the audio encoder and the MTP head. Add ``--kv-cache-dtype fp8``
   for NVFP4:

   .. code-block:: bash

      VLLM_PLUGINS=nemo_speechlm VLLM_WORKER_MULTIPROC_METHOD=spawn \
      python calibrate_activations.py \
         --checkpoint $WORK/ckpt_bf16_heads \
         --manifest calib.jsonl --output $WORK/ranges.json

   ``calib.jsonl`` holds one ``{"audio_filepath": ...}`` per line. Use a few
   hundred held-out clips that resemble your traffic, never your test sets.

5. Write the FP8 audio encoder and the FP8 MTP head into the final checkpoint.
   Unchanged files are hard-linked, or copied across filesystems:

   .. code-block:: bash

      python quantize_encoder_mtp_fp8.py \
         --checkpoint $WORK/ckpt_bf16_heads \
         --activation-ranges $WORK/ranges.json --output $OUT

A script replaces an existing output only when it offers ``--overwrite`` and
you pass it. Every script refuses an output that is, contains or lies inside
one of its inputs.

Serving
-------

.. code-block:: bash

   VLLM_PLUGINS=nemo_speechlm VLLM_WORKER_MULTIPROC_METHOD=spawn \
   vllm serve $OUT --trust-remote-code --dtype bfloat16 \
      --kv-cache-dtype fp8 \
      --speculative-config '{"method":"mtp","num_speculative_tokens":2}'

.. note::
   Use ``--kv-cache-dtype fp8`` for NVFP4 checkpoints. ModelOpt's NVFP4 export
   declares no KV cache scheme, so ``auto`` would keep the KV cache in BF16. FP8
   checkpoints carry calibrated KV cache scales and serve with ``auto``.

* For word timestamps from the CTC timestamp head, add
  ``--middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware``.
* On SM12x GPUs with cuDNN older than 9.23.1, set
  ``VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel``. The FlashInfer
  FP8 GEMM can crash partway through a run there.

The server log should show
``FP8 audio encoder: N Linears built from quantization_config`` and, for NVFP4,
``Using fp8 data type to store kv cache``.

Validation
----------

* Compare against the BF16 checkpoint served by the same NeMo build, with the
  same prompt and decoding settings. A different build, prompt or decoding
  setup can move WER more than quantization does.
* Report per-dataset and per-language deltas, not only averages. Quantization
  costs concentrate in a few languages.
* Serve each checkpoint at least twice to learn the run-to-run noise of batched
  serving before you call a delta real. Quantized checkpoints can vary more
  between runs than BF16.
* With MTP on, compare the draft acceptance in vLLM's ``SpecDecoding`` log
  lines with the BF16 checkpoint's on the same audio.
* Measure throughput on otherwise idle GPUs; parallel jobs perturb it.

Checks Built into the Scripts
-----------------------------

* ``quantize_decoder.py`` reports how many quantizers calibration never reached
  (MoE experts that no calibration token was routed to) and gives them the
  largest positive range recorded by any quantizer, so the export does not
  fail. A large count means the calibration texts reached few experts.
* Copied tensors (perception weights, MTP and CTC heads, FP32 Mamba parameters)
  are verified against the source by SHA-256.
* ``assemble_checkpoint.py`` refuses ModelOpt exports with unresolved module
  names.
* ``calibrate_activations.py`` runs the MTP draft head on vLLM's Triton MoE
  backend, whose unpacked expert weights its observers read, and refuses a
  packed layout. It stops if it recorded no encoder or no MTP ranges, for
  example when speculative decoding did not run.
* ``quantize_encoder_mtp_fp8.py`` requires a calibrated range for every encoder
  Linear it quantizes. It fails unless every FP8 weight has scalar FP32 scales
  and every perception Linear is FP8 exactly when the config says so.
* Unit tests, run from the repository root:
  ``python -m pytest --noconftest tests/collections/speechlm2/test_quantization_scripts.py``.
  ``--noconftest`` skips the repository-wide test setup, which downloads test
  data these tests do not need.

Limitations
-----------

* The decoder must be a NemotronH model whose MoE experts use a squared-ReLU
  activation without a gate.
* ``calibrate_activations.py`` always runs MTP speculative decoding, so the
  checkpoint needs an MTP draft head (``llm.mtp.*`` tensors) that vLLM can
  load.
* The encoder targets assume the Transformer ASR encoder these SpeechLMs use.
  Another encoder needs its own target list in ``calibrate_activations.py`` and
  ``quantize_encoder_mtp_fp8.py``.
* The plugin's FP8 audio encoder supports only static per-tensor scales.
* ``calibrate_activations.py`` hooks into vLLM 0.28 internals (the worker's
  draft model and its MoE runner); other vLLM versions may need changes to it.
