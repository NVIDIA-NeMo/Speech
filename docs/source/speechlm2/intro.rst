SpeechLM2
================================

.. note::
   The SpeechLM2 collection is still in active development and the code is likely to keep changing.

.. note::
   Install your chosen compatible PyTorch stack first, then install SpeechLM2 with
   ``uv pip install 'nemo-toolkit[speechlm2]'`` (or, from a source checkout, ``uv pip install -e '.[speechlm2]'``)
   to get all required dependencies including NeMo Automodel. See :ref:`installation` for details.

SpeechLM2 refers to a collection that augments pre-trained Large Language Models (LLMs) with speech understanding and generation capabilities.

This collection is designed to be compact, efficient, and to support easy swapping of different LLMs backed by HuggingFace AutoModel or NeMo Automodel. 
It has a first-class support for using dynamic batch sizes via Lhotse and various model parallelism techniques (e.g., FSDP2, Tensor Parallel, Sequence Parallel) via PyTorch DTensor API.

We currently support six main model types:

* **SALM** (Speech-Augmented Language Model) - a simple but effective approach to augmenting pre-trained LLMs with speech understanding capabilities. Available in two variants:

  * ``SALM`` — uses HuggingFace Transformers for the LLM backbone with optional HF PEFT LoRA.
  * ``SALMAutomodel`` — uses `NeMo Automodel <https://github.com/NVIDIA-NeMo/Automodel>`_ for the LLM backbone with native LoRA, advanced parallelism (FSDP2, TP, SP, EP via ``AutomodelParallelStrategy``), and MoE optimizations (Grouped GEMM, DeepEP) for efficient training with models like `NVIDIA Nemotron Nano V3 <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16>`_.
* **DuplexS2SModel** - a full-duplex speech-to-speech model with an ASR encoder, directly predicting discrete audio codes.
* **DuplexS2SSpeechDecoderModel** - a variant of DuplexS2SModel with a separate transformer decoder for speech generation.
* **DuplexEARTTS** - a ready-to-use duplex text-to-speech model that supports user interruption via a special text interruption token.
* **DuplexSTTModel** - a decoder model to generate agent text in duplex, in response to both user speech and text inputs.
* **NemotronVoiceChat** - an *inference-only* pipeline that seamlessly merges `DuplexSTTModel` and `DuplexEARTTS` to deliver an end-to-end, full-duplex conversational agent with high-fidelity speech generation.

Using Pretrained Models
-----------------------

After :ref:`installing NeMo<installation>`, you can load and use a pretrained speechlm2 model as follows:

.. code-block:: python

    import nemo.collections.speechlm2 as slm
    
    # Load a pretrained SALM model
    model = slm.models.SALM.from_pretrained("model_name_or_path")

    # Set model to evaluation mode
    model = model.eval()

Inference with Pretrained Models
--------------------------------

SALM
****

You can run inference using the loaded pretrained SALM model:

.. code-block:: python

    import torch
    import soundfile as sf
    from nemo.collections.audio.parts.utils.transforms import resample
    import nemo.collections.speechlm2 as slm

    model = slm.models.SALM.from_pretrained("path/to/pretrained_checkpoint").eval()
    
    # Load audio file
    audio_path = "path/to/audio.wav"
    audio_signal, sample_rate = sf.read(audio_path)
    audio_signal = torch.tensor(audio_signal).unsqueeze(0)
    
    # Resample if needed
    if sample_rate != 16000:  # Most models expect 16kHz audio
        audio_signal = resample(audio_signal, sample_rate, 16000)
        sample_rate = 16000
    
    # Prepare audio for model
    audio_signal = audio_signal.to(model.device)
    audio_len = torch.tensor([audio_signal.shape[1]], device=model.device)
    
    # Create a prompt for SALM model inference
    # The audio_locator_tag is a special token that will be replaced with audio embeddings
    prompt = [{"role": "user", "content": f"{model.audio_locator_tag}"}]
    
    # Generate response
    with torch.inference_mode():
        output = model.generate(
            prompts=[prompt],
            audios=audio_signal,
            audio_lens=audio_len,
            generation_config=None  # You can customize generation parameters here
        )
    
    # Process the output tokens
    response = model.tokenizer.ids_to_text(output[0])
    print(f"Model response: {response}")

SALMAutomodel
*************

``SALMAutomodel`` is the NeMo Automodel variant of SALM. It enables efficient training of
Speech LLMs with MoE architectures like `NVIDIA Nemotron Nano V3 <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16>`_
using MoE-specific optimizations (Grouped GEMM, DeepEP). It uses deferred initialization
(``configure_model()``) and supports distributed training and inference via
``AutomodelParallelStrategy``.

.. code-block:: python

    import torch
    import nemo.collections.speechlm2 as slm
    from nemo.collections.speechlm2.parts.parallel import setup_distributed

    # Initialize distributed and resolve the Automodel topology with EP=2.
    # The strategy packages the mesh and execution policies in distributed_setup.
    strategy = setup_distributed(ep_size=2)

    # Load a pretrained SALMAutomodel with the resolved distributed setup.
    model = slm.models.SALMAutomodel.from_pretrained(
        "path/to/checkpoint",
        distributed_setup=strategy.distributed_setup,
    ).eval()

    # Inference is identical to SALM
    with torch.inference_mode():
        output = model.generate(
            prompts=[prompt],
            audios=audio_signal,
            audio_lens=audio_len,
        )

DuplexS2SModel
**************

You can run inference using the loaded pretrained DuplexS2SModel:

.. code-block:: python

    import torch
    import soundfile as sf
    from nemo.collections.audio.parts.utils.transforms import resample
    import nemo.collections.speechlm2 as slm

    model = slm.models.DuplexS2SModel.from_pretrained("path/to/pretrained_checkpoint").eval()
    
    # Load audio file
    audio_path = "path/to/audio.wav"
    audio_signal, sample_rate = sf.read(audio_path)
    audio_signal = torch.tensor(audio_signal).unsqueeze(0)
    
    # Resample if needed
    if sample_rate != 16000:  # Most models expect 16kHz audio
        audio_signal = resample(audio_signal, sample_rate, 16000)
        sample_rate = 16000
    
    # Prepare audio for model
    audio_signal = audio_signal.to(model.device)
    audio_len = torch.tensor([audio_signal.shape[1]], device=model.device)
    
    # Run offline inference
    results = model.offline_inference(
        input_signal=audio_signal,
        input_signal_lens=audio_len
    )

    # Decode text and audio tokens
    transcription = results["text"][0]
    audio = results["audio"][0]

DuplexSTTModel
**************

You can run inference using the loaded pretrained DuplexSTTModel:

.. code-block:: python

    import torch
    import soundfile as sf
    from nemo.collections.audio.parts.utils.transforms import resample
    import nemo.collections.speechlm2 as slm

    model = slm.models.DuplexSTTModel.from_pretrained("path/to/pretrained_checkpoint").eval()

    # Load audio file
    audio_path = "path/to/audio.wav"
    audio_signal, sample_rate = sf.read(audio_path)
    audio_signal = torch.tensor(audio_signal).unsqueeze(0)

    # Resample if needed
    if sample_rate != 16000:
        audio_signal = resample(audio_signal, sample_rate, 16000)
        sample_rate = 16000

    # Prepare audio for model
    audio_signal = audio_signal.to(model.device)
    audio_len = torch.tensor([audio_signal.shape[1]], device=model.device)

    # Run offline inference - generates text output only
    results = model.offline_inference(
        input_signal=audio_signal,
        input_signal_lens=audio_len
    )

    # Decode text tokens
    transcription = results["text"][0]
    print(f"Transcription: {transcription}")

DuplexEARTTS
************

Because `DuplexEARTTS` relies on precise token padding and EOS placement to handle potential user interruptions, inference and evaluation are handled via the `duplex_eartts_eval.py` script following the MagpieTTS dataset format recipe. 

The evaluation script processes a `JSONL` file where each line is a dictionary containing the text, the reference audio for the speaker, and the desired output audio filename. 

**JSONL Format Examples:**

Single-Turn format (evaluates a continuous string):

.. code-block:: json

    {"text": "Like really quickly and then they run off.", "context_audio_filepath": "speaker_1.wav", "audio_filepath": "audio_1.wav"}

Multi-Turn format (evaluates sequential conversational turns, padded incrementally):

.. code-block:: json

    {"text": ["Yes.", "Sure.", "Right.", "I get what you’re saying."], "context_audio_filepath": "speaker_2.wav", "audio_filepath": "audio_2.wav"}

**Running the Evaluation/Inference Script:**

.. code-block:: bash

    python examples/speechlm2/duplex_eartts_eval.py \
        --config-path=conf/ \
        --config-name=duplex_eartts.yaml \
        ++checkpoint_path=/path/to/duplex_eartts/model.ckpt \
        ++datasets_json_path=/path/to/evalset_config.jsonl \
        ++out_dir=/path/to/output/audio_samples/ \
        ++user_custom_speaker_reference=/path/to/optional_override_speaker.wav

The script will decode the text, apply the target speaker conditioning, generate the resulting audio waveforms into `out_dir`, and compute ASR intelligibility metrics (CER/WER) on the generated speech.

NemotronVoiceChat
*****************

You can evaluate and run full-duplex inference using the `NemotronVoiceChat` pipeline. This model natively chains the `DuplexSTTModel` with the `DuplexEARTTS` speech decoder for an end-to-end response:

.. code-block:: python

    import torch
    import soundfile as sf
    from nemo.collections.audio.parts.utils.transforms import resample
    import nemo.collections.speechlm2 as slm

    model = slm.models.NemotronVoiceChat.from_pretrained("path/to/pretrained_checkpoint").eval()

    # Load user audio prompt
    audio_path = "path/to/user_audio.wav"
    audio_signal, sample_rate = sf.read(audio_path)
    audio_signal = torch.tensor(audio_signal).unsqueeze(0)

    # Resample to the source_sample_rate (usually 16kHz for STT perception)
    if sample_rate != 16000:
        audio_signal = resample(audio_signal, sample_rate, 16000)
        sample_rate = 16000

    # Prepare audio for model
    audio_signal = audio_signal.to(model.device)
    audio_len = torch.tensor([audio_signal.shape[1]], device=model.device)

    # (Optional) Load an explicit speaker reference audio to condition the agent's voice
    # speaker_audio, _ = sf.read("path/to/speaker_reference.wav")
    # speaker_audio = torch.tensor(speaker_audio).unsqueeze(0).to(model.device)
    # speaker_len = torch.tensor([speaker_audio.shape[1]], device=model.device)

    # Note: If an explicit audio reference is not passed into `offline_inference`, 
    # the model relies on the internal config parameters:
    # 1. model.cfg.inference_speaker_name (Highest priority preset, e.g., 'Megan')
    # 2. model.cfg.inference_speaker_reference (Fallback audio file path)
        
    # Run full offline inference
    results = model.offline_inference(
        input_signal=audio_signal,
        input_signal_lens=audio_len,
        # speaker_audio=speaker_audio,       # Pass speaker reference if available
        # speaker_audio_lens=speaker_len
    )

    # Decode the predicted text and generated speech waveform
    generated_text = results["text"][0]
    generated_speech = results["audio"][0]
    
    print(f"Agent response: {generated_text}")
    # generated_speech can now be saved or played (sampled at model.target_sample_rate)
    

Training a Model
----------------

This example demonstrates how to train a SALM model. 

.. note::
   **NemotronVoiceChat is an inference-only class.** It does not implement a `training_step` and cannot be trained using the pipeline below. To update its underlying capabilities, you must train the `DuplexSTTModel` and `DuplexEARTTS` models independently.

.. code-block:: python

    from omegaconf import OmegaConf
    import torch
    from lightning.pytorch import Trainer
    from lightning.pytorch.strategies import ModelParallelStrategy
    
    import nemo.collections.speechlm2 as slm
    from nemo.collections.speechlm2.data import SALMDataset, DataModule
    from nemo.utils.exp_manager import exp_manager
    
    # Load configuration
    config_path = "path/to/config.yaml"  # E.g., from examples/speechlm2/conf/salm.yaml
    cfg = OmegaConf.load(config_path)
    
    # Initialize PyTorch Lightning trainer
    trainer = Trainer(
        max_steps=100000,
        accelerator="gpu",
        devices=1,
        precision="bf16-true",
        strategy=ModelParallelStrategy(data_parallel_size=2, tensor_parallel_size=1),
        limit_train_batches=1000,
        val_check_interval=1000,
        use_distributed_sampler=False,
        logger=False,
        enable_checkpointing=False,
    )
    
    # Set up experiment manager for logging
    exp_manager(trainer, cfg.get("exp_manager", None))
    
    # Initialize model with configuration
    model = slm.models.SALM(OmegaConf.to_container(cfg.model, resolve=True))
    
    # Create dataset and datamodule
    dataset = SALMDataset(tokenizer=model.tokenizer)
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)
    
    # Train the model
    trainer.fit(model, datamodule)

Example Using Command-Line Training Script
------------------------------------------

Alternatively, you can train a model using the provided training scripts in the examples directory:

.. code-block:: bash

    # Train a SALM model
    python examples/speechlm2/salm_train.py \
      --config-path=examples/speechlm2/conf \
      --config-name=salm

    # For SALM inference/evaluation
    python examples/speechlm2/salm_eval.py \
      pretrained_name=/path/to/checkpoint \
      inputs=/path/to/test_manifest \
      batch_size=64 \
      max_new_tokens=128 \
      output_manifest=generations.jsonl

To train the SALMAutomodel variant (with NeMo Automodel backend), use the ``salm_automodel`` config:

.. code-block:: bash

    # Train SALMAutomodel with NVIDIA Nemotron Nano V3 MoE backbone on 8 GPUs
    torchrun --nproc_per_node=8 examples/speechlm2/salm_train.py \
      --config-name=salm_automodel \
      model.pretrained_llm=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

The ``salm_automodel.yaml`` config sets ``model.use_nemo_automodel: true``, which selects the
``SALMAutomodel`` class. This variant supports ``AutomodelParallelStrategy`` for FSDP2/TP/EP
parallelism and MoE optimizations (Grouped GEMM, DeepEP).

DFlash and DFlash2 draft training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same config includes an optional ``dflash:`` section for training a compact DFlash or DFlash2
draft against a frozen, audio-conditioned ``SALMAutomodel`` target. Set ``dflash.enabled=true``,
choose ``dflash.variant`` (``dflash`` or ``dflash2``), and provide a reserved
``dflash.mask_token_id``; ``salm_train.py`` then trains and exports draft-only weights while
preserving SALM's audio-placeholder expansion. The shipped DFlash2 settings mirror Automodel's
recipe, including its two-tap grouped dynamic convolution, top-16 rank-256 path selector, and
separately normalized backbone and selector losses. ``num_anchors`` limits sampled blocks per row;
packed THD batches use one row per rank. For non-packed BSHD batches, up to
``batch_size * num_anchors`` blocks are allocated per rank; reduce ``num_anchors`` when using larger batches.
Packed-document boundaries are enforced by document-local anchor selection, attention masks, and position IDs.
Both variants use dense draft logits (DFlash2 needs them for candidate selection), with
``use_fused_linear_ce=false``. This integration supports BSHD and packed THD batches with
``tp_size=pp_size=cp_size=1`` and does not directly load the published packed NVFP4 inference
checkpoint into BF16 training modules.
Target features are captured with SALM's original padding; before the draft forward,
leading padding is moved to the end of each row so it cannot enter a draft context.

``trainer.precision`` controls both target and draft dtypes during training. For BF16
training, either the shipped ``bf16-flash`` or ``bf16-automodel`` trainer precision
is supported. Target inference backend compatibility is validated before constructing the draft.

``dflash.lr`` defaults to 6e-4 with AdamW and no scheduler. To reproduce an adaptation
schedule, supply ``dflash.optimizer`` and ``dflash.lr_scheduler`` explicitly; for example:

.. code-block:: yaml

   dflash:
     optimizer:
       _target_: torch.optim.AdamW
       lr: 1.5e-4
       betas: [0.9, 0.999]
       eps: 1e-8
       weight_decay: 0.0
     lr_scheduler:
       _target_: nemo.core.optim.lr_scheduler.CosineAnnealing
       warmup_steps: 500
       min_lr: 1e-6
       max_steps: 8000

For a frozen target initialized from a Hugging Face directory after FSDP sharding,
``model.init_from_checkpoint_strict=true`` (default) requires every model parameter
in that checkpoint. Set it to ``false`` only for intentional partial initialization.

To warm-start a draft, set ``dflash.init_from_pretrained`` to a local HuggingFace
draft export matching the configured architecture. Loading rejects missing,
unexpected, or mismatched weights. This initializes draft weights only; optimizer,
scheduler, and step counters start fresh. The frozen target is still selected
independently with the model's checkpoint settings.

By default, ``dflash.label_source=ground_truth`` uses the reference tokens. Set
``dflash.label_source=target_argmax`` to supervise the draft with the frozen
backbone's greedy next-token predictions under the original, teacher-forced
context. These are hard labels, not soft-logit distillation or generated
rollouts. The input tokens and audio-conditioned hidden features remain the
original context. Labels use the final-normalized state at the preceding token;
packed document boundaries and padding never supply a predecessor. Only
supervised positions are projected, in chunks bounded by
``dflash.target_argmax_chunk_size`` (default 128), so full-sequence vocabulary
logits are not retained. Draft features still use pre-final-normalization layer
outputs. Validation uses the same configured label source.

For projection-only adaptation of a pretrained DFlash2 draft, set
``dflash.projection_only=true`` with ``dflash.init_from_pretrained``. Only the
input feature projection ``fc.weight`` is trainable; the target, draft decoder,
norms, convolutions, and candidate selector remain frozen. Gradients still flow
through the frozen draft into the projection, including with activation
checkpointing. The optimizer contains only trainable parameters, while draft
checkpoints and Hugging Face exports retain all draft weights.

For example, to adapt a compatible block8 DFlash2 export to block16:

.. code-block:: yaml

   dflash:
     enabled: true
     variant: dflash2
     init_from_pretrained: /path/to/block8-draft
     block_size: 16
     label_source: target_argmax
     target_argmax_chunk_size: 128
     projection_only: true
     use_fused_linear_ce: false

Use this as an override to the existing SALM configuration, preserving the
warm-start architecture and target-layer taps. The configured block size is
used for training and saved in the exported draft config; strict loading still
requires every weight shape to match. Set ``projection_only=false`` to train
the full draft. A weights-only warm start can change block size; resuming a
training checkpoint instead requires the same optimizer parameter selection.

For more detailed information on training at scale, model parallelism, and SLURM-based training, see :doc:`training and scaling <training_and_scaling>`.

Collection Structure
--------------------

The speechlm2 collection is organized into the following key components:

- **Models**: Contains implementations of DuplexS2SModel, DuplexS2SSpeechDecoderModel, DuplexSTTModel, SALM, SALMAutomodel, DuplexEARTTS, and the inference-only NemotronVoiceChat.
- **Modules**: Contains audio perception and speech generation modules.
- **Data**: Includes dataset classes and data loading utilities.

SpeechLM2 Documentation
-----------------------

For more information, see additional sections in the SpeechLM2 docs:

.. toctree::
   :maxdepth: 1

   models
   datasets
   configs
   training_and_scaling
   vllm_dflash
