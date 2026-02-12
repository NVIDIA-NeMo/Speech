SpeechLM2
================================

.. note::
   The SpeechLM2 collection is still in active development and the code is likely to keep changing.

SpeechLM2 refers to a collection that augments pre-trained Large Language Models (LLMs) with speech understanding and generation capabilities. 

This collection is designed to be compact, efficient, and to support easy swapping of different LLMs backed by HuggingFace AutoModel. 
It has a first-class support for using dynamic batch sizes via Lhotse and various model parallelism techniques (e.g., FSDP2, Tensor Parallel, Sequence Parallel) via PyTorch DTensor API.

We currently support three main model types:
* SALM (Speech-Augmented Language Model) - a simple but effective approach to augmenting pre-trained LLMs with speech understanding capabilities.
* DuplexS2SModel - a full-duplex speech-to-speech model with an ASR encoder, directly predicting discrete audio codes.
* DuplexS2SSpeechDecoderModel - a variant of DuplexS2SModel with a separate transformer decoder for speech generation.

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
    import torchaudio
    import nemo.collections.speechlm2 as slm

    model = slm.models.SALM.from_pretrained("path/to/pretrained_checkpoint").eval()
    
    # Load audio file
    audio_path = "path/to/audio.wav"
    audio_signal, sample_rate = torchaudio.load(audio_path)
    
    # Resample if needed
    if sample_rate != 16000:  # Most models expect 16kHz audio
        audio_signal = torchaudio.functional.resample(audio_signal, sample_rate, 16000)
        sample_rate = 16000
    
    # Prepare audio for model
    audio_signal = audio_signal.to(model.device)
    audio_len = torch.tensor([audio_signal.shape[1]], device=model.device)
    
    # Create a prompt for SALM model inference
    # The audio_locator_tag is a special token that will be replaced with audio embeddings
    prompt = [{"role": "user", "content": f"{model.audio_locator_tag}"}]
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            prompts=[prompt],
            audios=audio_signal,
            audio_lens=audio_len,
            generation_config=None  # You can customize generation parameters here
        )
    
    # Process the output tokens
    response = model.tokenizer.ids_to_text(output[0])
    print(f"Model response: {response}")

DuplexS2SModel
**************

You can run inference using the loaded pretrained DuplexS2SModel:

.. code-block:: python

    import torch
    import torchaudio
    import nemo.collections.speechlm2 as slm

    model = slm.models.DuplexS2SModel.from_pretrained("path/to/pretrained_checkpoint").eval()
    
    # Load audio file
    audio_path = "path/to/audio.wav"
    audio_signal, sample_rate = torchaudio.load(audio_path)
    
    # Resample if needed
    if sample_rate != 16000:  # Most models expect 16kHz audio
        audio_signal = torchaudio.functional.resample(audio_signal, sample_rate, 16000)
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

Training a Model
----------------

This example demonstrates how to train a SALM model. The remaining models can be trained in a similar manner.

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

    # For inference/evaluation 
    python examples/speechlm2/salm_eval.py \
      pretrained_name=/path/to/checkpoint \
      inputs=/path/to/test_manifest \
      batch_size=64 \
      max_new_tokens=128 \
      output_manifest=generations.jsonl

For more detailed information on training at scale, model parallelism, and SLURM-based training, see :doc:`training and scaling <training_and_scaling>`.

Collection Structure
--------------------

The speechlm2 collection is organized into the following key components:

- **Models**: Contains implementations of DuplexS2SModel, DuplexS2SSpeechDecoderModel, and SALM
- **Modules**: Contains audio perception and speech generation modules
- **Data**: Includes dataset classes and data loading utilities

SpeechLM2 Documentation
-----------------------

For more information, see additional sections in the SpeechLM2 docs:

.. toctree::
   :maxdepth: 1

   models
   datasets
   configs
   training_and_scaling
