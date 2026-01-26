.. _magpie-tts-po:

=======================================
Magpie-TTS Preference Optimization
=======================================

Preference optimization is a powerful technique for improving the quality of Magpie-TTS outputs by learning from ranked examples. Rather than relying solely on supervised learning with ground-truth audio, preference optimization teaches the model to distinguish between good and bad generations, allowing it to internalize quality metrics like intelligibility and speaker similarity directly into its generation process.

Magpie-TTS supports two complementary approaches to preference optimization: offline alignment using Direct Preference Optimization (DPO) and online optimization using Group Relative Policy Optimization (GRPO). While DPO requires pre-generating preference data before training, GRPO generates candidates on the fly and is generally recommended for its simplicity and effectiveness.


Offline Preference Alignment (DPO)
##################################

Direct Preference Optimization works by fine-tuning the model on pairs of chosen and rejected outputs. The training objective encourages the model to increase the likelihood of generating outputs similar to the chosen examples while decreasing the likelihood of rejected ones. This approach is particularly useful when you have access to human quality judgments or want fine-grained control over the preference data.

The DPO pipeline consists of four distinct steps, each building on the output of the previous one.


Step 1: Create Text-Context Pairs
---------------------------------

The first step is to assemble a collection of text-context pairs that will be used for preference data generation. A well-designed dataset should include a mix of challenging texts (such as tongue twisters, technical terms, and complex sentence structures) alongside regular transcripts. These texts are paired with various speaker contexts—either audio samples for voice cloning or text descriptions for style conditioning.

The diversity of this dataset is crucial for robust preference optimization. Including challenging examples helps the model learn to handle edge cases, while regular transcripts ensure it maintains quality on typical inputs. You can also include examples with text contexts to improve style conditioning capabilities.

.. code-block:: bash

    python scripts/magpietts/dpo/create_text_contextpairs.py \
        --challenging_texts /path/to/challenging_texts.txt \
        --regular_texts_for_audiocontext /path/to/regular_texts_for_audiocontext.txt \
        --regular_texts_for_textcontext /path/to/regular_texts_for_textcontext.txt \
        --audio_contexts /path/to/audio_context_list.json \
        --text_contexts /path/to/text_context_list.txt \
        --output_manifest /path/to/text_context_pairs.json \
        --nsamples_perpair 6

The ``nsamples_perpair`` parameter specifies how many audio samples will be generated for each text-context pair in the next step. Setting this to 6 provides enough variety to create meaningful preference pairs while keeping computation manageable. The output manifest serves as input for the generation step.


Step 2: Generate Audio Samples
------------------------------

With the text-context pairs prepared, the next step is to generate multiple audio samples for each pair using a base Magpie-TTS checkpoint. The generation process also computes quality metrics—Character Error Rate (CER) and Speaker Similarity (SSIM)—for each output, which will be used to create preference pairs.

This step can be parallelized across multiple GPUs and nodes to speed up generation. Each generated audio file is accompanied by a JSON file containing the computed metrics.

.. code-block:: bash

    python examples/tts/magpietts.py \
        --config-name=magpietts_po_inference \
        mode=test \
        batch_size=64 \
        +init_from_ptl_ckpt=/path/to/magpie_checkpoint \
        exp_manager.exp_dir=/path/to/po_experiment \
        +test_ds_meta.textcontextpairs.manifest_path=/path/to/text_context_pairs.json \
        +test_ds_meta.textcontextpairs.audio_dir="/" \
        +test_ds_meta.textcontextpairs.feature_dir="/" \
        model.codecmodel_path=/path/to/codec_model.nemo \
        model.prior_scaling_factor=null \
        model.load_cached_codes_if_available=false

.. note::

    The manifest contains absolute audio paths, so ``audio_dir`` is set to ``"/"``. Adjust the model configuration parameters to match your base checkpoint architecture.


Step 3: Create Preference Pairs
-------------------------------

Once audio samples are generated, you need to create chosen-rejected pairs based on the computed metrics. The script analyzes the CER and SSIM scores for each group of samples and selects the best and worst performers to form preference pairs.

.. code-block:: bash

    python scripts/magpietts/dpo/create_preference_pairs.py \
        --input_manifest /path/to/text_context_pairs.json \
        --generated_audio_dir /path/to/po_experiment/MagpieTTS-PO-Infer/version_0/audio \
        --group_size 6 \
        --cer_threshold 0.01 \
        --val_size 256

The ``cer_threshold`` parameter filters out pairs where even the chosen example has poor intelligibility (CER > 0.01). This ensures the model learns from genuinely good examples rather than just "less bad" ones. The script outputs train and validation manifests in the ``manifests/`` subdirectory.


Step 4: DPO Fine-tuning
-----------------------

The final step is fine-tuning the base model on the preference pairs using the DPO loss. This teaches the model to prefer generating outputs similar to the chosen examples over the rejected ones.

.. code-block:: bash

    python examples/tts/magpietts.py \
        batch_size=4 \
        +init_from_ptl_ckpt=/path/to/magpie_checkpoint \
        +mode="dpo_train" \
        max_epochs=10 \
        exp_manager.exp_dir=/path/to/dpo_experiment \
        exp_manager.checkpoint_callback_params.always_save_nemo=false \
        model.train_ds.dataset._target_="nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDatasetDPO" \
        model.validation_ds.dataset._target_="nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDatasetDPO" \
        +train_ds_meta.dpopreftrain.manifest_path="/path/to/manifests/" \
        +train_ds_meta.dpopreftrain.audio_dir="/" \
        +train_ds_meta.dpopreftrain.feature_dir="/" \
        +val_ds_meta.dpoprefval.manifest_path="/path/to/manifests/dpo_val_manifest.json" \
        +val_ds_meta.dpoprefval.audio_dir="/" \
        +val_ds_meta.dpoprefval.feature_dir="/" \
        +model.dpo_beta=0.01 \
        +model.dpo_sft_loss_weight=0.0 \
        model.codecmodel_path=/path/to/codec_model.nemo \
        model.alignment_loss_scale=0.001 \
        model.prior_scaling_factor=null \
        trainer.val_check_interval=200 \
        trainer.log_every_n_steps=10 \
        model.optim.lr=2e-7 \
        ~model.optim.sched

Key parameters for DPO training include ``dpo_beta``, which controls the strength of the preference signal, and a low learning rate (2e-7) to ensure stable fine-tuning.


Online Preference Optimization (GRPO)
#####################################

Group Relative Policy Optimization offers a more streamlined approach that eliminates the need for pre-generating preference data. Instead, GRPO generates multiple candidate outputs for each training example on the fly, computes reward signals based on quality metrics, and optimizes the model to maximize these rewards through policy gradient methods.

GRPO is generally recommended over DPO for several reasons. It continuously adapts to the model's current capabilities rather than relying on static preference data. It requires less setup and storage since there's no need to pre-generate and store audio samples. Additionally, it can optimize for multiple reward signals simultaneously, including CER, SSIM, and PESQ.


Setting Up GRPO Training
------------------------

The GRPO training process starts with preparing text-context pairs, similar to DPO but without the need for multiple samples per pair:

.. code-block:: bash

    python scripts/magpietts/dpo/create_text_contextpairs.py \
        --challenging_texts /path/to/challenging_texts.txt \
        --regular_texts_for_audiocontext /path/to/regular_texts_for_audiocontext.txt \
        --regular_texts_for_textcontext /path/to/regular_texts_for_textcontext.txt \
        --audio_contexts /path/to/audio_context_list.json \
        --text_contexts /path/to/text_context_list.txt \
        --output_manifest /path/to/text_context_pairs_grpo.json \
        --nsamples_perpair 1

Note that ``nsamples_perpair`` is set to 1 since GRPO generates candidates during training.


GRPO Training Configuration
---------------------------

GRPO training requires careful configuration of several hyperparameters. The following table summarizes the key parameters:

.. list-table:: GRPO Hyperparameters
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``num_generations_per_item``
     - 12
     - Number of candidate outputs generated per training example
   * - ``reference_free``
     - true
     - If true, skips KL divergence term and optimizes rewards directly
   * - ``grpo_beta``
     - 0.0
     - Coefficient for KL loss (only used when reference_free=false)
   * - ``cer_reward_weight``
     - 0.33
     - Weight of Character Error Rate in the reward function
   * - ``ssim_reward_weight``
     - 0.33
     - Weight of Speaker Similarity in the reward function
   * - ``pesq_reward_weight``
     - 0.33
     - Weight of PESQ score in the reward function
   * - ``use_pesq``
     - true
     - Whether to include PESQ in the reward computation
   * - ``reward_asr_model``
     - (none)
     - ASR model for CER computation; set to ``whisper`` for multilingual
   * - ``inference_temperature``
     - 0.8
     - Sampling temperature for candidate generation
   * - ``inference_topk``
     - 2016
     - Top-k sampling parameter (2016 effectively disables it)
   * - ``loss_type``
     - "grpo"
     - Loss function variant; can be "grpo" or "dr_grpo"
   * - ``scale_rewards``
     - true
     - Whether to normalize advantages by standard deviation


GRPO Training Command
---------------------

The following command demonstrates a complete GRPO training setup for multilingual models:

.. code-block:: bash

    python examples/tts/magpietts.py \
        --config-name=magpietts \
        batch_size=2 \
        +init_from_ptl_ckpt=/path/to/magpie_checkpoint \
        model.codecmodel_path=/path/to/codec_model.nemo \
        +mode="onlinepo_train" \
        max_epochs=20 \
        exp_manager.exp_dir=/path/to/grpo_experiment \
        +exp_manager.version=0 \
        exp_manager.checkpoint_callback_params.always_save_nemo=false \
        +train_ds_meta.dpopreftrain.manifest_path=/path/to/train_manifest.json \
        +train_ds_meta.dpopreftrain.audio_dir="/" \
        +train_ds_meta.dpopreftrain.feature_dir="/" \
        +val_ds_meta.dpoprefval.manifest_path=/path/to/val_manifest.json \
        +val_ds_meta.dpoprefval.audio_dir="/" \
        +val_ds_meta.dpoprefval.feature_dir="/" \
        +model.grpo_beta=0.0 \
        +model.num_generations_per_item=12 \
        +model.reference_free=true \
        +model.inference_cfg_prob=0.5 \
        +model.inference_cfg_scale=2.5 \
        +model.cer_reward_weight=0.45 \
        +model.ssim_reward_weight=0.45 \
        +model.pesq_reward_weight=0.1 \
        +model.use_pesq=true \
        +model.reward_asr_model="whisper" \
        model.cfg_unconditional_prob=0.0 \
        +model.inference_topk=2016 \
        +model.inference_temperature=0.7 \
        +model.use_kv_cache_during_online_po=true \
        +model.loss_type="grpo" \
        +model.max_decoder_steps=430 \
        model.decoder.p_dropout=0.0 \
        model.encoder.p_dropout=0.0 \
        model.alignment_loss_scale=0.0 \
        model.prior_scaling_factor=null \
        ~trainer.check_val_every_n_epoch \
        +trainer.val_check_interval=50 \
        trainer.log_every_n_steps=10 \
        model.optim.lr=1e-7 \
        ~model.optim.sched \
        exp_manager.checkpoint_callback_params.monitor="val_cer_gt" \
        exp_manager.checkpoint_callback_params.mode="min" \
        trainer.precision=32 \
        +trainer.gradient_clip_val=2.5


Important GRPO Training Considerations
--------------------------------------

Several configuration choices are critical for stable GRPO training:

**Disable Dropout**: Set ``p_dropout=0.0`` for all modules (encoder, decoder). This is essential when not using reference-free mode, as dropout causes the KL divergence loss to become unstable.

**Disable Attention Priors and CTC Loss**: Set ``alignment_loss_scale=0.0`` and ``prior_scaling_factor=null``. These training signals can interfere with the preference optimization objective.

**Use Small Batch Size**: Since GRPO generates ``num_generations_per_item`` samples for each batch item, the effective batch size becomes ``batch_size * num_generations_per_item``. A batch size of 2 with 12 generations per item results in 24 forward passes per step.

**Frequent Validation**: GRPO steps take longer than standard training steps due to the generation overhead. Configure more frequent validation with ``val_check_interval=50`` to monitor progress.

**Low Learning Rate**: Use a learning rate around 1e-7 to ensure stable optimization. The preference signal is noisy, and aggressive updates can destabilize training.

**Model-Specific Overrides**: Ensure your GRPO configuration matches the base model architecture, including attention heads, number of layers, local transformer settings, and tokenizer configuration.


Advanced: Local Transformer Optimization
----------------------------------------

For models with a Local Transformer, GRPO can optimize both with and without the LT by setting ``use_local_transformer_prob`` between 0 and 1. This trains the model to produce high-quality outputs regardless of whether the Local Transformer is used during inference, providing flexibility in the speed-quality trade-off at deployment time.

.. code-block:: bash

    +model.use_local_transformer_prob=0.5  # 50% of generations use LT


See Also
########

- :doc:`magpietts`: Main Magpie-TTS documentation
- `Preference Optimization Source Code <https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/tts/models/magpietts_preference_optimization.py>`__

