.. _magpie-tts:

==========
Magpie-TTS
==========

Large language models have revolutionized text-to-speech synthesis, enabling the generation of remarkably natural and expressive speech. However, these models often suffer from a critical limitation: hallucinations. Users frequently encounter issues such as repeated words, missing phrases, or speech that drifts out of alignment with the input text. These artifacts significantly degrade the user experience and limit the deployment of LLM-based TTS systems in production environments.

Magpie-TTS [1]_ addresses these challenges head-on by introducing monotonic alignment techniques that ensure robust, hallucination-free speech synthesis. Developed by NVIDIA, the model combines the expressiveness of modern neural TTS with the reliability required for real-world applications. At its core, Magpie-TTS uses CTC (Connectionist Temporal Classification) loss and attention priors to enforce monotonic cross-attention between text and audio, effectively preventing the model from skipping, repeating, or misaligning content during generation.

The model follows an encoder-decoder transformer architecture that operates on discrete audio tokens produced by a neural audio codec. This design choice enables high-quality waveform reconstruction while maintaining the flexibility of sequence-to-sequence modeling. Magpie-TTS supports voice cloning through audio context conditioning, allowing users to synthesize speech in any voice given just a few seconds of reference audio.


Model Architecture
##################

Magpie-TTS processes text through a transformer encoder that captures the linguistic structure and phonetic content of the input. The encoder uses a stack of self-attention layers with causal masking, producing contextual representations that guide the subsequent audio generation. For phoneme-based models, the text is first converted using an IPA tokenizer with grapheme-to-phoneme conversion, while character-level models can operate directly on raw text using byte-level tokenization.

The decoder autoregressively generates discrete audio tokens by attending to both the encoded text and any provided audio context. The cross-attention mechanism between the decoder and encoder is where the attention prior comes into play, biasing the model toward monotonic alignment patterns that match the natural left-to-right progression of speech. This architectural choice is what distinguishes Magpie-TTS from conventional LLM-based approaches and eliminates most hallucination artifacts.

For voice cloning, Magpie-TTS employs a dedicated context encoder that processes the reference audio. The context encoding is injected into the decoder, allowing the model to capture speaker characteristics such as pitch, timbre, and speaking style. The model also supports text-based context conditioning, enabling control over speaking style through textual descriptions rather than audio examples.


Frame Stacking and Local Transformer
------------------------------------

A key innovation in Magpie-TTS is the two-stage decoding architecture [4]_ that dramatically accelerates inference. The base decoder can be configured to process multiple consecutive audio frames in a single forward pass through a technique called frame stacking. Rather than generating one frame at a time, the base decoder produces a single latent representation for each group (or "stack") of frames, and a lightweight Local Transformer then expands this into individual frame-level predictions.

This two-stage approach delivers significant speed improvements for two reasons. First, the Local Transformer uses far fewer parameters than the base decoder, making each of its forward passes much faster. Second, the Local Transformer only attends to the current frame stack and its corresponding latent, rather than the entire audio sequence, resulting in much shorter attention contexts. The ``frame_stacking_factor`` parameter controls how many frames are grouped together, with values up to 4 tested successfully. Setting this to 1 (the default) disables frame stacking entirely.

The Local Transformer handles the prediction of tokens across multiple codebooks within each time frame. The audio codec uses multiple codebooks to represent audio at different levels of detail, and the Local Transformer can operate in either autoregressive mode, where codebooks are predicted sequentially, or MaskGit mode, which uses iterative parallel decoding for even faster inference. When frame stacking is enabled, using the Local Transformer is typically necessary to maintain high synthesis quality.


Model Configurations
--------------------

Magpie-TTS supports two model configurations to suit different use cases:

- **decoder_context_tts**: The standard configuration where text goes to the encoder and both context audio and target audio are processed by the decoder
- **decoder_ce**: Adds a learned context embedding network between the context encoder and decoder for more flexible voice conditioning


Key Features
############

The attention prior mechanism is central to Magpie-TTS's robustness. During training, the model learns to produce cross-attention patterns that follow a monotonic diagonal, matching the natural progression of speech from left to right through the text. This is achieved through a combination of CTC-based alignment loss and soft attention priors that guide the attention weights without being overly restrictive. During inference, the attention prior can be applied dynamically, further reducing the risk of hallucinations on out-of-distribution inputs.

Classifier-free guidance (CFG) [2]_ provides another dimension of control over the generation process. By training the model to occasionally drop conditioning information, Magpie-TTS learns to generate both conditioned and unconditioned outputs. At inference time, the difference between these outputs can be amplified to increase the fidelity and consistency of the generated speech. Users can adjust the CFG scale to balance between adherence to the conditioning and diversity in the output.

Magpie-TTS is designed for multilingual synthesis from the ground up. The model uses a flexible tokenization scheme that can handle multiple languages, with support for language-specific phoneme tokenizers as well as universal byte-level tokenization. This enables training on diverse multilingual datasets and synthesizing speech in languages beyond English. The 357M parameter multilingual checkpoint available on Hugging Face demonstrates these capabilities across a range of languages.

Long-form speech generation (beta) is supported by Magpie-TTS for English language only. This feature is currently in beta. The long input text should have punctuation, especially sentence boundaries. See the :doc:`Longform Inference Guide <magpietts-longform>` for detailed usage instructions.

Training
########

Training Magpie-TTS requires preparing your dataset in NeMo's manifest format, where each entry specifies the audio file path, transcript, and optional speaker information. The model expects pre-computed audio codec tokens, which can be generated using the accompanying audio codec model. For voice cloning, you should also include context audio segments that the model will learn to condition on.

The training script uses Hydra for configuration management, making it easy to customize model architecture, training hyperparameters, and dataset settings. The default configuration provides a solid starting point with a 6-layer encoder and 12-layer decoder, though these can be scaled up for higher quality or down for faster iteration.

.. code-block:: bash

    python examples/tts/magpietts.py \
        --config-name=magpietts \
        max_epochs=100 \
        batch_size=16 \
        model.codecmodel_path=/path/to/audio_codec.nemo \
        train_ds_meta.dataset_name.manifest_path=/path/to/train_manifest.json \
        train_ds_meta.dataset_name.audio_dir=/path/to/audio \
        val_ds_meta.dataset_name.manifest_path=/path/to/val_manifest.json \
        val_ds_meta.dataset_name.audio_dir=/path/to/audio \
        exp_manager.exp_dir=/path/to/experiments


Preference Optimization
-----------------------

Beyond standard supervised training, Magpie-TTS supports preference optimization techniques that can further refine output quality by learning from ranked examples. Two approaches are available: offline preference alignment using Direct Preference Optimization (DPO) and online optimization using Group Relative Policy Optimization (GRPO) (recommended).

**Offline Preference Alignment (DPO)** follows a four-step pipeline. First, you create a set of text-context pairs that will be used for preference data generation—typically a mix of challenging texts (tongue twisters, complex sentences) and regular transcripts paired with various speaker contexts. Second, you generate multiple audio samples for each pair using a base checkpoint, computing quality metrics like Character Error Rate (CER) and Speaker Similarity (SSIM) for each generation. Third, you create chosen-rejected pairs by selecting the best and worst outputs based on these metrics. Finally, you fine-tune the model on these preference pairs using the DPO loss.

.. code-block:: bash

    # Step 1: Create text-context pairs
    python scripts/magpietts/dpo/create_text_contextpairs.py \
        --challenging_texts /path/to/challenging_texts.txt \
        --audio_contexts /path/to/audio_context_list.json \
        --output_manifest /path/to/text_context_pairs.json \
        --nsamples_perpair 6

    # Step 4: DPO fine-tuning
    python examples/tts/magpietts.py \
        +init_from_ptl_ckpt=/path/to/base_checkpoint \
        +mode="dpo_train" \
        +model.dpo_beta=0.01 \
        model.optim.lr=2e-7

**Group Relative Policy Optimization (GRPO)** [3]_ offers a simpler alternative that generates preference data on-the-fly during training. Instead of pre-generating audio samples, GRPO produces multiple candidates for each training example and computes rewards based on CER, SSIM, and optionally PESQ (Perceptual Evaluation of Speech Quality). The model then learns to maximize these rewards through policy gradient optimization.

GRPO is particularly effective because it continuously adapts to the model's current capabilities rather than relying on static preference data. Key hyperparameters include ``num_generations_per_item`` (typically 12) which controls how many candidates are generated per example, and the reward weights that balance different quality metrics. For multilingual models, setting ``reward_asr_model="whisper"`` enables proper CER computation across languages.

.. code-block:: bash

    python examples/tts/magpietts.py \
        +init_from_ptl_ckpt=/path/to/base_checkpoint \
        +mode="onlinepo_train" \
        batch_size=2 \
        +model.num_generations_per_item=12 \
        +model.reference_free=true \
        +model.cer_reward_weight=0.33 \
        +model.ssim_reward_weight=0.33 \
        +model.pesq_reward_weight=0.33 \
        +model.use_pesq=true \
        +model.loss_type="grpo" \
        model.decoder.p_dropout=0.0 \
        model.optim.lr=1e-7

.. note::

    When using GRPO, it's important to disable dropout in all modules (``p_dropout=0.0``) to ensure stable KL divergence computation. Also disable attention priors and CTC loss during GRPO training, as these can interfere with the preference optimization process.

For comprehensive documentation on preference optimization, including step-by-step instructions for both DPO and GRPO, see the :doc:`Preference Optimization Guide <magpietts-po>`.

Inference
#########

Running inference with Magpie-TTS involves loading the trained model and providing the text to synthesize along with optional context audio for voice cloning. The inference script supports batched generation for efficient processing of multiple utterances and includes built-in evaluation metrics for assessing output quality.

Several parameters control the generation behavior. The temperature setting affects the randomness of token sampling, with lower values producing more deterministic output and higher values introducing more variation. Top-k sampling restricts the token selection to the k most likely candidates, helping to avoid low-probability artifacts. When using classifier-free guidance, the CFG scale parameter controls how strongly the conditioning influences the output.

.. code-block:: bash

    python examples/tts/magpietts_inference.py \
        --nemo_files /path/to/magpietts_model.nemo \
        --codecmodel_path /path/to/audio_codec.nemo \
        --datasets your_evaluation_set \
        --out_dir /path/to/output \
        --temperature 0.6 \
        --topk 80 \
        --use_cfg \
        --cfg_scale 2.5

For production deployments, enabling the attention prior during inference adds an extra layer of robustness. The prior gently biases the cross-attention toward monotonic patterns, catching any potential alignment drift before it manifests as audible artifacts. This is especially valuable when processing text that differs from the training distribution.

When using models trained with frame stacking, you can enable the Local Transformer during inference with ``--use_local_transformer``. The MaskGit decoding mode can be activated for faster inference at a slight quality trade-off, with ``--maskgit_n_steps`` controlling the number of refinement iterations.

To enable Long-form speech generation (beta) set ``--longform_mode`` to ``auto or always``; this will either automatically detect the long form text or always use the long form inference pipeline. For comprehensive documentation on longform inference, see the :doc:`Longform Inference Guide <magpietts-longform>`.

Resources
#########

To get started with Magpie-TTS, you can download the pretrained multilingual checkpoint from `Hugging Face <https://huggingface.co/nvidia/magpie_tts_multilingual_357m>`__ and try it out in the interactive `demo space <https://huggingface.co/spaces/nvidia/magpie_tts_multilingual_demo>`__. For deeper technical details, refer to [1]_, [2]_, [3]_, and [4]_. The complete source code is available in the `NeMo GitHub repository <https://github.com/NVIDIA/NeMo/tree/main/nemo/collections/tts/models/magpietts.py>`__.

Additional documentation on advanced features can be found in the repository:

- `Frame Stacking Guide <https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/tts/README_frame_stacking.md>`__: Detailed explanation of the two-stage decoding architecture

References
##########

.. [1] `Improving Robustness of LLM-based Speech Synthesis by Learning Monotonic Alignment <https://arxiv.org/pdf/2406.17957>`__

.. [2] `Koel-TTS: Enhancing LLM based Speech Generation with Preference Alignment and Classifier Free Guidance <https://aclanthology.org/2025.emnlp-main.1076.pdf>`__

.. [3] `ALIGN2SPEAK: IMPROVING TTS FOR LOW RESOURCE LANGUAGES VIA ASR-GUIDED ONLINE PREFERENCE OPTIMIZATION <https://arxiv.org/pdf/2509.21718?>`__

.. [4] `FRAME-STACKED LOCAL TRANSFORMERS FOR EFFICIENT MULTI-CODEBOOK SPEECH GENERATION <https://arxiv.org/pdf/2509.19592>`__
