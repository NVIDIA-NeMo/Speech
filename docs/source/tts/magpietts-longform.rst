.. _magpie-tts-longform:

==============================
Magpie-TTS Longform Inference
==============================

This document describes how longform (multi-sentence) text-to-speech inference works in Magpie-TTS.


Overview
########

Magpie-TTS supports generating speech for long text inputs by processing them in smaller, sentence-level chunks while maintaining prosodic continuity across the entire utterance. This approach overcomes the context window limitations of the underlying transformer architecture.


When Longform is Used
#####################

Longform inference is automatically triggered based on word count thresholds (approximately 20 seconds of audio):

.. list-table:: Language Word Thresholds
   :header-rows: 1
   :widths: 30 30

   * - Language
     - Word Threshold
   * - English
     - 45 words
   * - Spanish
     - 73 words
   * - French
     - 69 words
   * - German
     - 50 words
   * - Italian
     - 53 words
   * - Vietnamese
     - 50 words

.. note::

   Longform is best supported for English. Mandarin currently falls back to standard inference.


Algorithm
#########

The longform inference algorithm processes long text through the following steps:


Step 1: Sentence Splitting
--------------------------

The input text is split into individual sentences using punctuation markers (``.``, ``?``, ``!``, ``...``). The splitting is intelligent and handles abbreviations like "Dr.", "Mr.", "a.m." by checking if the period is followed by a space.

**Example:**

::

    Input:  "Dr. Smith arrived early. How are you today?"
    Output: ["Dr. Smith arrived early.", "How are you today?"]


Step 2: State Initialization
----------------------------

A ``LongformChunkState`` object is created to track information across sentence chunks:

- **History text tokens**: Text from previous chunks for context
- **History encoder context**: Encoder outputs that provide continuity
- **Attention tracking**: Monitors which positions have been attended to


Step 3: Iterative Chunk Processing
----------------------------------

For each sentence chunk, the following sub-steps are performed:

1. **Context Preparation**: Prepend history text and encoder context from previous chunks to maintain prosodic continuity.

2. **Attention Prior Application**: Apply a learned attention prior that guides the model to attend to the correct text positions, preventing repetition or skipping.

3. **Autoregressive Generation**: Generate audio codes token-by-token using the transformer decoder with temperature sampling.

4. **State Update**: Update the chunk state with:

   - New history text (last N tokens)
   - New encoder context
   - Updated attention tracking

5. **Code Collection**: Store the generated audio codes for this chunk.


Step 4: Code Concatenation
--------------------------

After all chunks are processed, concatenate the audio codes from each chunk along the time dimension into a single sequence.


Step 5: Audio Decoding
----------------------

Pass the concatenated codes through the neural audio codec decoder to produce the final waveform.


Key Components
--------------

1. **Sentence Splitting** (``split_by_sentence``): Intelligently splits text on sentence boundaries while handling abbreviations (e.g., "Dr.", "Mr.").

2. **Chunk State** (``LongformChunkState``): Maintains context across chunks:

   - ``history_text``: Text tokens from previous chunks
   - ``history_context_tensor``: Encoder outputs for continuity
   - ``last_attended_timesteps``: Attention tracking for smooth transitions

3. **Attention Prior**: Guides the model's attention to maintain proper alignment and prevent repetition/skipping.


Usage
#####


Method 1: Using ``do_tts`` (Recommended for Simple Use Cases)
-------------------------------------------------------------

The ``do_tts`` method automatically detects whether longform inference is needed:

.. code-block:: python

    import torch
    from nemo.collections.tts.models import MagpieTTSModel

    # Load model
    model = MagpieTTSModel.restore_from("path/to/magpietts.nemo")
    model.eval()
    model.cuda()

    # Short text - uses standard inference automatically
    short_audio, short_len = model.do_tts(
        transcript="Hello, how are you?",
        language="en",
    )

    # Long text - automatically switches to longform inference
    long_text = """
    The quick brown fox jumps over the lazy dog. This sentence contains every 
    letter of the alphabet. Sphinx of black quartz, judge my vow. Pack my box 
    with five dozen liquor jugs. How vexingly quick daft zebras jump. The five 
    boxing wizards jump quickly. Jackdaws love my big sphinx of quartz.
    """

    long_audio, long_len = model.do_tts(
        transcript=long_text,
        language="en",
        apply_TN=True,  # Apply text normalization
        temperature=0.7,
        topk=80,
        use_cfg=True,
        cfg_scale=2.5,
    )

    # Save audio
    import soundfile as sf
    sf.write("output.wav", long_audio[0].cpu().numpy(), 22050)


Method 2: Using CLI (``magpietts_inference.py``)
------------------------------------------------

For batch inference from manifests:

.. code-block:: bash

    # Auto-detect longform based on text length (default)
    python examples/tts/magpietts_inference.py \
        --nemo_files /path/to/magpietts.nemo \
        --datasets_json_path /path/to/evalset_config.json \
        --out_dir /path/to/output \
        --codecmodel_path /path/to/codec.nemo \
        --longform_mode auto

    # Force longform inference for all inputs
    python examples/tts/magpietts_inference.py \
        --nemo_files /path/to/magpietts.nemo \
        --datasets_json_path /path/to/evalset_config.json \
        --out_dir /path/to/output \
        --codecmodel_path /path/to/codec.nemo \
        --longform_mode always \
        --longform_max_decoder_steps 50000

**Longform CLI Options:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``--longform_mode``
     - ``auto``
     - ``auto``: detect from text, ``always``: force longform, ``never``: disable


Configuration Dataclasses
#########################


``LongformConfig``
------------------

Immutable tuning parameters (set in model):

.. literalinclude:: ../../../nemo/collections/tts/models/magpietts.py
   :language: python
   :pyobject: LongformConfig


``LongformChunkState``
----------------------

Mutable state passed between chunk iterations:

.. literalinclude:: ../../../nemo/collections/tts/models/magpietts.py
   :language: python
   :pyobject: LongformChunkState


Best Practices
##############

1. **Use ``apply_TN=True``** for raw text to ensure proper normalization before synthesis.

2. **Increase ``max_decoder_steps``** for very long texts (default 50000 is usually sufficient).

3. **Use ``longform_mode="auto"``** (default) to let the system decide based on text length.

4. **For non-English languages**, be aware that longform performance may vary. English is best supported.


Limitations
###########

- **Mandarin (zh)**: Currently falls back to standard inference due to character-based tokenization complexities.
- **Prosodic boundaries**: While the algorithm maintains continuity, natural paragraph breaks may not always be perfectly preserved in non-English languages.


See Also
########

- :doc:`magpietts`: Main Magpie-TTS documentation
- :doc:`magpietts-po`: Preference Optimization Guide

