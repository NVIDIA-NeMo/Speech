.. _canary_streaming:

Canary Chunked and Streaming Decoding
*************************************

Canary models support chunked and streaming inference for real-time speech recognition and translation. NeMo provides two approaches:

.. _canary_chunked_inference:

Chunked Inference
=================

This script chunks long audios into non-overlapping segments of ``chunk_len_in_secs`` 
seconds and performs inference on each segment individually. The results are then concatenated to form the final output.

**Key Parameters:**

* ``chunk_len_in_secs`` - Chunk duration (default: 40.0)
* ``timestamps`` - Enable timestamps (default: False)

.. code-block:: bash

    python examples/asr/asr_chunked_inference/aed/speech_to_text_aed_chunked_infer.py \
        model_path=null \
        pretrained_name="nvidia/canary-1b-flash" \
        audio_dir="<(optional) path to folder of audio files>" \
        dataset_manifest="<(optional) path to manifest>" \
        output_filename="<(optional) specify output filename>" \
        chunk_len_in_secs=40.0 \
        batch_size=16 \
        decoding.beam.beam_size=1

To return word and segment level timestamps, add ``timestamps=True`` to the above command.

Note: Canary-1b-v2 supports long-form inference via the ``.transcribe()`` method.
It will use dynamic chunking with overlapping windows for better performance.
This behavior is enabled automatically for long-form inference when transcribing a single 
audio file or when ``batch_size`` is set to 1.


.. _canary_streaming_inference:

Streaming Inference
===================

Real-time decoding with configurable latency using **Wait-k** or **AlignAtt** streaming policies:

**Wait-K** policy predicts only one token per each new speech chunk that increases the overall latency - `original paper <https://arxiv.org/abs/1810.08398>`__.
In this case, it is unclear at what point you can forget part of the left context when recognizing with a limited window.
It is recommended to set the left context to the maximum possible value (infinite left context) for the waitk policy.

**AlignAtt** policy predicts tokens according to the cross-attention condition - `original paper <https://arxiv.org/pdf/2305.11408>`__.
If the condition is met, then the audio size does not need to be increased, and the prediction of the next token continues.
Otherwise, the audio buffer size needs to be increased. This policy shows lower latency in comparison with waitk.
This policy is also suitable for window recognition (left context is fixed and not infinite), but you can lose accuracy in this case.

Usage
-----
Decoding policy is controlled by ``AEDStreamingDecodingConfig``. You can choose ``waitk`` or ``alignatt`` policy.
The remaining parameters (such as ``alignatt_thr`` or ``waitk_lagging``) need to be selected depending on the data and the task (for example, for AST, you can increase ``waitk_lagging``, which works for both policies).

Remember to manage prompt parameters using ``+prompt`` (for example, ``+prompt.pnc=yes/no``, ``+prompt.task=asr/ast``, ``+prompt.source_lang=en``, ``+prompt.target_lang=de``, and so on). This is especially important for AST task.


Key Parameters
--------------

* ``chunk_secs`` - Streaming chunk duration (default: 2.0)
* ``left_context_secs`` - Left context for quality (default: 10.0)
* ``right_context_secs`` - Right context, affects latency (default: 2.0)
* ``decoding.streaming_policy`` - "waitk" or "alignatt"
* ``decoding.alignatt_thr`` - Cross-attention threshold for AlignAtt policy (default: 8), alignatt only
* ``decoding.waitk_lagging`` - Number of chunks to wait in the beginning (default: 2), works for both policies
* ``decoding.exclude_sink_frames`` - Number of frames to exclude from the xatt scores calculation (default: 8), alignatt only
* ``decoding.xatt_scores_layer`` - Layer to get cross-attention (xatt) scores from (default: -2), alignatt only
* ``decoding.hallucinations_detector`` - Detect hallucinations in the predicted tokens (default: True), works for both policies
* ``+prompt.pnc`` - set punctuation and capitalization prompt (yes/no)
* ``+prompt.task`` - set task prompt (asr/ast)
* ``+prompt.source_lang`` - set source language prompt
* ``+prompt.target_lang`` - set target language prompt

.. code-block:: bash

    python3 examples/asr/asr_chunked_inference/aed/speech_to_text_aed_streaming_infer.py \
        pretrained_name=nvidia/canary-1b-v2 \
        dataset_manifest="<path to manifest>" \
        output_filename="<(optional) specify output filename>" \
        left_context_secs=10 \
        chunk_secs=1 \
        right_context_secs=0.5 \
        batch_size=32 \
        decoding.streaming_policy=waitk \ # [waitk or alignatt]
        decoding.alignatt_thr=8 \
        decoding.waitk_lagging=2 \
        decoding.exclude_sink_frames=8 \
        decoding.xatt_scores_layer=-2 \
        decoding.hallucinations_detector=True \
        +prompt.pnc=yes \
        +prompt.task=asr \
        +prompt.source_lang=en \
        +prompt.target_lang=en


The script supports latency calculation based on `Length-Adaptive Average Lagging metric (LAAL) <https://aclanthology.org/2022.autosimtrans-1.2.pdf>`_ for both streaming policies.

Brief comparison of the two streaming policies:
------------------------------------------------

* **Wait-k**: Higher accuracy, requires larger left context, higher latency
* **AlignAtt**: Lower latency, suitable for production, predicts multiple tokens per chunk
