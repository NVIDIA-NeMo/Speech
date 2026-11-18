.. _nemotron-transcribe:

Nemotron 3.5 Transcribe
=======================

`NVIDIA Nemotron 3.5 Transcribe <https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B>`_
is an open-weight, offline speech-augmented language model (SALM). It supports transcription in 40 languages,
recordings up to one hour, and a 65,536-token context. Text instructions control transcription style,
vocabulary context, speech translation, speaker attribution, and audio question answering.

The model combines a hybrid Mamba-Transformer mixture-of-experts language backbone with speech and speaker
encoders. Including the audio encoder, it has 34B total parameters and approximately 4.3B active parameters.

* `BF16 reference checkpoint <https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B>`_
* `FP8 checkpoint <https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8>`_
* `NVFP4 checkpoint <https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4>`_
* `Interactive demo <https://huggingface.co/spaces/nvidia/nemotron-transcribe>`_

The checkpoints use the `OpenMDW-1.1 license <https://openmdw.ai/license/1-1/>`_. See the
`model card <https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B>`_ for benchmark results,
hardware compatibility, and the language support matrix. Translation and multi-speaker coverage vary by language;
audio understanding and reasoning were evaluated only in English.

.. note::

   Transcribe processes complete recordings. For real-time audio input, choose
   `Nemotron-3.5-ASR-Streaming <https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b>`_.
   Streaming generated text from a Transcribe request does not make it streaming ASR.

Install and download
--------------------

Use the latest NeMo Speech ``main`` branch and its vLLM 0.32.0 serving environment on Linux.
The BF16 example below targets one H100 80 GB. Start in a separate checkout and environment from training;
the :doc:`general serving guide <vllm_serving>` explains which extras can be combined with ``vllm``.

.. code-block:: bash

   git clone https://github.com/NVIDIA-NeMo/Speech.git nemo-speech-transcribe
   cd nemo-speech-transcribe
   uv venv --python 3.13 .venv
   uv sync --locked --extra asr --extra vllm --no-install-project
   uv pip install --python .venv/bin/python --no-deps -e .
   # Audio decoding and OpenCV dependencies for the serving environment.
   uv pip install --python .venv/bin/python --no-deps av==19.0.2 opencv-python-headless==5.0.0.93
   source .venv/bin/activate
   hf download nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
     --local-dir ./NVIDIA-Nemotron-3.5-Transcribe-30B-A4B

Serve the BF16 checkpoint
-------------------------

In the activated environment, start the server with timestamp capture available for requests that opt in.
This follows the model card's MTP configuration; adjust memory utilization and concurrency for your hardware.
Keep the V1 runner for this starting configuration.

.. code-block:: bash

   export VLLM_PLUGINS=nemo_speechlm
   export VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
   export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180
   export VLLM_USE_V2_MODEL_RUNNER=0
   vllm serve ./NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
     --host 127.0.0.1 \
     --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
     --trust-remote-code --dtype bfloat16 --gpu-memory-utilization 0.94 \
     --max-model-len 65536 --max-num-batched-tokens 4096 --max-num-seqs 192 \
     --enable-chunked-prefill --no-enable-prefix-caching \
     --use-replayssm --mamba-backend flashinfer --mamba-cache-mode none \
     --mm-processor-cache-gb 0 --no-async-scheduling \
     --limit-mm-per-prompt '{"audio":{"count":6,"length":120000000}}' \
     --speculative-config '{"method":"mtp","num_speculative_tokens":8}' \
     --reasoning-parser nemotron_v3 \
     --trust-request-mm-kwargs \
     --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware

The server accepts local requests on port 8000. ``--trust-request-mm-kwargs`` permits per-request timestamp
capture; use it with trusted clients. Kernel compilation and warmup make the first requests slower.

Transcribe a recording
----------------------

In a second terminal, activate the same environment and run the following Python code with a local
16 kHz mono WAV file named ``audio.wav``. The OpenAI client is included in the serving dependencies.
Use chat completions for this SpeechLM workflow.

.. code-block:: python

   import base64
   from pathlib import Path

   from openai import OpenAI

   client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

   VERBATIM_PROMPT = (
       "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. "
       "Formatting: punctuation and capitalization. Output: transcript only."
   )

   def transcribe(audio_path, instruction=VERBATIM_PROMPT, timestamps=False):
       audio_data = base64.b64encode(Path(audio_path).read_bytes()).decode("ascii")
       options = {
           "chat_template_kwargs": {"enable_thinking": False},
           "skip_special_tokens": False,
       }
       if timestamps:
           options["mm_processor_kwargs"] = {"capture_ctc_timestamps": True}
       return client.chat.completions.create(
           model="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",
           messages=[
               {
                   "role": "user",
                   "content": [
                       {"type": "text", "text": instruction},
                       {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_data}"}},
                   ],
               }
           ],
           temperature=0,
           max_tokens=16384,
           extra_body=options,
       )

   response = transcribe("audio.wav")
   print(response.choices[0].message.content)
   print("Finish reason:", response.choices[0].finish_reason)

The request disables reasoning for transcription. If ``finish_reason`` is ``"length"``, increase
``max_tokens`` within the remaining context budget. Long recordings need room for both audio input and the
generated transcript; the one-hour audio limit alone does not guarantee that every transcript fits.

Choose the task with instructions
---------------------------------

Pass an instruction as the second argument to ``transcribe``. The templates below follow the model card's
`recommended prompts <https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B#prompt-recommendations>`_.
Keep the system prompt empty and send the instruction alongside the audio in the user message.
Replace angle-bracketed placeholders with the target language, vocabulary context, or accepted labels.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Task
     - Recommended instruction
   * - Verbatim transcript
     - ``Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only.``
   * - Readable transcript
     - ``Audio transcription. Language: as spoken. Wording: omit fillers and accidental repetitions; keep the remaining spoken wording unchanged. Formatting: punctuation and capitalization. Output: transcript only.``
   * - Vocabulary context
     - ``Transcribe with punctuation and capitalization. Context terms: <names and terms>.``
   * - Translation
     - ``Please translate the given speech to <target language>.``
   * - Language identification
     - ``Identify the spoken language. Respond with exactly one label from <labels> and nothing else.``

For audio question answering, use the model card's
`reasoning request settings <https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B#send-a-request>`_
and task prompts. The transcription helper above deliberately disables reasoning.

Word timestamps and speaker attribution
---------------------------------------

Reuse the helper above with timestamp capture enabled and the model card's recommended speaker-attribution prompt:

.. code-block:: python

   response = transcribe(
       "audio.wav",
       "Transcribe this audio with punctuation and capitalization. "
       "Include all spoken words and verbal fillers. "
       "Write numbers in their standard written form. "
       "Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription.",
       timestamps=True,
   )
   print(response.choices[0].message.content)
   alignment = response.model_dump()["ctc_timestamps"]
   if alignment["error"] is not None:
       raise RuntimeError(alignment["error"])
   for word in alignment["words"]:
       print(word["word"], word["start"], word["end"], word["speaker"])

The timestamp middleware returns word boundaries in seconds, diarization segments, and a mapping between
transcript speaker tags and diarization speakers. Keep ``skip_special_tokens=False`` to preserve speaker tags.
Use one recording and one completion per timestamp request, and check the alignment's ``error`` field.
See :doc:`models` for the timestamp response contract, including generated-text streaming.

Further workflows
------------------

* :doc:`vllm_serving` covers runtime configuration and throughput tuning.
* :doc:`speculative_decoding` covers MTP and separate draft models. Check the model card for model-specific
  draft requirements before replacing the MTP configuration above.
* :doc:`quantization` explains the FP8 and NVFP4 export recipes. For published quantized checkpoints, follow
  their model cards for hardware and serving settings.
* The `SALM fine-tuning tutorials <https://github.com/NVIDIA-NeMo/Speech/tree/main/tutorials/speechlm2/finetuning>`_
  and `fine-tuning toolkit <https://github.com/NVIDIA-NeMo/Speech/tree/main/examples/speechlm2/finetuning>`_
  cover adapting exported SALMAutomodel checkpoints. Use a separate training environment.

The model card distinguishes optimized NIM performance from the open-source serving implementation.
Measure performance with your own workload and serving configuration.
