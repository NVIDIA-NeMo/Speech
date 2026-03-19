.. _ten-minutes:

NeMo Speech Inference in 5 Minutes
===================================

This guide gives you a quick, hands-on tour of NeMo's core speech capabilities. By the end, you'll have transcribed audio, synthesized speech, identified speakers, and used a speech language model — all in about 50 lines of code.

.. note::

   Make sure you have :doc:`installed NeMo <install>` before starting.


1. Transcribe Speech (ASR)
--------------------------

Automatic Speech Recognition converts audio to text. NeMo's Parakeet model sits at the top of the `HuggingFace OpenASR Leaderboard <https://huggingface.co/spaces/hf-audio/open_asr_leaderboard>`_.

**Basic transcription** — 3 lines of code:

.. code-block:: python

   import nemo.collections.asr as nemo_asr

   asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
   transcript = asr_model.transcribe(["audio.wav"])[0].text
   print(transcript)

**With timestamps** — know *when* each word was spoken:

.. code-block:: python

   hypotheses = asr_model.transcribe(["audio.wav"], timestamps=True)
   for stamp in hypotheses[0].timestamp['word']:
       print(f"{stamp['start']}s - {stamp['end']}s : {stamp['word']}")

**From the command line**:

.. code-block:: bash

   python examples/asr/transcribe_speech.py \
       pretrained_name="nvidia/parakeet-tdt-0.6b-v2" \
       audio_dir=./my_audio_files/


2. Synthesize Speech (TTS)
--------------------------

Text-to-Speech generates natural audio from text. NeMo's **Magpie TTS** is a multilingual, codec-based model that supports multiple speakers and languages:

.. code-block:: python

   from nemo.collections.tts.models import MagpieTTSModel
   import soundfile as sf

   # Load model (multilingual 357M, from Hugging Face)
   model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
   model.eval()

   # Generate speech
   audio, audio_len = model.do_tts(
       transcript="Hello! Welcome to NeMo speech AI.",
       language="en",
   )

   # Save to file
   sf.write("output.wav", audio[0].cpu().numpy(), 22050)
   print("Speech saved to output.wav")


3. Identify Speakers (Diarization)
----------------------------------

Speaker diarization answers "who spoke when?" in multi-speaker audio.

.. code-block:: python

   from nemo.collections.asr.models import SortformerEncLabelModel

   diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2")
   diar_model.eval()

   segments = diar_model.diarize(audio=["meeting.wav"], batch_size=1)
   for seg in segments[0]:
       print(seg)  # (begin_seconds, end_seconds, speaker_index)


4. Speech Language Models (SpeechLM2)
-------------------------------------

SpeechLM2 augments large language models with speech understanding. Canary-Qwen combines an ASR encoder with a Qwen LLM:

.. code-block:: python

   from nemo.collections.speechlm2.models import SALM

   model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')

   answer_ids = model.generate(
       prompts=[[{
           "role": "user",
           "content": f"Transcribe the following: {model.audio_locator_tag}",
           "audio": ["speech.wav"],
       }]],
       max_new_tokens=128,
   )
   print(model.tokenizer.ids_to_text(answer_ids[0].cpu()))


What's Next?
------------

Now that you've seen the basics, dive deeper:

- :doc:`key_concepts` — Understand the speech AI fundamentals behind these models
- :doc:`choosing_a_model` — Find the best model for your specific use case
- :doc:`../asr/intro` — Full ASR documentation
- :doc:`../tts/intro` — Full TTS documentation
- :doc:`../asr/speaker_diarization/intro` — Speaker diarization and recognition
- :doc:`../starthere/tutorials` — Tutorial notebooks

