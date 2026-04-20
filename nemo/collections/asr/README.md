# Automatic Speech Recognition (ASR)

## Key Features

* [HuggingFace Space for Audio Transcription (File, Microphone and YouTube)](https://huggingface.co/spaces/smajumdar/nemo_multilingual_language_id)
* [Pretrained models](https://ngc.nvidia.com/catalog/collections/nvidia:nemo_asr) available in 14+ languages
* [Automatic Speech Recognition (ASR)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/intro.html)
    * Supported ASR [models](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html):
        * Jasper, QuartzNet, CitriNet, ContextNet
        * Conformer-CTC, Conformer-Transducer, FastConformer-CTC, FastConformer-Transducer
        * Squeezeformer-CTC and Squeezeformer-Transducer
        * LSTM-Transducer (RNNT) and LSTM-CTC
    * Supports the following decoders/losses:
        * CTC
        * Transducer/RNNT
        * Hybrid Transducer/CTC
        * NeMo Original [Multi-blank Transducers](https://arxiv.org/abs/2211.03541) and [Token-and-Duration Transducers (TDT)](https://arxiv.org/abs/2304.06795)
    * Streaming/Buffered ASR (CTC/Transducer) - [Chunked Inference Examples](https://github.com/NVIDIA/NeMo/tree/stable/examples/asr/asr_chunked_inference)
    * [Cache-aware Streaming Conformer](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#cache-aware-streaming-conformer) with multiple lookaheads (including microphone streaming [tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb).
    * Beam Search decoding
    * [Language Modelling for ASR (CTC and RNNT)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_language_modeling.html): N-gram LM in fusion with Beam Search decoding, Neural Rescoring with Transformer
    * [Support of long audios for Conformer with memory efficient local attention](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/results.html#inference-on-long-audio)
* [Speech Classification, Speech Command Recognition and Language Identification](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speech_classification/intro.html): MatchboxNet (Command Recognition), AmberNet (LangID)
* [Voice activity Detection (VAD)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speech_classification/models.html#marblenet-vad): MarbleNet
    * ASR with VAD Inference - [Example](https://github.com/NVIDIA/NeMo/tree/stable/examples/asr/asr_vad)
* [Speaker Recognition](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/intro.html): TitaNet, ECAPA_TDNN, SpeakerNet
* [Speaker Diarization](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_diarization/intro.html)
    * Clustering Diarizer: TitaNet, ECAPA_TDNN, SpeakerNet
    * Neural Diarizer: Sortformer
* [Speech Intent Detection and Slot Filling](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speech_intent_slot/intro.html): Conformer-Transformer

You can also get a high-level overview of NeMo ASR by watching the talk *NVIDIA NeMo: Toolkit for Conversational AI*, presented at PyData Yerevan 2022:


[![NVIDIA NeMo: Toolkit for Conversational AI](https://img.youtube.com/vi/J-P6Sczmas8/maxres3.jpg
)](https://www.youtube.com/embed/J-P6Sczmas8?mute=0&start=14&autoplay=0
 "NeMo presentation at PyData@Yerevan 2022")
