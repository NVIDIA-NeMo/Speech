Speaker tasks in general are broadly classified into two tasks:
- [Speaker Recognition](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/intro.html)
- [Speaker Diarization](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_diarization/intro.html)

**Speaker Recognition** is a research area which solves two major tasks: speaker identification (what is the identity of the speaker?) and speaker verification (is the speaker who they claim to be?). Whereas **Speaker Diarization** is a task segmenting audio recordings by speaker labels (Who Speaks When?). 

In *recognition* folder we provide scripts for training, inference and verification of audio samples.   
In *diarization* folder we provide scripts for inference of speaker diarization using pretrained VAD (optional) and Speaker embedding extractor  models
