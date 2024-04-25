# Speech Recognition Tutorials
------------

In this repository, you will find several tutorials discussing what is Automatic Speech Recognition (ASR), general concepts, specific models and multiple sub-domains of ASR such as Speech Classification, Voice Activity Detection, Speaker Recognition, Speaker Identification and Speaker Diarization.


------------

# Automatic Speech Recognition

1) `ASR_with_NeMo`: Discussion of the task of ASR, handling of data, understanding the acoustic features, using an Acoustic Model and train on an ASR dataset, and finally evaluating the model's performance.

2) `ASR_with_Subword_Tokenization`: Modern ASR models benefit from several improvements in neural network design and data processing. In this tutorial we discuss how we can use Tokenizers (commonly found in NLP) to significantly improve the efficiency of ASR models without sacrificing any accuracy during transcription.

3) `ASR_CTC_Language_Finetuning`: Until now, we have discussed how to train ASR models from scratch. Once we get pretrained ASR models, we can then fine-tune them on domain specific use cases, or even other languages! This notebook discusses how to fine-tune an English ASR model onto another language, and discusses several methods to improve the efficiency of transfer learning.

4) `Online_ASR_Microphone_Demo`: A short notebook that enables us to speak into a microphone and transcribe speech in an online manner. Note that this is not the most efficient way to perform streaming ASR, and it is more of a demo.

5) `Online_ASR_Microphone_Demo_Cache_Aware_Streaming`: This notebook allows you to do real-time ("streaming") speech recognition on audio recorded from your microphone, using "cache-aware" NeMo ASR models specifically tuned for the streaming ASR usecase.

6) `ASR_for_telephony_speech`: Audio sources are not homogenous, nor are the ways to store large audio datasets. Here, we discuss our observations and recommendations when working with audio obtained from Telephony speech sources.

7) `Online_Noise_Augmentation`: While academic datasets are useful for training ASR model, there can often be cases where such datasets are pristine and don't really represent the use case in the real world. So we discuss how to make the model more noise robust with Online audio augmentation.

8) `Intro_to_Transducers`: Previous tutorials discuss ASR models in context of the Connectionist Temporal Classification Loss. In this tutorial, we introduce the Transducer loss, and the components of this loss function that are constructed in the config file. This tutorial is a prerequisite to the `ASR_with_Transducers` tutorial.

9) `ASR_with_Transducers`: In this tutorial, we take a deep dive into Transducer based ASR models, discussing the similarity of setup and config to CTC models and then train a small ContextNet model on the AN4 dataset. We then discuss how to change the decoding strategy of a trained Transducer from greedy search to beam search. Finally, we wrap up this tutorial by extraining the alignment matrix from a trained Transducer model. 

10) `Self_Supervised_Pre_Training`: It can often be difficult to obtain labeled data for ASR training. In this tutorial, we demonstrate how to pre-train a speech model in an unsupervised manner, and then fine-tune with CTC loss.

11) `Offline_ASR_with_VAD_for_CTC_models`: In this tutorial, we will demonstrate how to use offline VAD to extract speech segments and transcribe the speech segments with CTC models. This will help to exclude some non_speech utterances and could save computation resources by removing unnecessary input to the ASR system.

12) `Multilang_ASR`: We will learn how to work with existing checkpoints of multilingual ASR models and how to train new ones. It is possible to create a multilingual version of any ASR model that uses tokenizers. This notebook shows how to create a multilingual version of the small monolingual Conformer Transducer model.

13) `ASR_Example_CommonVoice_Finetuning`: Learn how to fine-tune an ASR model using CommonVoice to a new alphabet, Esperanto. We walk through the data processing steps of MCV data using HuggingFace Datasets, preparation of the tokenizer, model and then setup fine-tuning.

14) `ASR_Context_Biasing`: This tutorial aims to show how to improve the recognition accuracy of specific words in NeMo Framework for CTC and Trasducer (RNN-T) ASR models by using the fast context-biasing method with CTC-based Word Spotter.

----------------

# Automatic Speech Recognition with Adapters

Please refer to the `asr_adapter` sub-folder which contains tutorials on the use of `Adapter` modules to perform domain adaptation on ASR models, as well as its sub-domains.

----------------

# Streaming / Buffered Automatic Speech Recognition

1) `Streaming_ASR`: Some ASR models cannot be used to evaluate very long audio segments due to their design. For example, self attention models consume quadratic memory with respect to sequence length. For such cases, this notebook shows how to perform streaming audio recognition in a buffered manner.

2) `Buffered_Transducer_Inference`: In this notebook, we explore a simple algorithm to perform streaming audio recognition in a buffered manner for Transducer models. This enables the use of transducers on very long speech segments, similar to CTC models.

3) `Buffered_Transducer_Inference_with_LCS_Merge`: This is an optional notebook, that discusses a different merge algorithm that can be utilized for streaming/buffered inference for Transducer models. It is not a required tutorial, but is useful for researchers who wish to analyse and improve buffered inference algorithms.

----------------

# Speech Command Recognition

1) `Speech_Commands`: Here, we study the task of speech classification - a subset of speech recognition that allows us to classify a spoken sentence into a single label. This allows to speak a command and the model can then recognize this command and perform an action.

2) `Online_Offline_Speech_Commands_Demo`: We perform a joint online-offline inference of speech command recognition. We utilize an online VAD model to detect speech segments (whether audio is in fact speech or background), and if speech is detected then a speech command recognition model classifies that speech in an offline manner. Note that this demo is a demonstration of a possible approach and is not meant for large scale use.

3) `Voice_Activity_Detection`: A special case of Speech Command Recognition - where the task is to classify whether some audio segment is speech or not. It is often a tiny model that is used prior to a large ASR model being used.

4) `Online_Offline_Microphone_VAD_Demo`: Similar to before, we demo an online-offline inference of voice activity detection. We discuss metrics for comparing the performance of streaming VAD models, and how one can try to perform streaming VAD inference with a microphone. Note that as always, this demo is a demonstration of a possible approach and is not meant for large scale use.
