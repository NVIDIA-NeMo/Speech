Automatic Speech Recognition (ASR)
==================================

ASR, or Automatic Speech Recognition, refers to the problem of getting a program to automatically transcribe spoken language 
(speech-to-text). Our goal is usually to have a model that minimizes the Word Error Rate (WER) metric when transcribing speech input. 
In other words, given some audio file (e.g. a WAV file) containing speech, how do we transform this into the corresponding text with 
as few errors as possible?

Traditional speech recognition takes a generative approach, modeling the full pipeline of how speech sounds are produced in order to 
evaluate a speech sample. We would start from a language model that encapsulates the most likely orderings of words that are generated 
(e.g. an n-gram model), to a pronunciation model for each word in that ordering (e.g. a pronunciation table), to an acoustic model that 
translates those pronunciations to audio waveforms (e.g. a Gaussian Mixture Model).

Then, if we receive some spoken input, our goal would be to find the most likely sequence of text that would result in the given audio 
according to our generative pipeline of models. Overall, with traditional speech recognition, we try to model ``Pr(audio|transcript)*Pr(transcript)``, 
and take the argmax of this over possible transcripts.

Over time, neural nets advanced to the point where each component of the traditional speech recognition model could be replaced by a 
neural model that had better performance and that had a greater potential for generalization. For example, we could replace an n-gram 
model with a neural language model, and replace a pronunciation table with a neural pronunciation model, and so on. However, each of 
these neural models need to be trained individually on different tasks, and errors in any model in the pipeline could throw off the 
whole prediction.

Thus, we can see the appeal of end-to-end ASR architectures: discriminative models that simply take an audio input and give a textual 
output, and in which all components of the architecture are trained together towards the same goal. The model's encoder would be 
akin to an acoustic model for extracting speech features, which can then be directly piped to a decoder which outputs text. If desired, 
we could integrate a language model that would improve our predictions, as well.

And the entire end-to-end ASR model can be trained at once--a much easier pipeline to handle!

A demo below allows evaluation of NeMo ASR models in multiple langauges from the browser:

.. raw:: html

    <iframe src="https://hf.space/embed/smajumdar/nemo_multilingual_language_id/+"
    width="100%" class="gradio-asr"></iframe>

    <script type="text/javascript" language="javascript">
        $('.gradio-asr').css('height', $(window).height()+'px');
    </script>


The full documentation tree is as follows:

.. toctree::
   :maxdepth: 8

   models
   datasets
   asr_language_modeling
   results
   scores
   configs
   api
   resources
   examples/kinyarwanda_asr.rst

.. include:: resources.rst
