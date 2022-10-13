.. _punctuation_capitalization_models:

Punctuation And Capitalization Models
==============================================

Automatic Speech Recognition (ASR) systems typically generate text with no punctuation and capitalization of the words.
There are two issues with non-punctuated ASR output:

- it could be difficult to read and understand
- models for some downstream tasks, such as named entity recognition, machine translation, or text-to-speech synthesis, are
  usually trained on punctuated datasets and using raw ASR output as the input to these models could deteriorate their
  performance


NeMo provides two types of Punctuation And Capitalization Models:

Lexical only model:

.. toctree::
   :maxdepth: 1

   punctuation_and_capitalization   


Lexical and audio model:

.. toctree::
   :maxdepth: 1

   punctuation_and_capitalization_lexical_audio

