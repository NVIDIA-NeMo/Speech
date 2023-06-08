# SpellMapper - spellchecking model for ASR Customization
Paper: https://arxiv.org/abs/2306.02317
This model was partly inspired by Microsoft's paper https://arxiv.org/pdf/2203.00888.pdf.
The goal is to build a model that gets as input a single ASR hypothesis (text) and a vocabulary of custom words/phrases and predicts which fragments in the ASR hypothesis should be replaced by which custom words/phrases if any.
Our model is non-autoregressive (NAR) based on transformer architecture (BERT with multiple separators).

As initial data we use about 5 mln entities from [YAGO corpus](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/yago/downloads/). These entities are short phrases from Wikipedia headings.
In order to get misspelled predictions we feed these data to TTS model and then to ASR model.
Having a "parallel" corpus of "correct + misspelled" phrases, we use statistical machine translation techniques to create a dictionary of possible ngram mappings with their respective frequencies.
We create an auxiliary algorithm that takes as input a sentence (ASR hypothesis) and a large custom dictionary (e.g. 5000 phrases) and selects top 10 candidate phrases that are probably contained in this sentence in a misspelled way.
The task of our final neural model is to predict which fragments in the ASR hypothesis should be replaced by which of top-10 candidate phrases if any.

The pipeline consists of multiple steps:

1. Download or generate training data. 
   See `https://github.com/bene-ges/nemo_compatible/tree/main/scripts/nlp/en_spellmapper/dataset_preparation`

2. [Optional] Convert training dataset to tarred files.
   `convert_dataset_to_tarred.sh`
 
3. Train spellchecking model.
   `run_training.sh`
   or 
   `run_training_tarred.sh`

4. Run evaluation.
   - [test_on_kensho.sh](https://github.com/bene-ges/nemo_compatible/blob/main/scripts/nlp/en_spellmapper/evaluation/test_on_kensho.sh)
   - [test_on_userlibri.sh](https://github.com/bene-ges/nemo_compatible/blob/main/scripts/nlp/en_spellmapper/evaluation/test_on_kensho.sh)
   - [test_on_spoken_wikipedia.sh](https://github.com/bene-ges/nemo_compatible/blob/main/scripts/nlp/en_spellmapper/evaluation/test_on_kensho.sh)

5. Run inference.
   `python run_infer.sh`
