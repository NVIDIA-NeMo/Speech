#######################################
ASR Language Modeling and Customization
#######################################

Language models have shown to help the accuracy of ASR models. NeMo supports the following two approaches to incorporate language models into the ASR models:

*  :ref:`ngram_modeling`
*  :ref:`neural_rescoring`

It is possible to use both approaches on the same ASR model.


.. _ngram_modeling:

************************
N-gram Language Modeling
************************

In this approach, an N-gram LM is trained on text data, then it is used in fusion with beam search decoding to find the
best candidates. The beam search decoders in NeMo support language models trained with KenLM library (
`https://github.com/kpu/kenlm <https://github.com/kpu/kenlm>`__).
The beam search decoders and KenLM library are not installed by default in NeMo, and you need to install them to be
able to use beam search decoding and N-gram LM.
Please refer to `scripts/asr_language_modeling/ngram_lm/install_beamsearch_decoders.sh <https://github.com/NVIDIA/NeMo/blob/stable/scripts/asr_language_modeling/ngram_lm/install_beamsearch_decoders.sh>`__
on how to install them. Alternatively, you can build Docker image
`scripts/installers/Dockerfile.ngramtools <https://github.com/NVIDIA/NeMo/blob/stable/scripts/installers/Dockerfile.ngramtools>`__ with all the necessary dependencies.

NeMo supports both character-based and BPE-based models for N-gram LMs. An N-gram LM can be used with beam search
decoders on top of the ASR models to produce more accurate candidates. The beam search decoder would incorporate
the scores produced by the N-gram LM into its score calculations as the following:

.. code-block::

    final_score = acoustic_score + beam_alpha*lm_score + beam_beta*seq_length

where acoustic_score is the score predicted by the acoustic encoder and lm_score is the one estimated by the LM.
Parameter 'beam_alpha' specifies amount of importance to place on the N-gram language model, and 'beam_beta' is a
penalty term to consider the sequence length in the scores. Larger alpha means more importance on the LM and less
importance on the acoustic model. Negative values for beta will give penalty to longer sequences and make the decoder
to prefer shorter predictions, while positive values would result in longer candidates.

.. _train-ngram-lm:

Train N-gram LM
===============

The script to train an N-gram language model with KenLM can be found at
`scripts/asr_language_modeling/ngram_lm/train_kenlm.py <https://github.com/NVIDIA/NeMo/blob/stable/scripts/asr_language_modeling/ngram_lm/train_kenlm.py>`__.

This script would train an N-gram language model with KenLM library which can be used with the beam search decoders
on top of the ASR models. This script supports both character level and BPE level encodings and models which are
detected automatically from the type of the model.


You may train the N-gram model as the following:

.. code-block::

    python train_kenlm.py nemo_model_file=<path to the .nemo file of the model> \
                              train_paths=<list of paths to the training text or JSON manifest files> \
                              kenlm_bin_path=<path to the bin folder of KenLM library> \
                              kenlm_model_file=<path to store the binary KenLM model> \
                              ngram_length=<order of N-gram model> \
                              preserve_arpa=true

The `train_paths` parameter allows for various input types, such as a list of text files, JSON manifests, or directories, to be used as the training data.
If the file's extension is anything other than `.json`, it assumes that data format is plain text. For plain text format, each line should contain one
sample. For JSON manifest file, the file need to contain json formatted samples per each line like this:

.. code-block::

    {"audio_filepath": "/data_path/file1.wav", "text": "The transcript of the audio file."}

It just extracts the `text` field from each line to create the training text file. After the N-gram model is trained,
it is stored at the path specified by `kenlm_model_file`.

The following is the list of the arguments for the training script:

+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| **Argument**     | **Type**  | **Default** | **Description**                                                                                                                |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| nemo_model_file  | str       | Required    | The path to `.nemo` file of the ASR model, or name of a pretrained NeMo model to extract a tokenizer.                          |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| train_paths      | List[str] | Required    | List of training files or folders. Files can be a plain text file or ".json" manifest or ".json.gz".                           |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| kenlm_model_file | str       | Required    | The path to store the KenLM binary model file.                                                                                 |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| kenlm_bin_path   | str       | Required    | The path to the bin folder of KenLM. It is a folder named `bin` under where KenLM is installed.                                |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| ngram_length**   | int       | Required    | Specifies order of N-gram LM.                                                                                                  |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| ngram_prune      | List[int] | [0]         | List of thresholds to prune N-grams. Example: [0,0,1]. See Pruning section on the https://kheafield.com/code/kenlm/estimation  |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| cache_path       | str       | ``""``      | Cache path to save tokenized files.                                                                                            |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| preserve_arpa    | bool      | ``False``   | Whether to preserve the intermediate ARPA file after construction of the BIN file.                                             |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| verbose          | int       | 1           | Verbose level.                                                                                                                 |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+

** Note: Recommend to use 6 as the order of the N-gram model for BPE-based models. Higher orders may need the re-compilation of KenLM to support it.

Evaluate by Beam Search Decoding and N-gram LM
==============================================

NeMo's beam search decoders are capable of using the KenLM's N-gram models to find the best candidates.
The script to evaluate an ASR model with beam search decoding and N-gram models can be found at
`scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram.py <https://github.com/NVIDIA/NeMo/blob/stable/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram.py>`__.

This script has a large number of possible argument overrides, therefore it is advised to use ``python eval_beamsearch_ngram.py --help`` to see the full list of arguments.

You may evaluate an ASR model as the following:

.. code-block::

    python eval_beamsearch_ngram.py nemo_model_file=<path to the .nemo file of the model> \
           input_manifest=<path to the evaluation JSON manifest file \
           kenlm_model_file=<path to the binary KenLM model> \
           beam_width=[<list of the beam widths, separated with commas>] \
           beam_alpha=[<list of the beam alphas, separated with commas>] \
           beam_beta=[<list of the beam betas, separated with commas>] \
           preds_output_folder=<optional folder to store the predictions> \
           probs_cache_file=null \
           decoding_mode=beamsearch_ngram \
           decoding_strategy="<Beam library such as beam, pyctcdecode or flashlight>"

It can evaluate a model in the three following modes by setting the argument `--decoding_mode`:

*  greedy: Just greedy decoding is done, and no beam search decoding is performed.
*  beamsearch: The beam search decoding is done but without using the N-gram language model, final results would be equivalent to setting the weight of LM (beam_beta) to zero.
*  beamsearch_ngram: The beam search decoding is done with N-gram LM.

The `beamsearch` mode would evaluate by beam search decoding without any language model.
It would report the performances in terms of Word Error Rate (WER) and Character Error Rate (CER). Moreover,
the WER/CER of the model when the best candidate is selected among the candidates is also reported as the best WER/CER.
It can be an indicator of how good the predicted candidates are.

The script would initially load the ASR model and predict the outputs of the model's encoder as log probabilities.
This part would be computed in batches on a device selected by `--device`, which can be CPU (`--device=cpu`) or a
single GPU (`--device=cuda:0`). The batch size of this part can get specified by `--acoustic_batch_size`. You may use
the largest batch size feasible to speed up the step of calculating the log probabilities. You may also use `--use_amp`
to speed up the calculation of log probabilities and make it possible to use larger sizes for `--acoustic_batch_size`.
Currently multi-GPU is not supported for calculating the log probabilities, but using `--probs_cache_file` can help.
It stores the log probabilities produced from the model's encoder into a pickle file so that next time the first step
can get skipped.

The following is the list of the important arguments for the evaluation script:

+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| **Argument**                         | **Type** | **Default**      | **Description**                                                         |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| nemo_model_file                      | str      | Required         | The path of the `.nemo` file of the ASR model to extract the tokenizer. |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| input_manifest                       | str      | Required         | Path to the training file, it can be a text file or JSON manifest.      |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| kenlm_model_file                     | str      | Required         | The path to store the KenLM binary model file.                          |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| preds_output_folder                  | str      | None             | The path to an optional folder to store the predictions.                |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| probs_cache_file                     | str      | None             | The cache file for storing the outputs of the model.                    |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| acoustic_batch_size                  | int      | 16               | The batch size to calculate log probabilities.                          |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| use_amp                              | bool     | False            | Whether to use AMP if available to calculate log probabilities.         |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| device                               | str      | cuda             | The device to load the model onto to calculate log probabilities.       |
|                                      |          |                  | It can `cpu`, `cuda`, `cuda:0`, `cuda:1`, ...                           |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| decoding_mode                        | str      | beamsearch_ngram | The decoding scheme to be used for evaluation.                          |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| beam_width                           | float    | Required         | List of the width or list of the widths of the beam search decoding.    |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| beam_alpha                           | float    | Required         | List of the alpha parameter for the beam search decoding.               |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| beam_beta                            | float    | Required         | List of the beta parameter for the beam search decoding.                |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| beam_batch_size                      | int      | 128              | The batch size to be used for beam search decoding.                     |
|                                      |          |                  | Larger batch size can be a little faster, but uses larger memory.       |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| decoding_strategy                    | str      | beam             | String argument for type of decoding strategy for the model.            |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| decoding                             | Dict     | BeamCTC          | Subdict of beam search configs. Values found via                        |
|                                      | Config   | InferConfig      | python eval_beamsearch_ngram.py --help                                  |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| text_processing.do_lowercase         | bool     | ``False``        | Whether to make the training text all lower case.                       |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| text_processing.punctuation_marks    | str      | ``""``           | String with punctuation marks to process. Example: ".\,?"               |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| text_processing.rm_punctuation       |  bool    | ``False``        | Whether to remove punctuation marks from text.                          |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| text_processing.separate_punctuation | bool     | ``True``         | Whether to separate punctuation with the previous word by space.        |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+

Width of the beam search (`--beam_width`) specifies the number of top candidates/predictions the beam search decoder
would search for. Larger beams result in more accurate but slower predictions.

.. note::

    The ``eval_beamsearch_ngram.py`` script contains the entire subconfig used for CTC Beam Decoding.
    Therefore it is possible to forward arguments for various beam search libraries such as ``flashlight``
    and ``pyctcdecode`` via the ``decoding`` subconfig.

There is also a tutorial to learn more about evaluating the ASR models with N-gram LM here:
`Offline ASR Inference with Beam Search and External Language Model Rescoring <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Offline_ASR.ipynb>`_

Beam Search Engines
-------------------

NeMo ASR CTC supports multiple beam search engines for decoding. The default engine is ``beam`` which is the OpenSeq2Seq
decoding library.

OpenSeq2Seq (``beam``)
~~~~~~~~~~~~~~~~~~~~~~

CPU-based beam search engine that is quite efficient and supports char and subword models. It requires a character/subword
KenLM model to be provided.

The config for this decoding library is described above.

Flashlight (``flashlight``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Flashlight is a C++ library for ASR decoding provided at `https://github.com/flashlight/flashlight <https://github.com/flashlight/flashlight>`_. It is a CPU and CUDA-based beam search engine that is quite efficient and supports
char and subword models. It requires an ARPA KenLM file.

It supports several advanced features such as lexicon based / lexicon free decoding, beam pruning threshold, and more.

.. code-block:: python

    @dataclass
    class FlashlightConfig:
        lexicon_path: Optional[str] = None
        boost_path: Optional[str] = None
        beam_size_token: int = 16
        beam_threshold: float = 20.0
        unk_weight: float = -math.inf
        sil_weight: float = 0.0

.. code-block::

    # Lexicon-based decoding
    python eval_beamsearch_ngram.py ... \
           decoding_strategy="flashlight" \
           decoding.beam.flashlight_cfg.lexicon_path='/path/to/lexicon.lexicon' \
           decoding.beam.flashlight_cfg.beam_size_token = 32 \
           decoding.beam.flashlight_cfg.beam_threshold = 25.0

    # Lexicon-free decoding
    python eval_beamsearch_ngram.py ... \
           decoding_strategy="flashlight" \
           decoding.beam.flashlight_cfg.beam_size_token = 32 \
           decoding.beam.flashlight_cfg.beam_threshold = 25.0

PyCTCDecode (``pyctcdecode``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyCTCDecode is a Python library for ASR decoding provided at `https://github.com/kensho-technologies/pyctcdecode <https://github.com/kensho-technologies/pyctcdecode>`_. It is a CPU-based beam search engine that is somewhat efficient for a pure python library, and supports char and subword models. It requires a character/subword KenLM ARPA / BINARY model to be provided.

It has advanced features such as word boosting which can be useful for transcript customization.

.. code-block:: python

   @dataclass
    class PyCTCDecodeConfig:
        beam_prune_logp: float = -10.0
        token_min_logp: float = -5.0
        prune_history: bool = False
        hotwords: Optional[List[str]] = None
        hotword_weight: float = 10.0

.. code-block::

    # PyCTCDecoding
    python eval_beamsearch_ngram.py ... \
           decoding_strategy="pyctcdecode" \
           decoding.beam.pyctcdecode_cfg.beam_prune_logp = -10. \
           decoding.beam.pyctcdecode_cfg.token_min_logp = -5. \
           decoding.beam.pyctcdecode_cfg.hotwords=[<List of str words>] \
           decoding.beam.pyctcdecode_cfg.hotword_weight=10.0


Hyperparameter Grid Search
--------------------------

Beam search decoding with N-gram LM has three main hyperparameters: `beam_width`, `beam_alpha`, and `beam_beta`.
The accuracy of the model is dependent to the values of these parameters, specially beam_alpha and beam_beta.
You may specify a single or list of values for each of these parameters to perform grid search. It would perform the
beam search decoding on all the combinations of the these three hyperparameters.
For instance, the following set of parameters would results in 2*1*2=4 beam search decodings:

.. code-block::

    python eval_beamsearch_ngram.py ... \
                        beam_width=[64,128] \
                        beam_alpha=[1.0] \
                        beam_beta=[1.0,0.5]


Beam search ngram decoding for Transducer models (RNNT and HAT)
===============================================================

The similar script to evaluate an RNNT/HAT model with beam search decoding and N-gram models can be found at
`scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_transducer.py <https://github.com/NVIDIA/NeMo/blob/stable/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_transducer.py>`_

.. code-block::

    python eval_beamsearch_ngram_transducer.py nemo_model_file=<path to the .nemo file of the model> \
            input_manifest=<path to the evaluation JSON manifest file \
            kenlm_model_file=<path to the binary KenLM model> \
            beam_width=[<list of the beam widths, separated with commas>] \
            beam_alpha=[<list of the beam alphas, separated with commas>] \
            preds_output_folder=<optional folder to store the predictions> \
            probs_cache_file=null \
            decoding_strategy=<greedy_batch or maes decoding>
            maes_prefix_alpha=[<list of the maes prefix alphas, separated with commas>] \
            maes_expansion_gamma=[<list of the maes expansion gammas, separated with commas>] \
            hat_subtract_ilm=<in case of HAT model: subtract internal LM or not (True/False)> \
            hat_ilm_weight=[<in case of HAT model: list of the HAT internal LM weights, separated with commas>] \



.. _neural_rescoring:

****************
Neural Rescoring
****************

In this approach a neural network is used which can gives scores to a candidate. A candidate is the text transcript predicted by the decoder of the ASR model.
The top K candidates produced by the beam search decoding (beam width of K) are given to a neural language model to rank them.
Ranking can be done by a language model which gives a score to each candidate.
This score is usually combined with the scores from the beam search decoding to produce the final scores and rankings.

Train Neural Rescorer
=====================

An example script to train such a language model with Transformer can be found at `examples/nlp/language_modeling/transformer_lm.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/language_modeling/transformer_lm.py>`__.
It trains a ``TransformerLMModel`` which can be used as a neural rescorer for an ASR system. Full documentation on language models training is available at:

:doc:`../nlp/language_modeling`

You may also use a pretrained language model from HuggingFace library like Transformer-XL and GPT instead of training your model.
Models like BERT and RoBERTa are not supported by this script as they are trained as a Masked Language Model and are not efficient and effective to score sentences out of the box.


Evaluation
==========

Given a trained TransformerLMModel `.nemo` file or a pretrained HF model, the script available at
`scripts/asr_language_modeling/neural_rescorer/eval_neural_rescorer.py <https://github.com/NVIDIA/NeMo/blob/stable/scripts/asr_language_modeling/neural_rescorer/eval_neural_rescorer.py>`__
can be used to re-score beams obtained with ASR model. You need the `.tsv` file containing the candidates produced
by the acoustic model and the beam search decoding to use this script. The candidates can be the result of just the beam
search decoding or the result of fusion with an N-gram LM. You may generate this file by specifying `--preds_output_folder` for
`scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram.py <https://github.com/NVIDIA/NeMo/blob/stable/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram.py>`__.

The neural rescorer would rescore the beams/candidates by using two parameters of `rescorer_alpha` and `rescorer_beta` as the following:

.. code-block::

    final_score = beam_search_score + rescorer_alpha*neural_rescorer_score + rescorer_beta*seq_length

Parameter `rescorer_alpha` specifies amount of importance to place on the neural rescorer model, and `rescorer_beta` is
a penalty term to consider the sequence length in the scores. They have similar effects like the parameters
`beam_alpha` and `beam_beta` of beam search decoder and N-gram LM.

You may follow the following steps to evaluate a neural LM:

#. Obtain `.tsv` file with beams and their corresponding scores. Scores can be from a regular beam search decoder or
   in fusion with an N-gram LM scores. For a given beam size `beam_size` and a number of examples
   for evaluation `num_eval_examples`, it should contain (`num_eval_examples` x `beam_size`) lines of
   form `beam_candidate_text \t score`. This file can be generated by `scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram.py <https://github.com/NVIDIA/NeMo/blob/stable/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram.py>`__

#. Rescore the candidates by `scripts/asr_language_modeling/neural_rescorer/eval_neural_rescorer.py <https://github.com/NVIDIA/NeMo/blob/stable/scripts/asr_language_modeling/neural_rescorer/eval_neural_rescorer.py>`__.

.. code-block::

    python eval_neural_rescorer.py
        --lm_model=[path to .nemo file of the LM or the name of a HF pretrained model]
        --beams_file=[path to beams .tsv file]
        --beam_size=[size of the beams]
        --eval_manifest=[path to eval manifest .json file]
        --batch_size=[batch size used for inference on the LM model]
        --alpha=[the value for the parameter rescorer_alpha]
        --beta=[the value for the parameter rescorer_beta]
        --scores_output_file=[the optional path to store the rescored candidates]

The candidates along with their new scores would be stored at the file specified by `--scores_output_file`.

The following is the list of the arguments for the evaluation script:

+---------------------+--------+------------------+-------------------------------------------------------------------------+
| **Argument**        |**Type**| **Default**      | **Description**                                                         |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| lm_model            | str    | Required         | The path of the '.nemo' file of an ASR model, or the name of a          |
|                     |        |                  | HuggingFace pretrained model like 'transfo-xl-wt103' or 'gpt2'          |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| eval_manifest       | str    | Required         | Path to the evaluation manifest file (.json manifest file)              |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| beams_file          | str    | Required         | path to beams file (.tsv) containing the candidates and their scores    |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| beam_size           | int    | Required         | The width of the beams (number of candidates) generated by the decoder  |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| alpha               | float  | None             | The value for parameter rescorer_alpha                                  |
|                     |        |                  | Not passing value would enable linear search for rescorer_alpha         |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| beta                | float  | None             | The value for parameter rescorer_beta                                   |
|                     |        |                  | Not passing value would enable linear search for rescorer_beta          |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| batch_size          | int    | 16               | The batch size used to calculate the scores                             |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| max_seq_length      | int    | 512              | Maximum sequence length (in tokens) for the input                       |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| scores_output_file  | str    | None             | The optional file to store the rescored beams                           |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| use_amp             | bool   | ``False``        | Whether to use AMP if available calculate the scores                    |
+---------------------+--------+------------------+-------------------------------------------------------------------------+
| device              | str    | cuda             | The device to load LM model onto to calculate the scores                |
|                     |        |                  | It can be 'cpu', 'cuda', 'cuda:0', 'cuda:1', ...                        |
+---------------------+--------+------------------+-------------------------------------------------------------------------+


Hyperparameter Linear Search
----------------------------

This script also supports linear search for parameters `alpha` and `beta`. If any of the two is not
provided, a linear search is performed to find the best value for that parameter. When linear search is used, initially
`beta` is set to zero and the best value for `alpha` is found, then `alpha` is fixed with
that value and another linear search is done to find the best value for `beta`.
If any of the of these two parameters is already specified, then search for that one is skipped. After each search for a
parameter, the plot of WER% for different values of the parameter is also shown.

It is recommended to first use the linear search for both parameters on a validation set by not providing any values for `--alpha` and `--beta`.
Then check the WER curves and decide on the best values for each parameter. Finally, evaluate the best values on the test set.


Word Boosting
=============

The Flashlight decoder supports word boosting during CTC decoding using a KenLM binary and corresponding lexicon. Word boosting only
works in lexicon decoding mode, it does not work in lexicon-free mode. Word boosting allows one to bias the decoder for certain words,
such that you can manually increase or decrease the probability of emitting certain words. This can be very helpful if you have certain
uncommon or industry-specific words which you want to ensure transcribe correctly.

For more information on word boosting, `here <https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-customizing.html#word-boosting>`__

In order to use word boosting in Nemo, you need to create a simple tab-separated text file which contains each word to be boosted, followed by
tab, and then the boosted score for that word.

For example:

.. code-block::

    nvidia	40
    geforce	50
    riva	80
    turing	30
    badword	-100

Positive scores boost words higher in the LM decoding step so they show up more frequently, whereas negative scores
squelch words so they show up less frequently. The recommended range for the boost score is +/- 20 to 100.

The boost file handles both in-vocabulary words and OOV words just fine, so you can specify both IV and OOV words with corresponding scores.

You can then pass this file to your flashlight config object during decoding:

.. code-block::

    # Lexicon-based decoding
    python eval_beamsearch_ngram.py ... \
           decoding_strategy="flashlight" \
           decoding.beam.flashlight_cfg.lexicon_path='/path/to/lexicon.lexicon' \
           decoding.beam.flashlight_cfg.boost_path='/path/to/my_boost_file.boost' \
           decoding.beam.flashlight_cfg.beam_size_token = 32 \
           decoding.beam.flashlight_cfg.beam_threshold = 25.0


Combine N-gram Language Models
==============================

Before combining N-gram LMs install required OpenGrm NGram library using `scripts/installers/install_opengrm.sh <https://github.com/NVIDIA/NeMo/blob/stable/scripts/installers/install_opengrm.sh>`__.
Alternatively, you can use Docker image `scripts/installers/Dockerfile.ngramtools <https://github.com/NVIDIA/NeMo/blob/stable/scripts/installers/Dockerfile.ngramtools>`__ with all the necessary dependencies.

To combine two N-gram language models, you can use the script ngram_merge.py located at
`scripts/asr_language_modeling/ngram_lm/ngram_merge.py <https://github.com/NVIDIA/NeMo/blob/stable/scripts/asr_language_modeling/ngram_lm/ngram_merge.py>`__.

This script interpolate two ARPA N-gram language models and creates a KenLM binary file that can be used with the beam search decoders on top of ASR models.
You can specify weights (`--alpha` and `--beta`) for each of the models (`--ngram_a` and `--ngram_b`) correspondingly: `alpha` * `ngram_a` + `beta` * `ngram_b`.
This script supports both character level and BPE level encodings and models which are detected automatically from the type of the model.

To combine two N-gram models, you can use the following command:

.. code-block::

    python ngram_merge.py  --kenlm_bin_path <path to the bin folder of KenLM library> \
                    --ngram_bin_path  <path to the bin folder of OpenGrm Ngram library> \
                    --arpa_a <path to the ARPA N-gram model file A> \
                    --alpha <weight of N-gram model A> \
                    --arpa_b <path to the ARPA N-gram model file B> \
                    --beta <weight of N-gram model B> \
                    --out_path <path to folder to store the output files>



If you provide `--test_file` and `--nemo_model_file`, the script will calculate the perplexity of the resulting N-gram model on the test set.
Note, the result of each step during the process is cached in the temporary file in the `--out_path`, to speed up further run.
You can use the `--force` flag to discard the cache and recalculate everything from scratch.

.. code-block::

    python ngram_merge.py  --kenlm_bin_path <path to the bin folder of KenLM library> \
                    --ngram_bin_path  <path to the bin folder of OpenGrm Ngram library> \
                    --arpa_a <path to the ARPA N-gram model file A> \
                    --alpha <weight of N-gram model A> \
                    --arpa_b <path to the ARPA N-gram model file B> \
                    --beta <weight of N-gram model B> \
                    --out_path <path to folder to store the output files>
                    --nemo_model_file <path to the .nemo file of the model> \
                    --test_file <path to the test file> \
                    --symbols <path to symbols (.syms) file> \
                    --force <flag to recalculate and rewrite all cached files>


The following is the list of the arguments for the opengrm script:

+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| **Argument**         |**Type**| **Default**      | **Description**                                                                                                 |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| kenlm_bin_path       | str    | Required         | The path to the bin folder of KenLM library. It is a folder named `bin` under where KenLM is installed.         |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| ngram_bin_path       | str    | Required         | The path to the bin folder of OpenGrm Ngram. It is a folder named `bin` under where OpenGrm Ngram is installed. |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| arpa_a               | str    | Required         | Path to the ARPA N-gram model file A                                                                            |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| alpha                | float  | Required         | Weight of N-gram model A                                                                                        |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| arpa_b               | int    | Required         | Path to the ARPA N-gram model file B                                                                            |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| beta                 | float  | Required         | Weight of N-gram model B                                                                                        |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| out_path             | str    | Required         | Path for writing temporary and resulting files.                                                                 |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| test_file            | str    | None             | Path to test file to count perplexity if provided.                                                              |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| symbols              | str    | None             | Path to symbols (.syms) file. Could be calculated if it is not provided.                                        |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| nemo_model_file      | str    | None             | The path to '.nemo' file of the ASR model, or name of a pretrained NeMo model.                                  |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| force                | bool   | ``False``        | Whether to recompile and rewrite all files                                                                      |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+


***************************************************
Context-biasing (word boosting) without external LM
***************************************************

NeMo toolkit supports a fast context-biasing method for CTC and Transducer (RNN-T) ASR models with CTC-based Word Spotter.
The method involves decoding CTC log probabilities with a context graph built for words and phrases from the context-biasing list.
The spotted context-biasing candidates (with their scores and time intervals) are compared by scores with words from the greedy CTC decoding results to improve recognition accuracy and pretend false accepts of context-biasing.

A Hybrid Transducer-CTC model (a shared encoder trained together with CTC and Transducer output heads) enables the use of the CTC-WS method for the Transducer model.
Context-biasing candidates obtained by CTC-WS are also filtered by the scores with greedy CTC predictions and then merged with greedy Transducer results.

Scheme of the CTC-WS method:

.. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.22.0/asset-post-v1.22.0-ctcws_scheme_1.png
    :align: center
    :alt: CTC-WS scheme
    :scale: 40%

High-level overview of the context-biasing words replacement with CTC-WS method:

.. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.22.0/asset-post-v1.22.0-ctcws_scheme_2.png
    :align: center
    :alt: CTC-WS high level overview
    :scale: 40%

More details about CTC-WS context-biasing can be found in the `tutorial <https://github.com/NVIDIA/NeMo/tree/main/tutorials/asr/ASR_Context_Biasing.ipynb>`__.

To use CTC-WS context-biasing, you need to create a context-biasing text file that contains words/phrases to be boosted, with its transcriptions (spellings) separated by underscore.
Multiple transcriptions can be useful for abbreviations ("gpu" -> "g p u"), compound words ("nvlink" -> "nv link"), 
or words with common mistakes in the case of our ASR model ("nvidia" -> "n video").

Example of the context-biasing file:

.. code-block::

    nvidia_nvidia
    omniverse_omniverse
    gpu_gpu_g p u
    dgx_dgx_d g x_d gx
    nvlink_nvlink_nv link
    ray tracing_ray tracing

The main script for CTC-WS context-biasing in NeMo is: 

.. code-block::

    {NEMO_DIR_PATH}/scripts/asr_context_biasing/eval_greedy_decoding_with_context_biasing.py

Context-biasing is managed by ``apply_context_biasing`` parameter [true or false].
Other important context-biasing parameters are:

*  ``beam_threshold`` - threshold for CTC-WS beam pruning
*  ``context_score`` - per token weight for context biasing
*  ``ctc_ali_token_weight`` - per token weight for CTC alignment (prevents false acceptances of context-biasing words)

All the context-biasing parameters are selected according to the default values in the script.
You can tune them according to your data and ASR model (list all the values in the [] separated by commas)
for example: ``beam_threshold=[7.0,8.0,9.0]``, ``context_score=[3.0,4.0,5.0]``, ``ctc_ali_token_weight=[0.5,0.6,0.7]``.
The script will run the recognition with all the combinations of the parameters and will select the best one based on WER value.

.. code-block::

    # Context-biasing with the CTC-WS method for CTC ASR model 
    python {NEMO_DIR_PATH}/scripts/asr_context_biasing/eval_greedy_decoding_with_context_biasing.py \
            nemo_model_file={ctc_model_name} \
            input_manifest={test_nemo_manifest} \
            preds_output_folder={exp_dir} \
            decoder_type="ctc" \
            acoustic_batch_size=64 \
            apply_context_biasing=true \
            context_file={cb_list_file_modified} \
            beam_threshold=[7.0] \
            context_score=[3.0] \
            ctc_ali_token_weight=[0.5]

To use Transducer head of the Hybrid Transducer-CTC model, you need to set ``decoder_type=rnnt``.
