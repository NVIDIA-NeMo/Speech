.. _ngram_modeling:

****************************
N-gram Language Model Fusion
****************************

In this approach, an N-gram LM is trained on text data, then it is used in fusion with beam search decoding to find the
best candidates. The beam search decoders in NeMo support language models trained with KenLM library (
`https://github.com/kpu/kenlm <https://github.com/kpu/kenlm>`__).
The beam search decoders and KenLM library are not installed by default in NeMo. 
You need to install them to be able to use beam search decoding and N-gram LM.
Please refer to `scripts/asr_language_modeling/ngram_lm/install_beamsearch_decoders.sh <https://github.com/NVIDIA/NeMo/blob/stable/scripts/asr_language_modeling/ngram_lm/install_beamsearch_decoders.sh>`__
on how to install them. Alternatively, you can build Docker image
`scripts/installers/Dockerfile.ngramtools <https://github.com/NVIDIA/NeMo/blob/stable/scripts/installers/Dockerfile.ngramtools>`__ with all the necessary dependencies.

Please, refer to :ref:`train-ngram-lm` for more details on how to train an N-gram LM using KenLM library.

NeMo supports both character-based and BPE-based models for N-gram LMs. An N-gram LM can be used with beam search
decoders on top of the ASR models to produce more accurate candidates. The beam search decoder would incorporate
the scores produced by the N-gram LM into its score calculations as the following:

.. code-block::

    final_score = acoustic_score + beam_alpha*lm_score + beam_beta*seq_length

where acoustic_score is the score predicted by the acoustic encoder and lm_score is the one estimated by the LM.
The parameter 'beam_alpha' determines the weight given to the N-gram language model, while 'beam_beta' is a penalty term that accounts for sequence length in the scores. A larger 'beam_alpha' places more emphasis on the language model and less on the acoustic model. Negative values for 'beam_beta' penalize longer sequences, encouraging the decoder to prefer shorter predictions. Conversely, positive values for 'beam_beta' favor longer candidates.

Evaluate by Beam Search Decoding and N-gram LM
==============================================

NeMo's beam search decoders are capable of using the KenLM's N-gram models to find the best candidates.
The script to evaluate an ASR model with beam search decoding and N-gram models can be found at
`scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_ctc.py <https://github.com/NVIDIA/NeMo/blob/stable/scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_ctc.py>`__.

This script has a large number of possible argument overrides; therefore, it is recommended that you use ``python eval_beamsearch_ngram_ctc.py --help`` to see the full list of arguments.

You can evaluate an ASR model using the following:

.. code-block::

    python eval_beamsearch_ngram_ctc.py nemo_model_file=<path to the .nemo file of the model> \
           input_manifest=<path to the evaluation JSON manifest file \
           kenlm_model_file=<path to the binary KenLM model> \
           beam_width=[<list of the beam widths, separated with commas>] \
           beam_alpha=[<list of the beam alphas, separated with commas>] \
           beam_beta=[<list of the beam betas, separated with commas>] \
           preds_output_folder=<optional folder to store the predictions> \
           probs_cache_file=null \
           decoding_mode=beamsearch_ngram \
           decoding_strategy="<Beam library such as beam, pyctcdecode or flashlight>"

It can evaluate a model in the following three modes by setting the argument ``--decoding_mode``:

*  greedy: Just greedy decoding is done and no beam search decoding is performed.
*  beamsearch: The beam search decoding is done, but without using the N-gram language model. Final results are equivalent to setting the weight of LM (beam_beta) to zero.
*  beamsearch_ngram: The beam search decoding is done with N-gram LM.

In ``beamsearch`` mode, the evaluation is performed using beam search decoding without any language model. The performance is reported in terms of Word Error Rate (WER) and Character Error Rate (CER). Moreover, when the best candidate is selected among the candidates, it is also reported as the best WER/CER. This can serve as an indicator of the quality of the predicted candidates.


The script initially loads the ASR model and predicts the outputs of the model's encoder as log probabilities. This part is computed in batches on a device specified by --device, which can be either a CPU (`--device=cpu`) or a single GPU (`--device=cuda:0`).
The batch size for this part is specified by ``--acoustic_batch_size``. Using the largest feasible batch size can speed up the calculation of log probabilities. Additionally, you can use `--use_amp` to accelerate the calculation and allow for larger --acoustic_batch_size values.
Currently, multi-GPU support is not available for calculating log probabilities. However, using ``--probs_cache_file`` can help. This option stores the log probabilities produced by the model's encoder in a pickle file, allowing you to skip the first step in future runs.

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
|                                      | Config   | InferConfig      | python eval_beamsearch_ngram_ctc.py --help                              |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| text_processing.do_lowercase         | bool     | ``False``        | Whether to make the training text all lower case.                       |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| text_processing.punctuation_marks    | str      | ``""``           | String with punctuation marks to process. Example: ".\,?"               |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| text_processing.rm_punctuation       |  bool    | ``False``        | Whether to remove punctuation marks from text.                          |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+
| text_processing.separate_punctuation | bool     | ``True``         | Whether to separate punctuation with the previous word by space.        |
+--------------------------------------+----------+------------------+-------------------------------------------------------------------------+

The width of the beam search (``--beam_width``) specifies the number of top candidates or predictions the beam search decoder will consider. Larger beam widths result in more accurate but slower predictions.

.. note::

    The ``eval_beamsearch_ngram_ctc.py`` script contains the entire subconfig used for CTC Beam Decoding.
    Therefore it is possible to forward arguments for various beam search libraries such as ``flashlight``
    and ``pyctcdecode`` via the ``decoding`` subconfig.

To learn more about evaluating the ASR models with N-gram LM, refer to the tutorial here: Offline ASR Inference with Beam Search and External Language Model Rescoring
`Offline ASR Inference with Beam Search and External Language Model Rescoring <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/asr/Offline_ASR.ipynb>`_

Beam Search Engines
-------------------

NeMo ASR CTC supports multiple beam search engines for decoding. The default engine is beam, which is the OpenSeq2Seq decoding library.

OpenSeq2Seq (``beam``)
~~~~~~~~~~~~~~~~~~~~~~

CPU-based beam search engine that is quite efficient and supports char and subword models. It requires a character/subword
KenLM model to be provided.

The config for this decoding library is described above.

Flashlight (``flashlight``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Flashlight is a C++ library for ASR decoding provided at `https://github.com/flashlight/flashlight <https://github.com/flashlight/flashlight>`_. It is a CPU- and CUDA-based beam search engine that is quite efficient and supports char and subword models. It requires an ARPA KenLM file.

It supports several advanced features, such as lexicon-based decoding, lexicon-free decoding, beam pruning threshold, and more.

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
    python eval_beamsearch_ngram_ctc.py ... \
           decoding_strategy="flashlight" \
           decoding.beam.flashlight_cfg.lexicon_path='/path/to/lexicon.lexicon' \
           decoding.beam.flashlight_cfg.beam_size_token = 32 \
           decoding.beam.flashlight_cfg.beam_threshold = 25.0

    # Lexicon-free decoding
    python eval_beamsearch_ngram_ctc.py ... \
           decoding_strategy="flashlight" \
           decoding.beam.flashlight_cfg.beam_size_token = 32 \
           decoding.beam.flashlight_cfg.beam_threshold = 25.0

PyCTCDecode (``pyctcdecode``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyCTCDecode is a Python library for ASR decoding provided at `https://github.com/kensho-technologies/pyctcdecode <https://github.com/kensho-technologies/pyctcdecode>`_. It is a CPU-based beam search engine that is somewhat efficient for a pure Python library, and supports char and subword models. It requires a character/subword KenLM ARPA / BINARY model to be provided.


It has advanced features, such as word boosting, which can be useful for transcript customization.

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
    python eval_beamsearch_ngram_ctc.py ... \
           decoding_strategy="pyctcdecode" \
           decoding.beam.pyctcdecode_cfg.beam_prune_logp = -10. \
           decoding.beam.pyctcdecode_cfg.token_min_logp = -5. \
           decoding.beam.pyctcdecode_cfg.hotwords=[<List of str words>] \
           decoding.beam.pyctcdecode_cfg.hotword_weight=10.0


Hyperparameter Grid Search
--------------------------

Beam search decoding with N-gram LM has three main hyperparameters: `beam_width`, `beam_alpha`, and `beam_beta`.
The accuracy of the model is dependent on the values of these parameters, specifically, beam_alpha and beam_beta. To perform grid search, you can specify a single value or a list of values for each of these parameters. In this case, it would perform the beam search decoding on all combinations of the three hyperparameters.
For example, the following set of parameters would result in 212=4 beam search decodings:

.. code-block::

    python eval_beamsearch_ngram_ctc.py ... \
                        beam_width=[64,128] \
                        beam_alpha=[1.0] \
                        beam_beta=[1.0,0.5]


Beam Search ngram Decoding for Transducer Models (RNNT and HAT)
===============================================================

You can also find a similar script to evaluate an RNNT/HAT model with beam search decoding and N-gram models at:
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


.. _wfst-ctc-decoding:

WFST CTC decoding
=================
Weighted Finite-State Transducers (WFST) are finite-state machines with input and output symbols on each transition and some weight element of a semiring. WFSTs can act as N-gram LMs in a special type of LM-forced beam search, called WFST decoding.

.. note::

    More precisely, WFST decoding is more of a greedy N-depth search with LM.
    Thus, it is asymptotically worse than conventional beam search decoding algorithms, but faster.

**WARNING**  
At the moment, NeMo supports WFST decoding only for CTC models and word-based LMs.

To run WFST decoding in NeMo, one needs to provide a NeMo ASR model and either an ARPA LM or a WFST LM (advanced). An ARPA LM can be built from source text with KenLM as follows: ``<kenlm_bin_path>/lmplz -o <ngram_length> --arpa <out_arpa_path> --prune <ngram_prune>``.

The script to evaluate an ASR model with WFST decoding and N-gram models can be found at
`scripts/asr_language_modeling/ngram_lm/eval_wfst_decoding_ctc.py
<https://github.com/NVIDIA/NeMo/blob/main/scripts/asr_language_modeling/ngram_lm/eval_wfst_decoding_ctc.py>`__.

This script has a large number of possible argument overrides, therefore it is advised to use ``python eval_wfst_decoding_ctc.py --help`` to see the full list of arguments.

You may evaluate an ASR model as the following:

.. code-block::

    python eval_wfst_decoding_ctc.py nemo_model_file=<path to the .nemo file of the model> \
           input_manifest=<path to the evaluation JSON manifest file> \
           arpa_model_file=<path to the ARPA LM model> \
           decoding_wfst_file=<path to the decoding WFST file> \
           beam_width=[<list of the beam widths, separated with commas>] \
           lm_weight=[<list of the LM weight multipliers, separated with commas>] \
           open_vocabulary_decoding=<whether to use open vocabulary mode for WFST decoding> \
           decoding_mode=<decoding mode, affects output. Usually "nbest"> \
           decoding_search_type=<WFST decoding library. Usually "riva"> \
           preds_output_folder=<optional folder to store the predictions> \
           probs_cache_file=null

.. note::

    Since WFST decoding is LM-forced (the search goes over the WIDEST graph), only word sequences accepted by the WFST can appear in the decoding results.
    To circumvent this restriction, one can pass ``open_vocabulary_decoding=true`` (experimental feature).


Quick start example
-------------------

.. code-block::

    wget -O - https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz | \
    gunzip -c | tr '[:upper:]' '[:lower:]' > 3-gram.pruned.1e-7.arpa && \
    python eval_wfst_decoding_ctc.py nemo_model_file="stt_en_conformer_ctc_small_ls" \
           input_manifest="<data_dir>/Librispeech/test_other.json" \
           arpa_model_file="3-gram.pruned.1e-7.arpa" \
           decoding_wfst_file="3-gram.pruned.1e-7.fst" \
           beam_width=[8] \
           lm_weight=[0.5,0.6,0.7,0.8,0.9]

.. note::

    Building a decoding WFST is a long process, so it is better to provide a ``decoding_wfst_file`` path even if you don't have it.
    This way, the decoding WFST will be buffered to the specified file path and there will be no need to re-build it on the next run.
