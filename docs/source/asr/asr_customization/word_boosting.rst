.. _word_boosting:

****************************************************
Word Boosting
****************************************************

.. _word_boosting_gpupb:

GPU-PB
========================

GPU-PB is a GPU-accelerated Phrase-Boosting method supported for CTC, RNN-T/TDT, and AED (Canary) models based on NGPU-LM infrastructure.
The method supports greedy and beam search decoding, including CUDA graphs mode. GPU-PB is compatible with NGPU-LM at the same decoding run.

GPU-PB is applied only at the decoding step in shallow fusion mode. You do not need to retrain the ASR model.
During greedy or beam search decoding, GPU-PB rescales ASR model scores with a boosting tree at the token level.
The boosting tree is built from a context phrases list, which is provided by the user.

**NOTE**: for ASR models that support capitalization by default (e.g., Canary or parakeet-tdt-0.6b-v2), you need to capitalize all the key phrases in advance (and capitalize the full word for abbreviations).
You can use LLM for this task.

More details about GPU-PB method can be found in the `original paper <https://arxiv.org/abs/2508.07014>`__.

Usage
-----
We support three ways to pass the context phrases into the decoding script:

1. Build a boosting tree for a specific ASR model (step 0.0) and use it for all the decoding evaluation by ``boosting_tree.model_path`` (step 1.1-3.1).
2. Provide a file with context phrases ``boosting_tree.key_phrases_file`` - one phrase per line  (step 1.1-3.1).
3. Provide a python list of context phrases ``boosting_tree.key_phrases_list`` (step 1.1-3.1).

The use of the Phrase-Boosting tree is controlled by ``boosting_tree`` config (``BoostingTreeModelConfig``) for all the models.
For prepared boosting tree use ``boosting_tree.model_path=${PATH_TO_BTREE}``.
We recommend to provide the list of context phrases directly into ``speech_to_text_eval.py`` by ``boosting_tree.key_phrases_file=${KEY_WORDS_LIST}``.

List of the most important parameters:

*  ``strategy`` - The strategy to use for decoding depending on the model type (CTC - greedy_batch or beam_batch; RNN-T/TDT - greedy_batch or malsd_batch; AED - beam).
*  ``model_path``, ``key_phrases_file``, ``key_phrases_list`` - The way to pass the context phrases into the decoding script.
*  ``context_score`` - The score for each arc transition in the context graph (1.0 is recommended).
*  ``depth_scaling`` - The scaling factor for the depth of the context graph (2.0 is recommended for CTC, RNN-T and TDT, 1.0 for Canary).
*  ``boosting_tree_alpha`` - Weight of the GPU-PB boosting tree during shallow fusion decoding (tune it according to your data).

**0.0. [Optional] Build the boosting tree for a specific ASR model:**

.. code-block::

    python scripts/asr_context_biasing/build_gpu_boosting_tree.py \
            asr_model_path=${ASR_NEMO_MODEL_FILE} \
            key_phrases_file=${CONTEXT_BIASING_LIST} \
            save_to=${PATH_TO_SAVE_BTREE} \
            context_score=${CONTEXT_SCORE} \
            depth_scaling=${DEPTH_SCALING} \
            use_triton=True

**1.1. CTC greedy batch decoding:**

.. code-block::

    python examples/asr/speech_to_text_eval.py \
        model_path=${MODEL_NAME} \
        dataset_manifest=${EVAL_MANIFEST} \
        batch_size=${BATCH_SIZE} \
        output_filename=${OUT_MANIFEST} \
        ctc_decoding.strategy="greedy_batch" \
        ctc_decoding.greedy.boosting_tree.key_phrases_file=${KEY_WORDS_LIST} \
        ctc_decoding.greedy.boosting_tree.context_score=1.0 \
        ctc_decoding.greedy.boosting_tree.depth_scaling=2.0 \
        ctc_decoding.greedy.boosting_tree_alpha=${BT_ALPHA}


**1.2. CTC beam batch decoding:**

.. code-block::

    python examples/asr/speech_to_text_eval.py \
        model_path=${MODEL_NAME} \
        dataset_manifest=${EVAL_MANIFEST} \
        batch_size=${BATCH_SIZE} \
        output_filename=${OUT_MANIFEST} \
        ctc_decoding.strategy="beam_batch" \
        ctc_decoding.beam.beam_size=${BEAM_SIZE} \
        ctc_decoding.beam.boosting_tree.key_phrases_file=${KEY_WORDS_LIST} \
        ctc_decoding.beam.boosting_tree.context_score=1.0 \
        ctc_decoding.beam.boosting_tree.depth_scaling=2.0 \
        ctc_decoding.beam.boosting_tree_alpha=${BT_ALPHA}

**2.1. RNN-T/TDT greedy batch decoding:**

.. code-block::

    python examples/asr/speech_to_text_eval.py \
        model_path=${MODEL_NAME} \
        dataset_manifest=${EVAL_MANIFEST} \
        batch_size=${BATCH_SIZE} \
        output_filename=${OUT_MANIFEST} \
        rnnt_decoding.strategy="greedy_batch" \
        rnnt_decoding.greedy.boosting_tree.key_phrases_file=${KEY_WORDS_LIST} \
        rnnt_decoding.greedy.boosting_tree.context_score=1.0 \
        rnnt_decoding.greedy.boosting_tree.depth_scaling=2.0 \
        rnnt_decoding.greedy.boosting_tree_alpha=${BT_ALPHA}

**2.2. RNN-T/TDT beam (malsd_batch) decoding:**

.. code-block::

    python examples/asr/speech_to_text_eval.py \
        model_path=${MODEL_NAME} \
        dataset_manifest=${EVAL_MANIFEST} \
        batch_size=${BATCH_SIZE} \
        output_filename=${OUT_MANIFEST} \
        rnnt_decoding.strategy="malsd_batch" \
        rnnt_decoding.beam.beam_size=${BEAM_SIZE} \
        rnnt_decoding.beam.boosting_tree.key_phrases_file=${KEY_WORDS_LIST} \
        rnnt_decoding.beam.boosting_tree.context_score=1.0 \
        rnnt_decoding.beam.boosting_tree.depth_scaling=2.0 \
        rnnt_decoding.beam.boosting_tree_alpha=${BT_ALPHA}

**3.1. AED (Canary) greedy (beam_size=1) or beam (beam_size>1) decoding:**

.. code-block::

    python examples/asr/speech_to_text_eval.py \
        model_path=${MODEL_NAME} \
        dataset_manifest=${EVAL_MANIFEST} \
        batch_size=${BATCH_SIZE} \
        output_filename=${OUT_MANIFEST} \
        multitask_decoding.strategy="beam" \
        multitask_decoding.beam.beam_size=${BEAM_SIZE} \
        multitask_decoding.beam.boosting_tree.key_phrases_file=${CONTEXT_BIASING_LIST} \
        multitask_decoding.beam.boosting_tree.context_score=1.0 \
        multitask_decoding.beam.boosting_tree.depth_scaling=1.0 \
        multitask_decoding.beam.boosting_tree_alpha=${BT_ALPHA} \
        gt_lang_attr_name="target_lang" \
        gt_text_attr_name="text"

Results evaluation
------------------

You can compute the F-score for the list of context phrases directly from the decoding manifest.

.. code-block::

    python scripts/asr_context_biasing/compute_key_words_fscore.py \
            --input_manifest=${DECODING_MANIFEST} \
            --key_words_file=${CONTEXT_PHRASES_LIST}


.. _word_boosting_flashlight:

Flashlight-based Word Boosting
==============================


The Flashlight decoder supports word boosting during CTC decoding using a KenLM binary and corresponding lexicon. Word boosting only works in lexicon-decoding mode and does not function in lexicon-free mode. It allows you to bias the decoder for certain words by manually increasing or decreasing the probability of emitting specific words. This can be very helpful if you have uncommon or industry-specific terms that you want to ensure are transcribed correctly.

For more information, go to `word boosting <https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-customizing.html#word-boosting>`__

To use word boosting in NeMo, create a simple tab-separated text file. Each line should contain a word to be boosted, followed by a tab, and then the boosted score for that word.

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

You can then pass this file to your Flashlight config object during decoding:

.. code-block::

    # Lexicon-based decoding
    python eval_beamsearch_ngram_ctc.py ... \
           decoding_strategy="flashlight" \
           decoding.beam.flashlight_cfg.lexicon_path='/path/to/lexicon.lexicon' \
           decoding.beam.flashlight_cfg.boost_path='/path/to/my_boost_file.boost' \
           decoding.beam.flashlight_cfg.beam_size_token = 32 \
           decoding.beam.flashlight_cfg.beam_threshold = 25.0

.. _word_boosting_ctcws:

CTC-WS: Context-biasing (Word Boosting) without External LM
===========================================================

NeMo toolkit supports a fast context-biasing method for CTC and Transducer (RNN-T) ASR models with CTC-based Word Spotter.
The method involves decoding CTC log probabilities with a context graph built for words and phrases from the context-biasing list.
The spotted context-biasing candidates (with their scores and time intervals) are compared by scores with words from the greedy CTC decoding results to improve recognition accuracy and prevent false accepts of context-biasing.

A Hybrid Transducer-CTC model (a shared encoder trained together with CTC and Transducer output heads) enables the use of the CTC-WS method for the Transducer model.
Context-biasing candidates obtained by CTC-WS are also filtered by the scores with greedy CTC predictions and then merged with greedy Transducer results.

Scheme of the CTC-WS method:

.. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.22.0/asset-post-v1.22.0-ctcws_scheme_1.png
    :align: center
    :alt: CTC-WS scheme
    :width: 80%

High-level overview of the context-biasing words replacement with CTC-WS method:

.. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.22.0/asset-post-v1.22.0-ctcws_scheme_2.png
    :align: center
    :alt: CTC-WS high level overview
    :width: 80%

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

*  ``beam_threshold`` - threshold for CTC-WS beam pruning.
*  ``context_score`` - per token weight for context biasing.
*  ``ctc_ali_token_weight`` - per token weight for CTC alignment (prevents false acceptances of context-biasing words).

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
