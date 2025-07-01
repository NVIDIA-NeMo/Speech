.. _word_boosting:

*************
Word Boosting
*************
.. _word_boosting_flashlight:

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

****************************************************
Context-biasing (Word Boosting) without External LM
****************************************************

NeMo toolkit supports a fast context-biasing method for CTC and Transducer (RNN-T) ASR models with CTC-based Word Spotter.
The method involves decoding CTC log probabilities with a context graph built for words and phrases from the context-biasing list.
The spotted context-biasing candidates (with their scores and time intervals) are compared by scores with words from the greedy CTC decoding results to improve recognition accuracy and pretend false accepts of context-biasing.

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
