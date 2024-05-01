NeMo Forced Aligner (NFA)
=========================

NFA is hosted here: https://github.com/NVIDIA/NeMo/tree/main/tools/nemo_forced_aligner.


NFA is a tool for generating token-, word- and segment-level timestamps of speech in audio using NeMo's CTC-based Automatic Speech Recognition models. 
You can provide your own reference text, or use ASR-generated transcription. 
You can use NeMo's ASR Model checkpoints out of the box in :ref:`14+ languages <asr-checkpoint-list-by-language>`, or train your own model.
NFA can be used on long audio files of 1+ hours duration (subject to your hardware and the ASR model used).

Demos & Tutorials
-----------------

* HuggingFace Space `demo <https://huggingface.co/spaces/erastorgueva-nv/NeMo-Forced-Aligner>`__ to quickly try out NFA in various languages.
* NFA "how-to" notebook `tutorial <https://nvidia.github.io/NeMo/blogs/2023/2023-08-forced-alignment/>`__.
* "How forced alignment works" NeMo blog `tutorial <https://colab.research.google.com/github/NVIDIA/NeMo/blob/main/tutorials/tools/NeMo_Forced_Aligner_Tutorial.ipynb>`__.

Quickstart
----------

1. Install `NeMo <https://github.com/NVIDIA/NeMo#installation>`__.
2. Prepare a NeMo-style manifest containing the paths of audio files you would like to proces, and (optionally) their text.
3. Run NFA's ``align.py`` script with the desired config, e.g.:

.. code-block::

    python <path_to_NeMo>/tools/nemo_forced_aligner/align.py \
	    pretrained_name="stt_en_fastconformer_hybrid_large_pc" \
	    manifest_filepath=<path to manifest of utterances you want to align> \
	    output_dir=<path to where your output files will be saved>

.. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.20.0/nfa_run.png

How do I use NeMo Forced Aligner?
---------------------------------

To use NFA, all you need to provide is a correct NeMo manifest (with ``"audio_filepath"`` and, optionally, ``"text"`` fields).

Call the ``align.py`` script, specifying the parameters as follows:

* ``pretrained_name``: string specifying the name of a CTC NeMo ASR model which will be automatically downloaded from NGC and used for generating the log-probs which we will use to do alignment. Any Quartznet, Citrinet, Conformer CTC model should work, in any language (only English has been tested so far). If ``model_path`` is specified, ``pretrained_name`` must not be specified.

	Note: Currently NFA can only use CTC models, or Hybrid CTC-Transducer models (in CTC mode). Pure Transducer models cannot be used.

* ``model_path``: string specifying the local filepath to a CTC NeMo ASR model which will be used to generate the log-probs which we will use to do alignment. If ``pretrained_name`` is specified, ``model_path`` must not be specified.

	.. note:: Currently NFA can only use CTC models, or Hybrid CTC-Transducer models (in CTC mode). Pure Transducer models cannot be used.

* ``manifest_filepath``: The path to the manifest of the data you want to align, containing ``'audio_filepath'`` and ``'text'`` fields. The audio filepaths need to be absolute paths.

* ``output_dir``: The folder where to save the output files (e.g. CTM, ASS) containing the generated alignments and new JSON manifest containing paths to those CTM/ASS files. The CTM file will be called ``<output_dir>/ctm/{tokens,words,segments}/<utt_id>.ctm`` and each line in each file will start with ``<utt_id>``. By default, ``utt_id`` will be the stem of the audio_filepath. This can be changed by overriding ``audio_filepath_parts_in_utt_id``. The new JSON manifest will be at ``<output_dir>/<original manifest file name>_with_ctm_paths.json``. The ASS files will be at ``<output_dir>/ass/{tokens,words}/<utt_id>.ass``. You can adjust which files should be saved by adjusting the parameter ``save_output_file_formats``. 

Optional parameters:
^^^^^^^^^^^^^^^^^^^^

* ``align_using_pred_text``: if True, will transcribe the audio using the ASR model (specified by ``pretrained_name`` or ``model_path``) and then use that transcription as the reference text for the forced alignment. The ``"pred_text"`` will be saved in the output JSON manifest at ``<output_dir>/{original manifest name}_with_ctm_paths.json``. To avoid over-writing other transcribed texts, if there are already ``"pred_text"`` entries in the original manifest, the program will exit without attempting to generate alignments.  (Default: False). 

* ``transcribe_device``: The device that will be used for generating log-probs (i.e. transcribing). If None, NFA will set it to 'cuda' if it is available (otherwise will set it to 'cpu'). If specified ``transcribe_device`` needs to be a string that can be input to the ``torch.device()`` method. (Default: ``None``).

* ``viterbi_device``: The device that will be used for doing Viterbi decoding. If None, NFA will set it to 'cuda' if it is available (otherwise will set it to 'cpu'). If specified ``transcribe_device`` needs to be a string that can be input to the ``torch.device()`` method.(Default: ``None``).

* ``batch_size``: The batch_size that will be used for generating log-probs and doing Viterbi decoding. (Default: 1).

* ``use_local_attention``: boolean flag specifying whether to try to use local attention for the ASR Model (will only work if the ASR Model is a Conformer model). If local attention is used, we will set the local attention context size to [64,64].

* ``additional_segment_grouping_separator``: an optional string used to separate the text into smaller segments. If this is not specified, then the whole text will be treated as a single segment. (Default: ``None``. Cannot be empty string or space (" "), as NFA will automatically produce word-level timestamps for substrings separated by spaces).

	.. note:: the ``additional_segment_grouping_separator`` will be removed from the reference text and all the output files, ie it is treated as a marker which is not part of the reference text. The separator will essentially be treated as a space, and any additional spaces around it will be amalgamated into one, i.e. if ``additional_segment_grouping_separator="|"``, the following texts will be treated equivalently: ``“abc|def”``, ``“abc |def”``, ``“abc| def”``, ``“abc | def"``.

* ``remove_blank_tokens_from_ctm``: a boolean denoting whether to remove <blank> tokens from token-level output CTMs. (Default: False). 

* ``audio_filepath_parts_in_utt_id``: This specifies how many of the 'parts' of the audio_filepath we will use (starting from the final part of the audio_filepath) to determine the utt_id that will be used in the CTM files. (Default: 1, i.e. utt_id will be the stem of the basename of audio_filepath). Note also that any spaces that are present in the audio_filepath will be replaced with dashes, so as not to change the number of space-separated elements in the CTM files.

* ``minimum_timestamp_duration``: a float indicating a minimum duration (in seconds) for timestamps in the CTM. If any line in the CTM has a duration lower than the ``minimum_timestamp_duration``, it will be enlarged from the middle outwards until it meets the minimum_timestamp_duration, or reaches the beginning or end of the audio file. Note that this may cause timestamps to overlap. (Default: 0, i.e. no modifications to predicted duration).

* ``use_buffered_chunked_streaming``: a flag to indicate whether to do buffered chunk streaming. Notice only CTC models (e.g., stt_en_citrinet_1024_gamma_0_25)with ``per_feature`` preprocessor are supported. The below two params are needed if this option set to ``True``.

* ``chunk_len_in_secs``: the chunk size for buffered chunked streaming inference. Default is 1.6 seconds.

* ``total_buffer_in_secs``: the buffer size for buffered chunked streaming inference. Default is 4.0 seconds.

* ``simulate_cache_aware_streaming``: a flag to indicate whether to use cache aware streaming to do get the logits for alignment. Default: ``False``.

* ``save_output_file_formats``: list of file formats to use for saving the output. Default: ``["ctm", "ass"]`` (these are all the available ones currently).

* ``ctm_file_config``: ``CTMFileConfig`` to specify the configuration of the output CTM files.

* ``ass_file_config``: ``ASSFileConfig`` to specify the configuration of the output ASS files.

Input manifest file format
--------------------------
By default, NFA needs to be provided with a 'manifest' file where each line specifies the absolute "audio_filepath" and "text" of each utterance that you wish to produce alignments for, like the format below:

.. code-block::

    {"audio_filepath": "/absolute/path/to/audio.wav", "text": "the transcription of the utterance"}

You can omit the ``"text"`` field from the manifest if you specify ``align_using_pred_text=true``. In that case, any ``"text"`` fields in the manifest will be ignored: the ASR model at ``pretrained_name`` or ``model_path`` will be used to transcribe the audio and obtain ``"pred_text"``, which will be used as the reference text for the forced alignment process. The ``"pred_text"`` will also be saved in the output manifest JSON file at ``<output_dir>/<original manifest file name>_with_output_file_paths.json``. To remove the possibility of overwriting ``"pred_text"``, NFA will raise an error if ``align_using_pred_text=true`` and there are existing ``"pred_text"`` fields in the original manifest.

	.. note:: NFA does not require ``"duration"`` fields in the manifest, and can align long audio files without running out of memory. The duration of audio file you can align will depend on the amount of memory on your machine. NFA will also produce better alignments the more accurate the reference text in ``"text"`` is.


Output CTM file format
----------------------

For each utterance specified in a line of ``manifest_filepath``, several CTM files will be generated:

* a CTM file containing token-level alignments at ``<output_dir>/ctm/tokens/<utt_id>.ctm``,
* a CTM file containing word-level alignments at ``<output_dir>/ctm/words/<utt_id>.ctm``,
* a CTM file containing segment-level alignments at ``<output_dir>/ctm/segments/<utt_id>.ctm``. If ``additional_segment_grouping_separator`` is specified, the segments will be parts of the text separated by ``additonal_segment_grouping_separator``. If it is not specified, the entire text will be treated as a single segment.

Each CTM file will contain lines of the format:
``<utt_id> 1 <start time in seconds> <duration in seconds> <text, ie token/word/segment>``.
Note the second item in the line (the 'channel ID', which is required by the CTM file format) is always 1, as NFA operates on single channel audio.

``CTMFileConfig`` parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``CTMFileConfig`` (which is passed into the main NFA config) has the following parameters:

* ``remove_blank_tokens``: bool (default ``False``) to specify if the token-level CTM files should have the timestamps of the blank tokens removed.
* ``minimum_timestamp_duration``: float (default ``0``) to specify the minimum duration that will be applied to all timestamps. If any line in the CTM has a duration lower than this, it will be enlarged from the middle outwards until it meets the ``minimum_timestamp_duration``, or reaches the beginning or end of the audio file. Note that using a non-zero value may cause timestamps to overlap.

Output ASS file format
----------------------

NFA will produce the following ASS files, which you can use to generate subtitle videos:

* ASS files with token-level highlighting will be at ``<output_dir>/ass/tokens/<utt_id>.ass,``
* ASS files with word-level highlighting will be at ``<output_dir>/ass/words/<utt_id>.ass``.

All words belonging to the same segment 'segments' will appear at the same time in the subtitles generated with the ASS files. If you find that your segments are not the right size, you can use set ``ass_file_config.resegment_text_to_fill_space=true`` and specify some number of ``ass_file_config.max_lines_per_segment``.

``ASSFileConfig`` parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ASSFileConfig`` (which is passed into the main NFA config) has the following parameters:

* ``fontsize``: int (default value ``20``) which will be the fontsize of the text
* ``vertical_alignment``: string (default value ``center``) to specify the vertical alignment of the text. Can be one of ``center``, ``top``, ``bottom``.
* ``resegment_text_to_fill_space``: bool (default value ``False``). If ``True``, the text will be resegmented such that each segment will not take up more than (approximately) ``max_lines_per_segment`` when the ASS file is applied to a video.
* ``max_lines_per_segment``: int (defaulst value ``2``) which specifies the number of lines per segment to display. This parameter is only used if ``resegment_text_to_fill_space`` is ``True``.
* ``text_already_spoken_rgb``: List of 3 ints (default value is [49, 46, 61], which makes a dark gray). The RGB values of the color that will be used to highlight text that has already been spoken.
* ``text_being_spoken_rgb``: List of 3 ints (default value is [57, 171, 9] which makes a dark green). The RGB values of the color that will be used to highlight text that is being spoken.
* ``text_not_yet_spoken_rgb``: List of 3 ints (default value is [194, 193, 199] which makes a dark green). The RGB values of the color that will be used to highlight text that has not yet been spoken.

Output JSON manifest file format
--------------------------------

A new manifest file will be saved at ``<output_dir>/<original manifest file name>_with_output_file_paths.json``. It will contain the same fields as the original manifest, and additionally:

* ``"token_level_ctm_filepath"`` (if ``save_output_file_formats`` contains ``ctm``)
* ``"word_level_ctm_filepath"`` (if ``save_output_file_formats`` contains ``ctm``)
* ``"segment_level_ctm_filepath"`` (if ``save_output_file_formats`` contains ``ctm``)
* ``"token_level_ass_filepath"`` (if ``save_output_file_formats`` contains ``ass``)
* ``"word_level_ass_filepath"`` (if ``save_output_file_formats`` contains ``ass``)
* ``"pred_text"`` (if ``align_using_pred_text=true``)


How do I evaluate the alignment accuracy?
-----------------------------------------

Ideally you would have some 'true' CTM files to compare with your generated CTM files. With these you could obtain metrics such as the mean (absolute) errors between predicted starts/ends and the 'true' starts/ends of the segments.

Alternatively (or additionally), you can visualize the quality of alignments using tools such as Gecko, which can play your audio file and display the predicted alignments at the same time. The Gecko tool requires you to upload an audio file and at least one CTM file. The Gecko tool can be accessed here: https://gong-io.github.io/gecko/. More information about the Gecko tool can be found on its Github page here: https://github.com/gong-io/gecko. 

.. note:: 
	The following may help improve your experience viewing the CTMs in Gecko:

	* setting ``minimum_timestamp_duration`` to a larger number, as Gecko may not display some tokens/words/segments properly if their timestamps are too short.
	* setting ``remove_blank_tokens_from_ctm=true`` if you are analyzing token-level CTMs, as it will make the Gecko visualization less cluttered.

