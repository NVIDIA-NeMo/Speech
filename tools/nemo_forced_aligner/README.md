# NeMo Forced Aligner (NFA)

A tool for doing Forced Alignment using Viterbi decoding of NeMo CTC-based models.

## Usage example 

``` bash
python <path_to_NeMo>/tools/nemo_forced_aligner/align.py \
        pretrained_name="stt_en_citrinet_1024_gamma_0_25" \
        model_downsample_factor=8 \
        manifest_filepath=<path to manifest of utterances you want to align> \
        output_dir=<path to where your ctm files will be saved>
```

## How do I use NeMo Forced Aligner?
To use NFA, all you need to provide is a correct NeMo manifest (with `"audio_filepath"` and `"text"` fields).

Call the `align.py` script, specifying the parameters as follows:

* `pretrained_name`: string specifying the name of a CTC NeMo ASR model which will be automatically downloaded from NGC and used for generating the log-probs which we will use to do alignment. Any Quartznet, Citrinet, Conformer CTC model should work, in any language (only English has been tested so far). If `model_path` is specified, `pretrained_name` must not be specified.
>Note: NFA can only use CTC models (not Transducer models) at the moment. If you want to transcribe a long audio file (longer than ~5-10 mins), do not use Conformer CTC model as that will likely give Out Of Memory errors.

* `model_path`: string specifying the local filepath to a CTC NeMo ASR model which will be used to generate the log-probs which we will use to do alignment. If `pretrained_name` is specified, `model_path` must not be specified.
>Note: NFA can only use CTC models (not Transducer models) at the moment. If you want to transcribe a long audio file (longer than ~5-10 mins), do not use Conformer CTC model as that will likely give Out Of Memory errors.

* `model_downsample_factor`: the downsample factor of the ASR model. It should be 2 if your model is QuartzNet, 4 if it is Conformer CTC, 8 if it is Citrinet.

* `manifest_filepath`: The path to the manifest of the data you want to align, containing `'audio_filepath'` and `'text'` fields. The audio filepaths need to be absolute paths.

* `output_dir`: The folder where to save CTM files containing the generated alignments and new JSON manifest containing paths to those CTM files. There will be one CTM file per utterance (ie one CTM file per line in the manifest). The files will be called `<output_dir>/{tokens,words,additional_segments}/<utt_id>.ctm` and each line in each file will start with `<utt_id>`. By default, `utt_id` will be the stem of the audio_filepath. This can be changed by overriding `audio_filepath_parts_in_utt_id`. The new JSON manifest will be at `<output_dir>/<original manifest file name>_with_ctm_paths.json`.

* **[OPTIONAL]** `align_using_pred_text`: if True, will transcribe the audio using the ASR model (specified by `pretrained_name` or `model_path`) and then use that transcription as the 'ground truth' for the forced alignment. The `"pred_text"` will be saved in the output JSON manifest at `<output_dir>/{original manifest name}_with_ctm_paths.json`. To avoid over-writing other transcribed texts, if there are already `"pred_text"` entries in the original manifest, the program will exit without attempting to generate alignments.  (Default: False). 

* **[OPTIONAL]** `transcribe_device`: The device that will be used for generating log-probs (i.e. transcribing). If None, NFA will set it to 'cuda' if it is available (otherwise will set it to 'cpu'). If specified `transcribe_device` needs to be a string that can be input to the `torch.device()` method. (Default: `None`).

* **[OPTIONAL]** `viterbi_device`: The device that will be used for doing Viterbi decoding. If None, NFA will set it to 'cuda' if it is available (otherwise will set it to 'cpu'). If specified `transcribe_device` needs to be a string that can be input to the `torch.device()` method.(Default: `None`).

* **[OPTIONAL]** `batch_size`: The batch_size that will be used for generating log-probs and doing Viterbi decoding. (Default: 1).

* **[OPTIONAL]** `additional_ctm_grouping_separator`: the string used to separate CTM segments if you want to obtain CTM files at a level that is not the token level or the word level. NFA will always produce token-level and word-level CTM files in: `<output_dir>/tokens/<utt_id>.ctm` and `<output_dir>/words/<utt_id>.ctm`. If `additional_ctm_grouping_separator` is specified, an additional folder `<output_dir>/{tokens/words/additional_segments}/<utt_id>.ctm` will be created containing CTMs for `addtional_ctm_grouping_separator`-separated segments. (Default: `None`. Cannot be empty string or space (" "), as space-separated word-level CTMs will always be saved in `<output_dir>/words/<utt_id>.ctm`.)
> Note: the `additional_ctm_grouping_separator` will be removed from the ground truth text and all the output CTMs, ie it is treated as a marker which is not part of the ground truth. The separator will essentially be treated as a space, and any additional spaces around it will be amalgamated into one, i.e. if `additional_ctm_grouping_separator="|"`, the following texts will be treated equivalently: `“abc|def”`, `“abc |def”`, `“abc| def”`, `“abc | def"`.

* **[OPTIONAL]** `remove_blank_tokens_from_ctm`: a boolean denoting whether to remove <blank> tokens from token-level output CTMs. (Default: False). 

* **[OPTIONAL]** `audio_filepath_parts_in_utt_id`: This specifies how many of the 'parts' of the audio_filepath we will use (starting from the final part of the audio_filepath) to determine the utt_id that will be used in the CTM files. (Default: 1, i.e. utt_id will be the stem of the basename of audio_filepath). Note also that any spaces that are present in the audio_filepath will be replaced with dashes, so as not to change the number of space-separated elements in the CTM files.

* **[OPTIONAL]** `minimum_timestamp_duration`: a float indicating a minimum duration (in seconds) for timestamps in the CTM. If any line in the CTM has a duration lower than the `minimum_timestamp_duration`, it will be enlarged from the middle outwards until it meets the minimum_timestamp_duration, or reaches the beginning or end of the audio file. Note that this may cause timestamps to overlap. (Default: 0, i.e. no modifications to predicted duration).

* **[OPTIONAL]** `use_buffered_chunked_streaming`: a flag to indicate whether to do buffered chunk streaming. Notice only CTC models (e.g., stt_en_citrinet_1024_gamma_0_25)with `per_feature` preprocessor are supported. The below two params are needed if this option set to `True`.

* **[OPTIONAL]** `chunk_len_in_secs`: the chunk size for buffered chunked streaming inference. Default is 1.6 seconds.

* **[OPTIONAL]** `total_buffer_in_secs`: the buffer size for buffered chunked streaming inference. Default is 4.0 seconds.

# Input manifest file format
By default, NFA needs to be provided with a 'manifest' file where each line specifies the absolute "audio_filepath" and "text" of each utterance that you wish to produce alignments for, like the format below:
```json
{"audio_filepath": "/absolute/path/to/audio.wav", "text": "the transcription of the utterance"}
```

You can omit the `"text"` field from the manifest if you specify `align_using_pred_text=true`. In that case, any `"text"` fields in the manifest will be ignored: the ASR model at `pretrained_name` or `model_path` will be used to transcribe the audio and obtain `"pred_text"`, which will be used as the 'ground truth' for the forced alignment process. The `"pred_text"` will also be saved in the output manifest JSON file at `<output_dir>/<original manifest file name>_with_ctm_paths.json`. To remove the possibility of overwriting `"pred_text"`, NFA will raise an error if `align_using_pred_text=true` and there are existing `"pred_text"` fields in the original manifest.

> Note: NFA does not require `"duration"` fields in the manifest, and can align long audio files without running out of memory. Depending on your machine specs, you can align audios up to 5-10 minutes on Conformer CTC models, up to around 1.5 hours for QuartzNet models, and up to several hours for Citrinet models. NFA will also produce better alignments the more accurate the ground-truth `"text"` is.


# Output CTM file format
For each utterance specified in a line of `manifest_filepath`, several CTM files will be generated:
* a CTM file containing token-level alignments at `<output_dir>/tokens/<utt_id>.ctm`,
* a CTM file containing word-level alignments at `<output_dir>/words/<utt_id>.ctm`,
* if `additional_ctm_grouping_separator` is specified, there will also be a CTM file containing those segments at `output_dir/additional_segments`.
Each CTM file will contain lines of the format:
`<utt_id> 1 <start time in seconds> <duration in seconds> <text, ie token/word/segment>`.
Note the second item in the line (the 'channel ID', which is required by the CTM file format) is always 1, as NFA operates on single channel audio.

# Output JSON manifest file format
A new manifest file will be saved at `<output_dir>/<original manifest file name>_with_ctm_paths.json`. It will contain the same fields as the original manifest, and additionally:
* `"token_level_ctm_filepath"`
* `"word_level_ctm_filepath"`
* `"additonal_segment_level_ctm_filepath"` (if `additional_ctm_grouping_separator` is specified)
* `"pred_text"` (if `align_using_pred_text=true`)


# How do I evaluate the alignment accuracy?
Ideally you would have some 'true' CTM files to compare with your generated CTM files. With these you could obtain metrics such as the mean (absolute) errors between predicted starts/ends and the 'true' starts/ends of the segments.

Alternatively (or additionally), you can visualize the quality of alignments using tools such as Gecko, which can play your audio file and display the predicted alignments at the same time. The Gecko tool requires you to upload an audio file and at least one CTM file. The Gecko tool can be accessed here: https://gong-io.github.io/gecko/. More information about the Gecko tool can be found on its Github page here: https://github.com/gong-io/gecko. 

**Note**: the following may help improve your experience viewing the CTMs in Gecko:
* setting `minimum_timestamp_duration` to a larger number, as Gecko may not display some tokens/words/segments properly if their timestamps are too short.
* setting `remove_blank_tokens_from_ctm=true` if you are analyzing token-level CTMs, as it will make the Gecko visualization less cluttered.
