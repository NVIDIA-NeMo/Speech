.. _lhotse-dataloading:

==================
Lhotse Dataloading
==================

NeMo supports using `Lhotse`_, a speech data handling library, as a dataloading option. The key features of Lhotse used in NeMo are:

* Dynamic batch sizes
    Lhotse samples mini-batches to satisfy the constraint of total speech duration in a mini-batch (``batch_duration``),
    rather than a specific number of examples (i.e., batch size).
* Dynamic bucketing
    Instead of statically pre-bucketing the data, Lhotse allocates training examples to buckets dynamically.
    This allows more rapid experimentation with bucketing settings (number of buckets, specific placement of bucket duration bins)
    to minimize the amount of padding and accelerate training.
* Quadratic duration penalty
    Adding a quadratic penalty to an utterance's duration allows to sample mini-batches so that the
    GPU utilization is more consistent across big batches of short utterances and small batches of long utterances when using
    models with quadratic time/memory complexity (such as transformer).
* Dynamic weighted data source multiplexing
    An approach to combining diverse data sources (e.g. multiple domains, languages, tasks)
    where each data source is treated as a separate stream with its own sampling probability. The resulting data stream is a
    multiplexer that samples from each sub-stream. This approach ensures that the distribution of different sources is approximately
    constant in time (i.e., stationary); in fact, each mini-batch will have roughly the same ratio of data coming from each source.
    Since the multiplexing is done dynamically, it is very easy to tune the sampling weights.

Lhotse dataloading supports the following types of inputs:

* NeMo manifests
    Regular NeMo JSON manifests.
* NeMo tarred data
    Tarred NeMo JSON manifests + audio tar files; we also support combination of multiple NeMo
    tarred data sources (e.g., multiple buckets of NeMo data or multiple datasets) via dynamic multiplexing.

    We support using a subset of Tarred NeMo JSON manifests along with audio tar files without disrupting the alignment between the tarred files and their corresponding manifests.
    This feature is essential because large datasets often consist of numerous tar files and multiple versions of Tarred NeMo JSON manifest subsets, which may contain only a portion of the audio files due to filtering for various reasons.
    To skip specific entries in the manifests without repeatedly copying and retarring audio files, the entries must include a ``_skipme`` key. This key should be set to ``True``, ``1``, or a reason for skipping (e.g., ``low character-rate``).

* Lhotse CutSet manifests
    Regular Lhotse CutSet manifests (typically gzipped JSONL).
    See `Lhotse Cuts documentation`_ to learn more about Lhotse data formats.
* Lhotse Shar data
    Lhotse Shar is a data format that also uses tar files for sequential data loading,
    but is designed to be modular (i.e., easily extensible with new data sources and with new feature fields).
    More details can be found here: |tutorial_shar|

.. caution:: As of now, Lhotse is mainly supported in most ASR model configurations. We aim to gradually extend this support to other speech tasks.

.. _Lhotse: https://github.com/lhotse-speech/lhotse
.. _Lhotse Cuts documentation: https://lhotse.readthedocs.io/en/latest/cuts.html
.. |tutorial_shar| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/lhotse-speech/lhotse/blob/master/examples/04-lhotse-shar.ipynb

Enabling Lhotse via configuration
----------------------------------

.. note:: Using Lhotse with tarred datasets will make the dataloader infinite, ditching the notion of an "epoch". "Epoch" may still be logged in W&B/TensorBoard, but it will correspond to the number of executed training loops between validation loops.

Start with an existing NeMo experiment YAML configuration. Typically, you'll only need to add a few options to enable Lhotse.
These options are::

    # NeMo generic dataloading arguments
    model.train_ds.manifest_filepath=...
    model.train_ds.tarred_audio_filepaths=...   # for tarred datasets only
    model.train_ds.num_workers=4
    model.train_ds.min_duration=0.3             # optional
    model.train_ds.max_duration=30.0            # optional
    model.train_ds.shuffle=true                 # optional

    # Lhotse dataloading related arguments
    ++model.train_ds.use_lhotse=True
    ++model.train_ds.batch_duration=1100
    ++model.train_ds.quadratic_duration=30
    ++model.train_ds.num_buckets=30
    ++model.train_ds.num_cuts_for_bins_estimate=10000
    ++model.train_ds.bucket_buffer_size=10000
    ++model.train_ds.shuffle_buffer_size=10000

    # PyTorch Lightning related arguments
    ++trainer.use_distributed_sampler=false
    ++trainer.limit_train_batches=1000
    trainer.val_check_interval=1000
    trainer.max_steps=300000

.. note:: The default values above are a reasonable starting point for a hybrid RNN-T + CTC ASR model on a 32GB GPU with a data distribution dominated by 15s long utterances.

Let's briefly go over each of the Lhotse dataloading arguments:

* ``use_lhotse`` enables Lhotse dataloading
* ``batch_duration`` is the total max duration of utterances in a mini-batch and controls the batch size; the more shorter utterances, the bigger the batch size, and vice versa.
* ``quadratic_duration`` adds a quadratically growing penalty for long utterances; useful in bucketing and transformer type of models. The value set here means utterances this long will count as if with a doubled duration.
* ``num_buckets`` is the number of buckets in the bucketing sampler. Bigger value means less padding but also less randomization.
* ``num_cuts_for_bins_estimate`` is the number of utterance we will sample before the start of the training to estimate the duration bins for buckets. Larger number results in a more accurate estimatation but also a bigger lag before starting the training.
* ``bucket_buffer_size`` is the number of utterances (data and metadata) we will hold in memory to be distributed between buckets. With bigger ``batch_duration``, this number may need to be increased for dynamic bucketing sampler to work properly (typically it will emit a warning if this is too low).
* ``shuffle_buffer_size`` is an extra number of utterances we will hold in memory to perform approximate shuffling (via reservoir-like sampling). Bigger number means more memory usage but also better randomness.

The PyTorch Lightning ``trainer`` related arguments:

* ``use_distributed_sampler=false`` is required because Lhotse has its own handling of distributed sampling.
* ``val_check_interval``/``limit_train_batches``
    These are required for dataloaders with tarred/Shar datasets
    because Lhotse makes the dataloader infinite, so we'd never go past epoch 0. This approach guarantees
    we will never hang the training because the dataloader in some node has less mini-batches than the others
    in some epochs. The value provided here will be the effective length of each "pseudo-epoch" after which we'll
    trigger the validation loop.
* ``max_steps`` is the total number of steps we expect to be training for. It is required for the same reason as ``limit_train_batches``; since we'd never go past epoch 0, the training would have never finished.

Some other Lhotse related arguments we support:

* ``cuts_path`` can be provided to read data from a Lhotse CutSet manifest instead of a NeMo manifest.
    Specifying this option will result in ``manifest_filepaths`` and ``tarred_audio_filepaths`` being ignored.
* ``shar_path``
    Can be provided to read data from a Lhotse Shar manifest instead of a NeMo manifest.
    Specifying this option will result in ``manifest_filepaths`` and ``tarred_audio_filepaths`` being ignored.
    This argument can be a string (single Shar directory), a list of strings (Shar directories),
    or a list of 2-item lists, where the first item is a Shar directory path, and the other is a sampling weight.
    The user can also provide a dict mapping Lhotse Shar fields to a list of shard paths with data for that field.
    For details about Lhotse Shar format, see: |tutorial_shar|
* ``bucket_duration_bins``
    Duration bins are a list of float values (seconds) that when provided, will skip the initial bucket bin estimation
    and save some time. It has to have a length of ``num_buckets - 1``. An optimal value can be obtained by running CLI:
    ``lhotse cut estimate-bucket-bins -b $num_buckets my-cuts.jsonl.gz``
* ``use_bucketing`` is a boolean which indicates if we want to enable/disable dynamic bucketing. By defalt it's enabled.
* ``text_field`` is the name of the key in the JSON (NeMo) manifest from which we should be reading text (default="text").
* ``lang_field`` is the name of the key in the JSON (NeMo) manifest from which we should be reading language tag (default="lang"). This is useful when working e.g. with ``AggregateTokenizer``.
* ``batch_size``
    Limits the number of examples in a mini-batch to this number, when combined with ``batch_duration``.
    When ``batch_duration`` is not set, it acts as a static batch size.
* ``seed`` sets a random seed for the shuffle buffer.

The full and always up-to-date list of supported options can be found in ``LhotseDataLoadingConfig`` class.

.. _asr-dataset-config-format:

Extended multi-dataset configuration format
--------------------------------------------

Combining a large number of datasets and defining weights for them can be tricky.
We offer an extended configuration format that allows you to explicitly define datasets,
dataset groups, and their weights either inline in the experiment configuration,
or as a path to a separate YAML file.

In addition to the features above, this format introduces a special ``tags`` dict-like field.
The keys and values in ``tags`` are automatically attached to every sampled example, which
is very useful when combining multiple datasets with different properties.
The dataset class which converts these examples to tensors can partition the mini-batch and apply
different processing to each group.
For example, you may want to construct different prompts for the model using metadata in ``tags``.

.. note:: When fine-tuning a model that was trained with ``input_cfg`` option, typically you'd only need
    to override the following options: ``input_cfg=null`` and ``manifest_filepath=path/to/manifest.json``.

Example 1. Combine two datasets with equal weights and attach custom metadata in ``tags`` to each cut:

.. code-block:: yaml

    input_cfg:
      - type: nemo_tarred
        manifest_filepath: /path/to/manifest__OP_0..512_CL_.json
        tarred_audio_filepath: /path/to/tarred_audio/audio__OP_0..512_CL_.tar
        weight: 0.4
        tags:
          lang: en
          pnc: no
      - type: nemo_tarred
        manifest_filepath: /path/to/other/manifest__OP_0..512_CL_.json
        tarred_audio_filepath: /path/to/other/tarred_audio/audio__OP_0..512_CL_.tar
        weight: 0.6
        tags:
          lang: pl
          pnc: yes

Example 2. Combine multiple (4) datasets, corresponding to different tasks (ASR, AST).
Each task gets its own group and its own weight.
Then within each task, each dataset get its own within-group weight as well.
The final weight is the product of outer and inner weight:

.. code-block:: yaml

    input_cfg:
      - type: group
        weight: 0.7
        tags:
          task: asr
        input_cfg:
          - type: nemo_tarred
            manifest_filepath: /path/to/asr1/manifest__OP_0..512_CL_.json
            tarred_audio_filepath: /path/to/tarred_audio/asr1/audio__OP_0..512_CL_.tar
            weight: 0.6
            tags:
              source_lang: en
              target_lang: en
          - type: nemo_tarred
            manifest_filepath: /path/to/asr2/manifest__OP_0..512_CL_.json
            tarred_audio_filepath: /path/to/asr2/tarred_audio/audio__OP_0..512_CL_.tar
            weight: 0.4
            tags:
              source_lang: pl
              target_lang: pl
      - type: group
        weight: 0.3
        tags:
          task: ast
        input_cfg:
          - type: nemo_tarred
            manifest_filepath: /path/to/ast1/manifest__OP_0..512_CL_.json
            tarred_audio_filepath: /path/to/ast1/tarred_audio/audio__OP_0..512_CL_.tar
            weight: 0.2
            tags:
              source_lang: en
              target_lang: pl
          - type: nemo_tarred
            manifest_filepath: /path/to/ast2/manifest__OP_0..512_CL_.json
            tarred_audio_filepath: /path/to/ast2/tarred_audio/audio__OP_0..512_CL_.tar
            weight: 0.8
            tags:
              source_lang: pl
              target_lang: en

Configuring multimodal dataloading
-----------------------------------

Our configuration format supports specifying data sources from other modalities than just audio.
At this time, this support is extended to audio and text modalities. We provide the following parser types:

**Raw text files.** Simple text files where each line is an individual text example. This can represent standard language modeling data.
This parser is registered under ``type: txt``.

Data format examples::

    # file: document_0.txt
    This is a language modeling example.
    Wall Street is expecting major news tomorrow.

    # file: document_1.txt
    Invisible bats have stormed the city.
    What an incredible event!

Dataloading configuration example::

    input_cfg:
      - type: txt
        paths: /path/to/document_{0..1}.txt
        language: en  # optional

Python object example::

    from nemo.collections.common.data.lhotse.text_adapters import TextExample

    example = TextExample(
        text="This is a language modeling example.",
        language="en",  # optional
    )

Python dataloader instantiation example::

    from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config

    dl = get_lhotse_dataloader_from_config({
            "input_cfg": [
                {"type": "txt", "paths": "/path/to/document_{0..1}.txt", "language": "en"},
            ],
            "use_multimodal_dataloading": True,
            "batch_size": 4,
        },
        global_rank=0,
        world_size=1,
        dataset=MyDatasetClass(),  # converts CutSet -> dict[str, Tensor]
        tokenizer=my_tokenizer,
    )

**Raw text file pairs.** Pairs of raw text files with corresponding lines. This can represent machine translation data.
This parser is registered under ``type: txt_pair``.

Data format examples::

    # file: document_en_0.txt
    This is a machine translation example.
    Wall Street is expecting major news tomorrow.

    # file: document_pl_0.txt
    To jest przykład tłumaczenia maszynowego.
    Wall Street spodziewa się jutro ważnych wiadomości.

Dataloading configuration example::

    input_cfg:
      - type: txt_pair
        source_path: /path/to/document_en_{0..N}.txt
        target_path: /path/to/document_pl_{0..N}.txt
        source_language: en  # optional
        target_language: pl  # optional

Python object example::

    from nemo.collections.common.data.lhotse.text_adapters import SourceTargetTextExample

    example = SourceTargetTextExample(
        source=TextExample(
            text="This is a language modeling example.",
            language="en",  # optional
        ),
        target=TextExample(
            text="To jest przykład tłumaczenia maszynowego.",
            language="pl",  # optional
        ),
    )

Python dataloader instantiation example::

    from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config

    dl = get_lhotse_dataloader_from_config({
            "input_cfg": [
                {
                    "type": "txt_pair",
                    "source_path": "/path/to/document_en_{0..N}.txt",
                    "target_path": "/path/to/document_pl_{0..N}.txt",
                    "source_language": "en"
                    "target_language": "en"
                },
            ],
            "use_multimodal_dataloading": True,
            "prompt_format": "t5nmt",
            "batch_size": 4,
        },
        global_rank=0,
        world_size=1,
        dataset=MyDatasetClass(),  # converts CutSet -> dict[str, Tensor]
        tokenizer=my_tokenizer,
    )

**NeMo multimodal conversations.** A JSON-Lines (JSONL) file that defines multi-turn conversations with mixed text and audio turns.
This parser is registered under ``type: multimodal_conversation``.

Data format examples::

    # file: chat_0.jsonl
    {"id": "conv-0", "conversations": [{"from": "user", "value": "speak to me", "type": "text"}, {"from": "assistant": "value": "/path/to/audio.wav", "duration": 17.1, "type": "audio"}]}
    {"id": "conv-1", "conversations": [{"from": "user", "value": "speak to me", "type": "text"}, {"from": "assistant": "value": "/path/to/audio.wav", "duration": 5, "offset": 17.1, "type": "audio"}]}

Dataloading configuration example::

    token_equivalent_duration: 0.08
    input_cfg:
      - type: multimodal_conversation
        manifest_filepath: /path/to/chat_{0..N}.jsonl
        audio_locator_tag: [audio]

Python object example::

    from lhotse import Recording
    from nemo.collections.common.data.lhotse.text_adapters import MultimodalConversation, TextTurn, AudioTurn

    conversation = NeMoMultimodalConversation(
        id="conv-0",
        turns=[
            TextTurn(value="speak to me", role="user"),
            AudioTurn(cut=Recording.from_file("/path/to/audio.wav").to_cut(), role="assistant", audio_locator_tag="[audio]"),
        ],
        token_equivalent_duration=0.08,  # this value will be auto-inserted by the dataloader
    )

Python dataloader instantiation example::

    from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config

    dl = get_lhotse_dataloader_from_config({
            "input_cfg": [
                {
                    "type": "multimodal_conversation",
                    "manifest_filepath": "/path/to/chat_{0..N}.jsonl",
                    "audio_locator_tag": "[audio]",
                },
            ],
            "use_multimodal_dataloading": True,
            "token_equivalent_duration": 0.08,
            "prompt_format": "llama2",
            "batch_size": 4,
        },
        global_rank=0,
        world_size=1,
        dataset=MyDatasetClass(),  # converts CutSet -> dict[str, Tensor]
        tokenizer=my_tokenizer,
    )

**Dataloading and bucketing of text and multimodal data.** When dataloading text or multimodal data, pay attention to the following config options (we provide example values for convenience):

* ``use_multimodal_sampling: true`` tells Lhotse to switch from measuring audio duration to measuring token counts; required for text.

* ``prompt_format: "prompt-name"`` will apply a specified PromptFormatter during data sampling to accurately reflect its token counts.

* ``measure_total_length: true`` customizes length measurement for decoder-only and encoder-decoder models. Decoder-only models consume a linear sequence of context + answer, so we should measure the total length (``true``). On the other hand, encoder-decoder models deal with two different sequence lengths: input (context) sequence length for the encoder, and output (answer) sequence length for the decoder. For such models set this to ``false``.

* ``min_tokens: 1``/``max_tokens: 4096`` filters examples based on their token count (after applying the prompt format).

* ``min_tpt: 0.1``/``max_tpt: 10`` filter examples based on their output-token-per-input-token-ratio. For example, a ``max_tpt: 10`` means we'll filter every example that has more than 10 output tokens per 1 input token. Very useful for removing sequence length outliers that lead to OOM. Use ``estimate_token_bins.py`` to view token count distributions for calbirating this value.

* (multimodal-only) ``token_equivalent_duration: 0.08`` is used to be able to measure audio examples in the number of "tokens". For example, if we're using fbank with 0.01s frame shift and an acoustic model that has a subsampling factor of 0.08, then a reasonable setting for this could be 0.08 (which means every subsampled frame counts as one token). Calibrate this value to fit your needs.

**Text/multimodal bucketing and OOMptimizer.** Analogous to bucketing for audio data, we provide two scripts to support efficient bucketing:

* ``scripts/speech_llm/estimate_token_bins.py`` which estimates 1D or 2D buckets based on the input config, tokenizer, and prompt format. It also estimates input/output token count distribution and suggested ``max_tpt`` (token-per-token) filtering values.

* (experimental) ``scripts/speech_llm/oomptimizer.py`` which works with SALM/BESTOW GPT/T5 models and estimates the optimal ``bucket_batch_size`` for a given model config and bucket bins value. Given the complexity of Speech LLM some configurations may not be supported yet at the time of writing (e.g., model parallelism).

To enable bucketing, set ``batch_size: null`` and use the following options:

* ``use_bucketing: true``

* ``bucket_duration_bins`` - the output of ``estimate_token_bins.py``. If ``null``, it will be estimated at the start of training at the cost of some run time (not recommended).

* (oomptimizer-only) ``bucket_batch_size`` - the output of OOMptimizer.

* (non-oomptimizer-only) ``batch_tokens`` is the maximum number of tokens we want to find inside a mini-batch. Similarly to ``batch_duration``, this number does consider padding tokens too, therefore enabling bucketing is recommended to maximize the ratio of real vs padding tokens. Note that it's just a heuristic for determining the optimal batch sizes for different buckets, and may be less efficient than using OOMptimizer.

* (non-oomptimizer-only) ``quadratic_factor`` is a quadratic penalty to equalize the GPU memory usage between buckets of short and long sequence lengths for models with quadratic memory usage. It is only a heuristic and may not be as efficient as using OOMptimizer.

**Joint dataloading of text/audio/multimodal data.** The key strength of this approach is that we can easily combine audio datasets and text datasets,
and benefit from every other technique we described in this doc, such as: dynamic data mixing, data weighting, dynamic bucketing, and so on.

This approach is described in the `EMMeTT`_ paper. There's also a notebook tutorial called Multimodal Lhotse Dataloading. We construct a separate sampler (with its own batching settings) for each modality,
and specify how the samplers should be fused together via the option ``sampler_fusion``:

* ``sampler_fusion: "round_robin"`` will iterate single sampler per step, taking turns. For example: step 0 - audio batch, step 1 - text batch, step 2 - audio batch, etc.

* ``sampler_fusion: "randomized_round_robin"`` is similar, but at each chooses a sampler randomly using ``sampler_weights: [w0, w1]`` (weights can be unnormalized).

* ``sampler_fusion: "zip"`` will draw a mini-batch from each sampler at every step, and merge them into a single ``CutSet``. This approach combines well with multimodal gradient accumulation (run forward+backward for one modality, then the other, then the update step).

.. _EMMeTT: https://arxiv.org/abs/2409.13523

Example. Combine an ASR (audio-text) dataset with an MT (text-only) dataset so that mini-batches have some examples from both datasets:

.. code-block:: yaml

    model:
      ...
      train_ds:
        multi_config: True,
        sampler_fusion: zip
        shuffle: true
        num_workers: 4

        audio:
          prompt_format: t5nmt
          use_bucketing: true
          min_duration: 0.5
          max_duration: 30.0
          max_tps: 12.0
          bucket_duration_bins: [[3.16, 10], [3.16, 22], [5.18, 15], ...]
          bucket_batch_size: [1024, 768, 832, ...]
          input_cfg:
            - type: nemo_tarred
              manifest_filepath: /path/to/manifest__OP_0..512_CL_.json
              tarred_audio_filepath: /path/to/tarred_audio/audio__OP_0..512_CL_.tar
              weight: 0.5
              tags:
                context: "Translate the following to English"

        text:
          prompt_format: t5nmt
          use_multimodal_sampling: true
          min_tokens: 1
          max_tokens: 256
          min_tpt: 0.333
          max_tpt: 3.0
          measure_total_length: false
          use_bucketing: true
          bucket_duration_bins: [[10, 4], [10, 26], [15, 10], ...]
          bucket_batch_size: [512, 128, 192, ...]
          input_cfg:
            - type: txt_pair
              source_path: /path/to/en__OP_0..512_CL_.txt
              target_path: /path/to/pl__OP_0..512_CL_.txt
              source_language: en
              target_language: pl
              weight: 0.5
              tags:
                question: "Translate the following to Polish"

.. caution:: We strongly recommend to use multiple shards for text files as well so that different nodes and dataloading workers are able to randomize the order of text iteration. Otherwise, multi-GPU training has a high risk of duplication of text examples.

Pre-computing bucket duration bins
------------------------------------

We recommend to pre-compute the bucket duration bins in order to accelerate the start of the training -- otherwise, the dynamic bucketing sampler will have to spend some time estimating them before the training starts.
The following script may be used:

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins.py -b 30 manifest.json

    # The script's output:
    Use the following options in your config:
            num_buckets=30
            bucket_duration_bins=[1.78,2.34,2.69,...
    <other diagnostic information about the dataset>

For multi-dataset setups, one may provide a dataset config directly:

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins.py -b 30 input_cfg.yaml

    # The script's output:
    Use the following options in your config:
            num_buckets=30
            bucket_duration_bins=[1.91,3.02,3.56,...
    <other diagnostic information about the dataset>

It's also possible to manually specify the list of data manifests (optionally together with weights):

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins.py -b 30 [[manifest.json,0.7],[other.json,0.3]]

    # The script's output:
    Use the following options in your config:
            num_buckets=30
            bucket_duration_bins=[1.91,3.02,3.56,...
    <other diagnostic information about the dataset>

2D bucketing
-------------

To achieve maximum training efficiency for some classes of models it is necessary to stratify the sampling
both on the input sequence lengths and the output sequence lengths.
One such example are attention encoder-decoder models, where the overall GPU memory usage can be factorized
into two main components: input-sequence-length bound (encoder activations) and output-sequence-length bound
(decoder activations).
Classical bucketing techniques only stratify on the input sequence length (e.g. duration in speech),
which leverages encoder effectively but leads to excessive padding on on decoder's side.

To amend this we support a 2D bucketing technique which estimates the buckets in two stages.
The first stage is identical to 1D bucketing, i.e. we determine the input-sequence bucket bins so that
every bin holds roughly an equal duration of audio.
In the second stage, we use a tokenizer and optionally a prompt formatter (for prompted models) to
estimate the total number of tokens in each duration bin, and sub-divide it into several sub-buckets,
where each sub-bucket again holds roughly an equal number of tokens.

To run 2D bucketing with 30 buckets sub-divided into 5 sub-buckets each (150 buckets total), use the following script:

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins_2d.py \
        --tokenizer path/to/tokenizer.model \
        --buckets 30 \
        --sub-buckets 5 \
        input_cfg.yaml

    # The script's output:
    Use the following options in your config:
            use_bucketing=1
            num_buckets=30
            bucket_duration_bins=[[1.91,10],[1.91,17],[1.91,25],...
    The max_tps setting below is optional, use it if your data has low quality long transcript outliers:
            max_tps=[13.2,13.2,11.8,11.8,...]

Note that the output in ``bucket_duration_bins`` is a nested list, where every bin specifies
the maximum duration and the maximum number of tokens that go into the bucket.
Passing this option to Lhotse dataloader will automatically enable 2D bucketing.

Note the presence of ``max_tps`` (token-per-second) option.
It is optional to include it in the dataloader configuration: if you do, we will apply an extra filter
that discards examples which have more tokens per second than the threshold value.
The threshold is determined for each bucket separately based on data distribution, and can be controlled
with the option ``--token_outlier_threshold``.
This filtering is useful primarily for noisy datasets to discard low quality examples / outliers.

We also support aggregate tokenizers for 2D bucketing estimation:

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins_2d.py \
        --tokenizer path/to/en/tokenizer.model path/to/pl/tokenizer1.model \
        --langs en pl \
        --buckets 30 \
        --sub-buckets 5 \
        input_cfg.yaml

To estimate 2D buckets for a prompted model such as Canary-1B, provide prompt format name and an example prompt.
For Canary-1B, we'll also provide the special tokens tokenizer. Example:

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins_2d.py \
        --prompt-format canary \
        --prompt "[{'role':'user','slots':{'source_lang':'en','target_lang':'de','task':'ast','pnc':'yes'}}]" \
        --tokenizer path/to/spl_tokens/tokenizer.model path/to/en/tokenizer.model path/to/de/tokenizer1.model \
        --langs spl_tokens en de \
        --buckets 30 \
        --sub-buckets 5 \
        input_cfg.yaml

Pushing GPU utilization to the limits with bucketing and OOMptimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default approach of specifying a ``batch_duration``, ``bucket_duration_bins`` and ``quadratic_duration``
is quite flexible, but is not maximally efficient. We observed that in practice it often leads to under-utilization
of GPU memory and compute for most buckets (especially those with shorter durations).
While it is impossible to estimate GPU memory usage up-front, we can determine it empirically with a bit of search.

OOMptimizer is an approach that given a NeMo model, optimizer, and a list of buckets (1D or 2D)
estimates the maximum possible batch size to use for each bucket.
It performs a binary search over batch sizes that succeed or lead to CUDA OOM until convergence.
We find that the resulting bucketing batch size profiles enable full GPU utilization in training,
while it only takes a couple of minutes to complete the search.

In order to run OOMptimizer, you only need the bucketing bins (from previous sections) and a model configuration:

.. code-block:: bash

    $ python scripts/speech_recognition/oomptimizer.py \
        --config-path fast-conformer_aed.yaml \
        --module-name nemo.collections.asr.models.EncDecMultiTaskModel \
        --buckets '[[3.975,30],[3.975,48],[4.97,37],...]'

    # The script's output:
    <output logs from the search>
    The final profile is:
            bucket_duration_bins=[[3.975,30],[3.975,48],...]
            bucket_batch_size=[352,308,280,...]
            max_tps=12.0
            max_duration=40.0

Use the resulting options in your training configuration (typically under namespace ``model.train_ds``) to apply the profile.

It's also possible to run OOMptimizer using a pretrained model's name and bucket bins corresponding
to your fine-tuning data:

.. code-block:: bash

    $ python scripts/speech_recognition/oomptimizer.py \
        --pretrained-name nvidia/canary-1b \
        --buckets '[2.0,3.1,5.6,6.6,...]'

Note that your training script can perform some additional actions using GPU RAM that cannot be anticipated by the OOMptimizer.
By default, we let the script use up to 90% of GPU's RAM for this estimation to account for that.
In the unlikely case you run into an OutOfMemoryError during training, you can try re-estimating the profile with the option ``--memory-fraction 0.75`` (or another value) that will further cap OOMptimizer's available GPU RAM.

Seeds and randomness
---------------------

In Lhotse dataloading configuration we have two parameters controlling randomness: ``seed`` and ``shard_seed``.
Both of them can be either set to a fixed number, or one of two string options ``"randomized"`` and ``"trng"``.
Their roles are:

* ``seed`` is the base random seed, and is one of several factors used to initialize various RNGs participating in dataloading.

* ``shard_seed`` controls the shard randomization strategy in distributed data parallel setups when using sharded tarred datasets.

Below are the typical examples of configuration with an explanation of the expected outcome.

Case 1 (default): ``seed=<int>`` and ``shard_seed="trng"``:

* The ``trng`` setting discards ``seed`` and causes the actual random seed to be drawn using OS's true RNG. Each node/GPU/dataloading worker draws its own unique random seed when it first needs it.

* Each node/GPU/dataloading worker yields data in a different order (no mini-batch duplication).

* On each training script run, the order of dataloader examples are **different**.

* Since the random seed is unpredictable, the exact dataloading order is not replicable.

Case 2: ``seed=<int>`` and ``shard_seed="randomized"``:

* The ``randomized`` setting uses ``seed`` along with DDP ``rank`` and dataloading ``worker_id`` to set a unique but deterministic random seed in each dataloading process across all GPUs.

* Each node/GPU/dataloading worker yields data in a different order (no mini-batch duplication).

* On each training script run, the order of dataloader examples are **identical** as long as ``seed`` is the same.

* This setup guarantees 100% dataloading reproducibility.

* Resuming training without changing of the ``seed`` value will cause the model to train on data it has already seen. For large data setups, not managing the ``seed`` may cause the model to never be trained on a majority of data. This is why this mode is not the default.

* If you're combining DDP with model parallelism techniques (Tensor Parallel, Pipeline Parallel, etc.) you need to use ``shard_seed="randomized"``. Using ``"trng"`` will cause different model parallel ranks to desynchronize and cause a deadlock.

* Generally the seed can be managed by the user by providing a different value each time the training script is launched. For example, for most models the option to override would be ``model.train_ds.seed=<value>``. If you're launching multiple tasks queued one after another on a grid system, you can generate a different random seed for each task, e.g. on most Unix systems ``RSEED=$(od -An -N4 -tu4 < /dev/urandom | tr -d ' ')`` would generate a random uint32 number that can be provided as the seed.

Other, more exotic configurations:

* With ``shard_seed=<int>``, all dataloading workers will yield the same results. This is only useful for unit testing and maybe debugging.

* With ``seed="trng"``, the base random seed itself will be drawn using a TRNG. It will be different on each GPU training process. This setting is not recommended.

* With ``seed="randomized"``, the base random seed is set to Python's global RNG seed. It might be different on each GPU training process. This setting is not recommended.
